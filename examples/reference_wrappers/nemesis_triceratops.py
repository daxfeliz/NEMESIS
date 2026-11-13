#!/usr/bin/env python3
"""
nemesis_triceratops.py

Statistical validation of NEMESIS transiting planet candidates with
TRICERATOPS (Giacalone & Dressing 2020; Giacalone et al. 2021).

Self-contained. The only required third-party packages are numpy, pandas,
matplotlib, astropy and triceratops. NOTHING from NEMESIS is imported: the
one NEMESIS helper needed, ``readNDarr``, is eleven lines of pickle handling
and is reimplemented here verbatim, so the same file runs unmodified against
``NEMESIS_pipeline.py`` on the HPC and ``NEMESIS_pipeline_new_temp_5.py``
locally without caring which module name is in scope.

What it does, for one star-sector:

  1. Locate and read the NEMESIS bestfit table and light curve.
  2. Read the persisted aperture mask and convert it from cutout indices to
     the absolute TESS CCD [column, row] pairs TRICERATOPS expects.
  3. Phase-fold and trim the detrended light curve.
  4. Obtain or reuse a cached TRILEGAL background population.
  5. Compute dilution-corrected depths against the true NEMESIS aperture.
  6. Run N independent calc_probs realizations.
  7. Write figures, per-trial and per-scenario tables, and append summary
     columns to the bestfit CSV.

Usage, library:

    from nemesis_triceratops import run_triceratops_on_nemesis

    res = run_triceratops_on_nemesis(
        tic_id=270380593, sector=4,
        report_dir=settings['saveReportpath_TLS'],
        savelcpath=settings['savelcpath'],
        hdu=hdu, n_trials=20)

Usage, command line:

    python nemesis_triceratops.py --tic 270380593 --sector 4 \\
        --report-dir /path/NEMESIS_FFI_TLS_Report/ \\
        --savelcpath /path/NEMESIS_FFI_LCs/ \\
        --hdu /path/cache/TESSCut/TIC_270380593/tess-s0004-...-11x11_astrocut.fits \\
        --posterior-path /path/TIC_270380593_Sector_0004_jaxoplanet_posterior.csv \\
        --lk-cache-dir /path/triceratops/cache/TIC_270380593/ \\
        --n-trials 20

Author: D. Feliz, NEMESIS II
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import pickle
import re
import ssl
import sys
import time
import urllib.request
import warnings
from contextlib import contextmanager
from typing import Any, Iterator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits

from triceratops.funcs import save_trilegal

__version__ = "1.3.0"

__all__ = [
    "read_ndarray",
    "find_nemesis_bestfit",
    "find_nemesis_lightcurve",
    "read_jaxoplanet_posterior",
    "open_hdu",
    "render_to_file",
    "trilegal_ssl_workaround",
    "VALIDATION_EXPTIME_MAX_D",
    "RP_FLAG_RE",
    "nemesis_mask_to_ccd_pixels",
    "prepare_triceratops_lightcurve",
    "nemesis_to_triceratops_inputs",
    "N_DURATIONS_DEFAULT",
    "OUTCOME_COMPUTED",
    "OUTCOME_CANNOT_HOST",
    "TRIAL_PROBS_COLUMNS",
    "read_nemesis_fcr",
    "convert_to_aperture_units",
    "target_host_check",
    "check_trial_probs",
    "count_negative_flux",
    "SCENARIO_STATS_COLUMNS",
    "summarize_scenarios",
    "OUTCOME_POORLY_VETTED",
    "OUTCOME_ALL_NAN",
    "STAR_PARAM_COLUMNS",
    "classify_triceratops",
    "target_star_params",
    "summarize_trials",
    "save_trilegal_when_ready",
    "pdf_page_size_px",
    "run_triceratops_on_nemesis",
    "compose_triceratops_page",
    "compose_triceratops_report_page",
    "append_png_to_pdf",
]

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

TRILEGAL_HOST = "stev.oapd.inaf.it"

BUILTIN_TRUE = ("true", "t", "yes", "y", "1")

#: Bestfit columns searched for the planet radius, in order.
RP_COLUMN_CANDIDATES: tuple[str, ...] = (
    "Planet Radius [RE]",
    "TLS Planet Radius [RE]",
    "BLS Planet Radius [RE]",
    "Planet Radius (Earth Radii)",
)

#: Bestfit columns searched for the NEMESIS flux contamination ratio.
#: NEMESIS defines it as c = 1 - F_target / F_total (calc_flux_contamination),
#: so the TRICERATOPS target fluxratio should equal 1 - c.
FCR_COLUMN_CANDIDATES: tuple[str, ...] = (
    "flux_contamination_ratio",
    "Flux Contamination Ratio",
    "flux_contam",
    "FCR",
    "FLUX_CONTAMINATION_RATIO",
)

#: (extension index, column keyword, row keyword) tried in order when
#: resolving the cutout-to-CCD pixel offset.
CCD_OFFSET_KEYWORDS: tuple[tuple[int, str, str], ...] = (
    (2, "CRVAL1P", "CRVAL2P"),
    (1, "CRVAL1P", "CRVAL2P"),
    (1, "1CRV4P", "2CRV4P"),
    (1, "1CRV5P", "2CRV5P"),
)

#: Exposure time in DAYS above which the "validated planet" tier is
#: withheld. Giacalone et al. (2021) Section 4.2 reran TRICERATOPS on
#: 30-minute FFI light curves of the same 68 TOIs they had analyzed at
#: 2-minute cadence and found no region of the FPP-NFPP plane in which
#: planets could be confidently validated. The likely-planet region
#: (FPP < 0.5, NFPP < 1e-3) and the nearby-false-positive region
#: (NFPP > 0.1) both survived. 10 minutes sits between the 2-minute
#: cadence where validation was demonstrated and the 30-minute cadence
#: where it was not.
VALIDATION_EXPTIME_MAX_D: float = 10.0 / 1440.0


#: Radius in Earth radii above which a candidate is flagged in
#: ``TRICERATOPS_Rp_flag_gt_8``. A flag, not a ceiling (09-29 decision):
#: every reconciled PC is evaluated regardless of radius.
RP_FLAG_RE: float = 8

#: jaxoplanet posterior columns that supply the ephemeris: posterior medians
#: of period (days), mid-transit time (BTJD) and duration (hours).
POSTERIOR_EPHEMERIS_COLUMNS: tuple[str, str, str] = (
    "MCMC Period [d]",
    "MCMC T0 [BTJD]",
    "MCMC Dur [hrs]",
)

#: Fractional period difference between the jaxoplanet posterior and the
#: search bestfit above which a warning is raised. The jaxoplanet period
#: prior is centered on the search period, so a larger difference points to
#: a mismatched file rather than a real refinement.
POSTERIOR_PERIOD_TOL: float = 0.01

#: Fold half-width in transit durations, for every target (10-14-2026).
#: At +/-5, single cadences 3.9 and 4.7 durations from mid-transit pushed
#: every scenario's evidence for TIC 138819293 S22 and TIC 404664390 S13
#: below double-precision range (FPP NaN). +/-3 is the widest whole-number
#: window that excludes both, and the 10-14 pre-check flagged no target at
#: +/-3 that it did not also flag at +/-5.
N_DURATIONS_DEFAULT: int = 3

#: TRICERATOPS_outcome values.
OUTCOME_COMPUTED: str = "computed"
OUTCOME_CANNOT_HOST: str = ("target cannot make the dip under triceratops' "
                            "light model")

#: Columns of target.probs in triceratops 1.0.20, plus the trial index.
TRIAL_PROBS_COLUMNS: tuple[str, ...] = (
    "ID", "scenario", "M_s", "R_s", "P_orb", "inc", "b", "ecc", "w", "R_p",
    "M_EB", "R_EB", "prob", "trial")

#: Columns of the per-type scenario table, the same as 1.2.0's, so the
#: scenarios CSV and the TRI_prob_* columns keep their layout.
SCENARIO_STATS_COLUMNS: tuple[str, ...] = (
    "scenario", "prob_median", "prob_mean", "prob_std", "n_trials")

#: TRICERATOPS_outcome values added in 1.3.0 part 2 (10-15-2026): every
#: trial NaN with a point below zero in the handed light curve, and every
#: trial NaN without one.
OUTCOME_POORLY_VETTED: str = "poorly vetted"
OUTCOME_ALL_NAN: str = "every trial NaN, no below-zero points"

#: Target columns triceratops needs before calc_probs; without any one it
#: stops early and returns equal placeholder probabilities.
STAR_PARAM_COLUMNS: tuple[str, ...] = ("mass", "rad", "Teff", "plx")

# --------------------------------------------------------------------------
# NEMESIS product I/O, reimplemented so nothing from NEMESIS is imported
# --------------------------------------------------------------------------

def save_trilegal_when_ready(
    output_url: str,
    tic_id: int,
    min_rows: int = 50,
    max_wait_s: float = 900.0,
    poll_interval_s: float = 20.0,
    verbose: bool = True,
) -> str | None:
    """
    Wait for a submitted TRILEGAL job to finish, then save its output.

    TRILEGAL is a submit-then-poll service: ``tr.target()`` submits a job and
    returns an output URL immediately, but the file at that URL is empty
    until the simulation completes. Calling ``save_trilegal`` right away
    raises ``pandas.errors.EmptyDataError`` from its completion check at
    ``funcs.py`` line 318. Working interactively hides this, because minutes
    pass between the two notebook cells; a script calling them back to back
    hits the race every time.

    This polls the URL until it returns at least ``min_rows`` rows, then
    hands off to ``triceratops.funcs.save_trilegal``. Both the polling and
    the save run inside :func:`trilegal_ssl_workaround`, because the INAF
    host serves an incomplete certificate chain.

    Parameters
    ----------
    output_url : str
        ``target.trilegal_url``, as returned by ``query_TRILEGAL``.
    tic_id : int
        TIC identifier. ``save_trilegal`` takes this as its second argument
        and derives the output filename ``{tic_id}_TRILEGAL.csv`` from it.
    min_rows : int
        Row count below which the job is considered still running. A
        completed population file has thousands of rows; a queued job
        returns an empty document or a short status page.
    max_wait_s : float
        Give up after this many seconds. TRILEGAL usually finishes in under
        two minutes for a small field, but queues.
    poll_interval_s : float
        Seconds between polls. Do not lower this much: it is a shared
        public service.
    verbose : bool
        Print each poll.

    Returns
    -------
    str or None
        The filename ``save_trilegal`` wrote, or None if it returned
        nothing.

    Raises
    ------
    TimeoutError
        If the job has not produced ``min_rows`` rows within ``max_wait_s``.
    ValueError
        If ``output_url`` is None or empty.

    Examples
    --------
    >>> fname = save_trilegal_when_ready(target.trilegal_url, 270380593)
    """
    from triceratops.funcs import save_trilegal

    if not output_url:
        raise ValueError(
            "output_url is empty; target.trilegal_url was never set, so the "
            "TRILEGAL query did not submit successfully"
        )

    deadline = time.time() + float(max_wait_s)
    attempt = 0

    with trilegal_ssl_workaround(verbose=verbose):
        while True:
            attempt += 1
            n_rows = 0
            err: str | None = None
            try:
                probe = pd.read_csv(output_url, header=None)
                n_rows = len(probe)
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"

            if n_rows >= min_rows:
                if verbose:
                    print(f"  TRILEGAL ready after {attempt} poll(s), "
                          f"{n_rows} rows")
                return save_trilegal(output_url, tic_id)

            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(
                    f"TRILEGAL job at {output_url} produced {n_rows} rows "
                    f"(need {min_rows}) after {max_wait_s:.0f} s. Last "
                    f"probe: {err or 'no error, file too short'}. The job "
                    f"may be queued; retry later, or open the URL in a "
                    f"browser to check."
                )

            if verbose:
                print(f"  poll {attempt}: {n_rows} rows"
                      f"{' (' + err + ')' if err else ''}, waiting "
                      f"{poll_interval_s:.0f} s, "
                      f"{remaining:.0f} s left")
            time.sleep(min(poll_interval_s, max(remaining, 1.0)))

def read_ndarray(path: str, filename: str) -> Any:
    """
    Deserialize a pickle written by the NEMESIS ``saveNDarr`` helper.

    Byte-for-byte equivalent to ``NEMESIS_pipeline.readNDarr``, reproduced
    here so this module has no NEMESIS import and therefore does not care
    whether the pipeline module is named ``NEMESIS_pipeline`` (HPC) or
    ``NEMESIS_pipeline_new_temp_5`` (laptop).

    Note the string concatenation rather than ``os.path.join``: NEMESIS
    builds the path as ``path + filename + '.pkl'``, so ``path`` must end
    with a separator. Callers here normalize it first.

    Parameters
    ----------
    path : str
        Directory containing the file, ending with a path separator.
    filename : str
        Base filename without extension, e.g.
        ``'TIC_270380593_Sector_0004_pix_mask'``.

    Returns
    -------
    object
        The deserialized object, typically a numpy array of shape
        ``(ny, nx)`` for a mask or ``(T, ny, nx)`` for an image cube.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.

    Examples
    --------
    >>> m = read_ndarray(savelcpath, "TIC_270380593_Sector_0004_pix_mask")
    """
    with open(path + filename + ".pkl", "rb") as pkl_file:
        return pickle.load(pkl_file)


def find_nemesis_bestfit(
    report_dir: str,
    tic_id: int,
    sector: int | None = None,
    search: str = "TLS",
    verbose: bool = True,
) -> list[str]:
    """
    Locate NEMESIS bestfit CSVs for one TIC, without NEMESIS helpers.

    Replaces ``recursive_search``. Matches the filename convention written
    by NEMESIS: ``TIC_{ID}_{sector_tag}_{BLS|TLS}_bestfit.csv``, where
    ``sector_tag`` is ``Sector_{S:04d}`` or ``Multi_Sector``.

    The ``{search}`` token is required in the pattern so a TLS request never
    silently returns the BLS file sitting beside it.

    Parameters
    ----------
    report_dir : str
        Directory to search, normally ``settings['saveReportpath_TLS']``.
        Searched recursively, which also covers the
        ``injections/Period_.../`` subtree.
    tic_id : int
        TIC identifier.
    sector : int, optional
        Restrict to one sector. ``None`` returns every sector found,
        including any ``Multi_Sector`` file.
    search : str
        ``'TLS'`` or ``'BLS'``.
    verbose : bool
        Print what was found.

    Returns
    -------
    list of str
        Sorted absolute paths. Empty if nothing matched.

    Examples
    --------
    >>> hits = find_nemesis_bestfit(report_dir, 270380593, sector=4)
    """
    search = search.upper()
    if sector is None:
        pattern = f"TIC_{tic_id}_*_{search}_bestfit.csv"
    else:
        pattern = f"TIC_{tic_id}_Sector_{int(sector):04d}_{search}_bestfit.csv"

    hits = sorted(
        os.path.abspath(p)
        for p in glob.glob(os.path.join(report_dir, "**", pattern),
                           recursive=True)
    )
    if verbose:
        print(f"bestfit search: {pattern}")
        print(f"  under       : {report_dir}")
        print(f"  found       : {len(hits)}")
        for h in hits:
            print(f"    {h}")
    return hits


def find_nemesis_lightcurve(
    savelcpath: str,
    tic_id: int,
    sector: int,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load the final NEMESIS light curve table for one star-sector from disk.

    Reads ``TIC_{ID}_Sector_{S:04d}_final_LC.csv`` from ``savelcpath``. Use
    this when driving from archived products rather than from a live
    ``full_pipeline`` return value.

    Parameters
    ----------
    savelcpath : str
        Light curve directory, normally ``settings['savelcpath']``.
    tic_id : int
        TIC identifier.
    sector : int
        TESS sector.
    verbose : bool
        Print the resolved path and row count.

    Returns
    -------
    pd.DataFrame
        Columns include ``Time``, ``SAP Flux``, ``SAP Error``, ``PLD Flux``,
        ``PLD Error``, ``PLD Model``, ``Detrended Flux``,
        ``Detrended Error``.

    Raises
    ------
    FileNotFoundError
        If the file is absent.

    Examples
    --------
    >>> LC_df = find_nemesis_lightcurve(savelcpath, 270380593, 4)
    """
    savelcpath = os.path.join(savelcpath, "")
    fname = f"TIC_{tic_id}_Sector_{int(sector):04d}_final_LC.csv"
    full = savelcpath + fname
    if not os.path.isfile(full):
        raise FileNotFoundError(
            f"light curve not found: {full}. Pass LC_df explicitly, or check "
            f"that the run completed and cleanup did not remove it."
        )
    df = pd.read_csv(full)
    if verbose:
        print(f"light curve   : {full}  ({len(df)} rows)")
    return df

def read_jaxoplanet_posterior(
    posterior_path: str,
    tic_id: int,
    sector: int,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Read the transit ephemeris from one jaxoplanet posterior summary CSV.

    The file is the single-row ``*_jaxoplanet_posterior.csv`` written by
    ``nemesis_jaxoplanet_refit.refine_transit_parameters``. The posterior
    medians in :data:`POSTERIOR_EPHEMERIS_COLUMNS` supply period,
    mid-transit time and duration. ``MCMC Planet Radius [RE]`` and
    ``MCMC converged`` are read when present.

    Depth is deliberately NOT read. ``MCMC Depth [ppt]`` is (Rp/Rs)^2, a
    geometric depth without limb darkening, while TRICERATOPS needs the
    observed dip depth in the light curve it receives to decide which
    neighbors are bright enough to host the signal. The caller keeps the
    search depth measured on that light curve.

    Identity is enforced as for the bestfit: TIC and sector are parsed from
    the file name and must match, and the ``TIC_ID`` and ``Sector`` columns
    are checked whenever they hold finite values. Nothing from NEMESIS or
    jaxoplanet is imported.

    Parameters
    ----------
    posterior_path : str
        Path to the posterior CSV.
    tic_id : int
        TIC identifier the posterior must belong to.
    sector : int
        Sector the posterior must belong to.
    verbose : bool
        Print the values read.

    Returns
    -------
    dict
        ``P_orb`` (days), ``T0`` (BTJD), ``duration_days``, ``Rp_RE``
        (NaN when absent), ``converged`` (bool, or None when the column is
        absent or empty), ``path`` (absolute path).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If TIC or sector cannot be parsed from the name, disagree with the
        expected values, or the ephemeris values are not finite and
        positive.
    KeyError
        If an ephemeris column is absent.

    Examples
    --------
    >>> post = read_jaxoplanet_posterior(
    ...     "/path/TIC_259377017_Sector_0004_jaxoplanet_posterior.csv",
    ...     tic_id=259377017, sector=4)
    >>> post["P_orb"], post["duration_days"]
    """
    if not os.path.isfile(posterior_path):
        raise FileNotFoundError(
            f"jaxoplanet posterior not found: {posterior_path}")

    base = os.path.basename(posterior_path)
    m_tic = re.search(r"TIC[_ ]?(\d+)", base)
    m_sec = re.search(r"Sector[_ ]?(\d+)", base)
    if m_tic is None or m_sec is None:
        raise ValueError(f"cannot parse TIC and sector from {base!r}")
    if int(m_tic.group(1)) != int(tic_id) or int(m_sec.group(1)) != int(sector):
        raise ValueError(
            f"posterior {base} names TIC {m_tic.group(1)} Sector "
            f"{int(m_sec.group(1))}, but the bestfit is TIC {tic_id} Sector "
            f"{sector}. Pairing one signal's ephemeris with another's light "
            f"curve yields a plausible FPP for the wrong signal. Refusing."
        )

    df = pd.read_csv(posterior_path)
    if df.empty:
        raise ValueError(f"posterior file has no rows: {posterior_path}")
    row = df.iloc[0]

    for col, want in (("TIC_ID", int(tic_id)), ("Sector", int(sector))):
        if col not in df.columns:
            continue
        try:
            got = float(row[col])
        except (TypeError, ValueError):
            continue
        if np.isfinite(got) and int(got) != want:
            raise ValueError(
                f"{base}: column {col} = {int(got)} but expected {want}.")

    missing = [c for c in POSTERIOR_EPHEMERIS_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"{base} is missing {missing}. Present: {list(df.columns)[:12]} ...")

    def _num(col: str) -> float:
        if col not in df.columns:
            return float("nan")
        try:
            return float(row[col])
        except (TypeError, ValueError):
            return float("nan")

    period, t0, dur_hr = (_num(c) for c in POSTERIOR_EPHEMERIS_COLUMNS)
    if not (np.isfinite(period) and period > 0 and np.isfinite(t0)
            and np.isfinite(dur_hr) and dur_hr > 0):
        raise ValueError(
            f"{base}: unusable ephemeris (P={period}, T0={t0}, "
            f"dur={dur_hr} hr).")

    rp = _num("MCMC Planet Radius [RE]")

    converged: bool | None = None
    if "MCMC converged" in df.columns and not pd.isna(row["MCMC converged"]):
        converged = str(row["MCMC converged"]).strip().lower() in BUILTIN_TRUE
        if not converged:
            warnings.warn(
                f"{base}: MCMC converged is False. The ephemeris is used, "
                f"but this posterior did not pass the jaxoplanet "
                f"convergence gate.", RuntimeWarning)

    if verbose:
        print(f"posterior     : {base}")
        print(f"  P {period:.7f} d, T0 {t0:.6f} BTJD, dur {dur_hr:.3f} hr, "
              f"Rp {rp:.3f} RE, converged {converged}")

    return {
        "P_orb": float(period),
        "T0": float(t0),
        "duration_days": float(dur_hr) / 24.0,
        "Rp_RE": float(rp),
        "converged": converged,
        "path": os.path.abspath(posterior_path),
    }

def open_hdu(hdu: Any) -> tuple[fits.HDUList, bool]:
    """
    Normalize an HDU argument to an open ``HDUList``.

    Accepts an already-open ``HDUList`` (returned unchanged, not closed by
    the caller) or a path to a FITS file (opened read-only).

    Parameters
    ----------
    hdu : fits.HDUList or str
        Open HDUList, or a path to a TESScut FITS file.

    Returns
    -------
    (hdulist, opened_here) : tuple
        ``opened_here`` is True when this call opened the file, so the
        caller knows whether to close it.

    Raises
    ------
    TypeError
        If ``hdu`` is neither an HDUList nor an existing path.

    Examples
    --------
    >>> hl, mine = open_hdu("/path/to/cutout.fits")
    >>> if mine: hl.close()
    """
    if isinstance(hdu, fits.HDUList):
        return hdu, False
    if isinstance(hdu, (list, tuple)) and len(hdu) and isinstance(
        hdu[0], fits.HDUList
    ):
        return hdu[0], False
    if isinstance(hdu, str) and os.path.isfile(hdu):
        return fits.open(hdu, memmap=False), True
    raise TypeError(
        f"hdu must be an astropy HDUList or a path to a FITS file; "
        f"got {type(hdu).__name__} ({hdu!r})"
    )


# --------------------------------------------------------------------------
# TRILEGAL SSL workaround
# --------------------------------------------------------------------------

@contextmanager
def trilegal_ssl_workaround(
    host: str = TRILEGAL_HOST,
    verbose: bool = True,
) -> Iterator[None]:
    """
    Temporarily disable TLS certificate verification for one hostname,
    across both HTTP stacks triceratops reaches it through.

    The TRILEGAL service at ``stev.oapd.inaf.it`` serves an incomplete
    certificate chain: it presents its leaf certificate without the
    intermediate needed to build a path to a trusted root. Python's certifi
    bundle and the macOS keychain both reject it, confirmed by curl and
    requests failing identically, so no client-side CA bundle fixes it.
    ``triceratops.target.__init__`` hardcodes ``verify_ssl=True`` when
    calling ``query_TRILEGAL`` despite accepting a ``verify_ssl`` argument,
    so the package's own escape hatch is unreachable.

    Two interception points are needed:

    - ``query_TRILEGAL`` submits the form through ``mechanicalsoup``, which
      wraps ``requests``. Patching
      ``requests.Session.merge_environment_settings`` catches it where the
      effective ``verify`` value is resolved, so it survives the
      http-to-https redirect and is not overridden by ``REQUESTS_CA_BUNDLE``.
    - ``save_trilegal`` reads the result with ``pandas.read_csv(url)``,
      which goes through ``urllib.request.urlopen`` and never touches
      ``requests``.

    Scope is one hostname and the duration of the block. No triceratops
    source is modified, and both patches revert on exit even if the body
    raises. This is narrower than triceratops' own ``verify_ssl=False``
    path, which replaces ``ssl._create_default_https_context`` process-wide
    and would silently disable verification for later MAST and Gaia queries.

    Security note: inside the block a response from ``host`` could be
    substituted by an attacker on the network path. The payload is a
    synthetic background stellar population used as a scenario prior; a
    corrupted one produces visibly wrong source counts rather than a subtly
    biased FPP, and nothing is executed from it.

    Parameters
    ----------
    host : str
        Hostname substring matched against request URLs.
    verbose : bool
        Print when the patches are installed and removed.

    Yields
    ------
    None

    Examples
    --------
    >>> with trilegal_ssl_workaround():
    ...     target = tr.target(ID=270380593, sectors=np.array([4]))
    """
    import requests

    original_merge = requests.Session.merge_environment_settings
    original_urlopen = urllib.request.urlopen

    def patched_merge(self, url, proxies, stream, verify, cert):
        settings = original_merge(self, url, proxies, stream, verify, cert)
        if host in str(url):
            settings["verify"] = False
        return settings

    def patched_urlopen(url, *args, **kwargs):
        full = url.full_url if hasattr(url, "full_url") else str(url)
        if host in full and kwargs.get("context") is None:
            kwargs["context"] = ssl._create_unverified_context()
        return original_urlopen(url, *args, **kwargs)

    requests.Session.merge_environment_settings = patched_merge
    urllib.request.urlopen = patched_urlopen
    if verbose:
        print(f"[trilegal_ssl_workaround] verification disabled for {host} "
              f"(requests + urllib)")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*[Uu]nverified HTTPS request.*")
            yield
    finally:
        requests.Session.merge_environment_settings = original_merge
        urllib.request.urlopen = original_urlopen
        if verbose:
            print(f"[trilegal_ssl_workaround] verification restored for {host}")


# --------------------------------------------------------------------------
# Aperture bridge
# --------------------------------------------------------------------------

def nemesis_mask_to_ccd_pixels(
    hdu: fits.HDUList,
    mask: np.ndarray,
    verbose: bool = True,
) -> np.ndarray:
    """
    Convert a NEMESIS boolean aperture mask into the absolute CCD pixel list
    TRICERATOPS expects.

    NEMESIS apertures are boolean arrays of shape ``(ny, nx)`` indexed in
    cutout coordinates, as returned by ``thresholdmask`` and
    ``create_circular_mask``. TRICERATOPS ``calc_depths`` expects an
    ``(N, 2)`` integer array of absolute TESS CCD ``[column, row]`` pairs,
    matching the format of ``target.get_spoc_apertures()``. The mapping is a
    constant offset read from the TESScut header.

    For a TESS pixel-coordinate WCS, ``CRVALnP`` gives the physical CCD
    coordinate of the reference pixel and ``CRPIXnP`` its 1-indexed position
    within the cutout, so for a 0-indexed cutout index ``j``::

        ccd_column = CRVAL1P + (j + 1 - CRPIX1P)
        ccd_row    = CRVAL2P + (i + 1 - CRPIX2P)

    Keywords are tried in the order given by :data:`CCD_OFFSET_KEYWORDS` and
    the pair actually used is reported, so a product with an unexpected
    layout is visible rather than silent.

    Parameters
    ----------
    hdu : fits.HDUList
        TESScut HDUList: ``hdu[1]`` PIXELS, ``hdu[2]`` APERTURE.
    mask : np.ndarray of bool
        Aperture mask, shape ``(ny, nx)``. A per-cadence stack is rejected
        rather than silently collapsed, since a time-varying aperture has no
        single TRICERATOPS equivalent.
    verbose : bool
        Print the resolved offset, pixel count and CCD extent.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(N, 2)``, columns ``[ccd_column, ccd_row]``,
        sorted by row then column. Wrap as ``np.array([pixels])`` for a
        single-sector ``calc_depths`` call.

    Raises
    ------
    ValueError
        If the mask is not 2D, is empty, disagrees with the APERTURE
        extension shape, or if no recognized offset keywords are present.

    Examples
    --------
    >>> ap = nemesis_mask_to_ccd_pixels(hdu, pix_mask)
    >>> target.calc_depths(tdepth=0.0043, all_ap_pixels=np.array([ap]))
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(
            f"mask must be 2D (ny, nx); got shape {mask.shape}. A per-cadence "
            f"mask stack has no single TRICERATOPS equivalent."
        )
    mask = mask.astype(bool)
    if not mask.any():
        raise ValueError("mask selects zero pixels")

    crval1 = crval2 = crpix1 = crpix2 = None
    used = ""
    for ext, k1, k2 in CCD_OFFSET_KEYWORDS:
        if ext >= len(hdu):
            continue
        hdr = hdu[ext].header
        if k1 in hdr and k2 in hdr:
            crval1, crval2 = int(hdr[k1]), int(hdr[k2])
            p1 = k1.replace("CRVAL", "CRPIX").replace("CRV", "CRP")
            p2 = k2.replace("CRVAL", "CRPIX").replace("CRV", "CRP")
            crpix1 = int(hdr.get(p1, 1))
            crpix2 = int(hdr.get(p2, 1))
            used = f"hdu[{ext}] {k1}/{k2}"
            break

    if crval1 is None:
        raise ValueError(
            "no CCD offset keywords found. Tried "
            + ", ".join(f"hdu[{e}].{a}/{b}" for e, a, b in CCD_OFFSET_KEYWORDS)
            + ". Inspect the header and extend CCD_OFFSET_KEYWORDS."
        )

    if len(hdu) > 2 and getattr(hdu[2], "data", None) is not None:
        ap_shape = np.asarray(hdu[2].data).shape
        if ap_shape != mask.shape:
            raise ValueError(
                f"mask shape {mask.shape} does not match the APERTURE "
                f"extension shape {ap_shape}; the mask may belong to a "
                f"different cutout."
            )

    rows, cols = np.nonzero(mask)
    ccd_cols = crval1 + (cols + 1 - crpix1)
    ccd_rows = crval2 + (rows + 1 - crpix2)

    pixels = np.column_stack([ccd_cols, ccd_rows]).astype(int)
    pixels = pixels[np.lexsort((pixels[:, 0], pixels[:, 1]))]

    if verbose:
        print(f"offset source : {used}")
        print(f"corner        : col {crval1 + 1 - crpix1}, "
              f"row {crval2 + 1 - crpix2}")
        print(f"pixels        : {len(pixels)}")
        print(f"CCD column    : {pixels[:, 0].min()} to {pixels[:, 0].max()}")
        print(f"CCD row       : {pixels[:, 1].min()} to {pixels[:, 1].max()}")

    return pixels


# --------------------------------------------------------------------------
# Light curve preparation
# --------------------------------------------------------------------------

def prepare_triceratops_lightcurve(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    n_durations: int = N_DURATIONS_DEFAULT,
    n_bins: int | None = None,
    oot_factor: float = 1.5,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Phase-fold, trim and optionally bin a NEMESIS light curve into the arrays
    TRICERATOPS ``calc_probs`` expects.

    Binning is OFF by default. At 30-minute FFI cadence a folded single
    sector yields only a few hundred points already spread near-continuously
    in phase, so binning discards the sampling that folding just produced.
    Enable it only for 2-minute or 20-second data, where the folded point
    count runs to thousands and runtime scales with it.

    The trim window scales to the fitted transit duration rather than being
    fixed in days, because a fixed window is a large fraction of the period
    for the short-period M-dwarf candidates this survey targets.

    The returned uncertainty is always measured on the array actually
    returned. When binning is enabled it comes from the BINNED
    out-of-transit residuals; taking it from the unbinned scatter and
    applying it to binned points, as some published example notebooks do,
    overstates the error by roughly ``sqrt(points per bin)``, which flattens
    the scenario likelihood ratios and biases FPP toward the priors.

    ``calc_probs`` takes ``flux_err_0`` as a scalar, so one value is
    returned rather than a per-point vector.

    Parameters
    ----------
    time : np.ndarray
        Time stamps in BTJD.
    flux : np.ndarray
        Normalized flux, median near 1.0.
    period : float
        Orbital period in days.
    t0 : float
        Transit mid-time in BTJD, same epoch convention as ``time``.
    duration : float
        Fitted transit duration in DAYS. NEMESIS stores hours in
        ``{search} Dur [hrs]``, so divide by 24 before calling.
    n_durations : int
        Half-width of the retained window, in transit durations.
    n_bins : int or None
        ``None`` returns folded, unbinned points. An integer bins to that
        many phase bins, dropping empty ones; a value implying fewer than
        about 8 points per bin is rejected.
    oot_factor : float
        Points beyond ``oot_factor * duration / 2`` from mid-transit are
        treated as out of transit for the noise estimate.
    verbose : bool
        Print counts, the derived uncertainty and the depth significance.

    Returns
    -------
    (t_out, f_out, flux_err) : tuple
        Phase in days from mid-transit, normalized flux, and the scalar
        1-sigma uncertainty per returned point.

    Raises
    ------
    ValueError
        If inputs disagree in length, if ``period`` or ``duration`` are not
        positive, if the trim window retains nothing, if fewer than 5 points
        are out of transit, or if ``n_bins`` is too large for the data.

    Examples
    --------
    >>> t, f, e = prepare_triceratops_lightcurve(
    ...     time, flux, period=3.8362, t0=1414.136, duration=2.57 / 24)
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    if time.shape != flux.shape:
        raise ValueError(
            f"time and flux must match; got {time.shape} and {flux.shape}")
    if not (period > 0 and duration > 0):
        raise ValueError(
            f"period and duration must be positive; got {period}, {duration}")

    good = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[good], flux[good]
    n_input = time.size

    phase = (time - t0 + 0.5 * period) % period - 0.5 * period

    half_window = n_durations * duration
    if 2.0 * half_window >= period:
        half_window = 0.45 * period
        if verbose:
            print(f"window clipped to +/- {half_window:.4f} d "
                  f"({n_durations} durations exceeded the period)")

    keep = np.abs(phase) <= half_window
    ph, fl = phase[keep], flux[keep]
    if ph.size == 0:
        raise ValueError("no points inside the trim window")

    order = np.argsort(ph)
    ph, fl = ph[order], fl[order]

    if n_bins is None:
        t_out, f_out = ph, fl
        mode, per_bin = "unbinned", 1
    else:
        if ph.size < 8 * n_bins:
            raise ValueError(
                f"n_bins={n_bins} gives {ph.size / n_bins:.1f} points per bin "
                f"from {ph.size} folded points. Use n_bins=None for FFI "
                f"cadence, or at most n_bins={ph.size // 8}."
            )
        edges = np.linspace(-half_window, half_window, n_bins + 1)
        idx = np.clip(np.digitize(ph, edges) - 1, 0, n_bins - 1)
        counts = np.bincount(idx, minlength=n_bins)
        sums = np.bincount(idx, weights=fl, minlength=n_bins)
        occupied = counts > 0
        centers = 0.5 * (edges[:-1] + edges[1:])
        t_out = centers[occupied]
        f_out = sums[occupied] / counts[occupied]
        mode = f"binned to {int(occupied.sum())} of {n_bins}"
        per_bin = int(np.median(counts[occupied]))

    oot = np.abs(t_out) > oot_factor * 0.5 * duration
    if oot.sum() < 5:
        oot = np.abs(t_out) > 0.5 * half_window
    if oot.sum() < 5:
        raise ValueError(
            f"only {int(oot.sum())} out-of-transit points; increase "
            f"n_durations above {n_durations}")

    resid = f_out[oot] - np.median(f_out[oot])
    flux_err = float(1.4826 * np.median(np.abs(resid)))
    if not np.isfinite(flux_err) or flux_err <= 0:
        flux_err = float(np.std(resid, ddof=1))

    in_tr = np.abs(t_out) <= 0.5 * duration

    if verbose:
        depth = 1.0 - np.median(f_out[in_tr]) if in_tr.any() else np.nan
        print(f"window        : +/- {half_window:.4f} d")
        print(f"points kept   : {ph.size} of {n_input}")
        print(f"mode          : {mode}  (median {per_bin} per point)")
        print(f"in transit    : {int(in_tr.sum())}")
        print(f"out of transit: {int(oot.sum())}")
        print(f"flux_err      : {flux_err:.6f}  (robust, on returned array)")
        if np.isfinite(depth):
            snr = depth / flux_err * np.sqrt(max(int(in_tr.sum()), 1))
            print(f"median depth  : {depth:.6f}  -> {snr:.1f} sigma total")

    return t_out, f_out, flux_err


# --------------------------------------------------------------------------
# Input assembly
# --------------------------------------------------------------------------
def nemesis_to_triceratops_inputs(
    hdu: fits.HDUList,
    LC_df: pd.DataFrame,
    bestfit_df: pd.DataFrame,
    savelcpath: str,
    bestfit_path: str | None = None,
    tic_id: int | None = None,
    sector: int | None = None,
    flux_type: str = "Detrended",
    search: str = "TLS",
    n_durations: int = N_DURATIONS_DEFAULT,
    n_bins: int | None = None,
    verbose: bool = True,
    posterior_path: str | None = None,
) -> dict[str, Any]:
    """
    Assemble every TRICERATOPS input from the products of one NEMESIS run on
    a single star-sector.

    Identity is enforced rather than assumed: ``tic_id`` and ``sector`` are
    preferentially parsed from ``bestfit_path``, and any explicitly passed
    value that disagrees raises. This matters more than it looks. Pairing one
    star's ephemeris with another star's aperture produces a plausible FPP
    with no error, and in a batch over a shortlist every mask file exists, so
    nothing would fail loudly. The jaxoplanet posterior, when given, is
    checked the same way.

    Ephemeris and depth come from different places on purpose. With
    ``posterior_path`` the period, mid-transit time and duration are the
    jaxoplanet posterior medians. The depth is always the search depth
    ``{search} depth [ppt]`` from the bestfit, because it was measured on the
    same light curve handed to TRICERATOPS and includes limb darkening and
    30-minute smearing as observed. ``MCMC Depth [ppt]`` is (Rp/Rs)^2 and
    would understate the observed dip, which lets fainter neighbors pass the
    TRICERATOPS host-brightness screen. Without ``posterior_path`` the search
    ephemeris is used, as in earlier versions.

    The light curve, error bar and depth are returned in NEMESIS' deblended
    units (the target's own flux), exactly as the survey measured them.
    TRICERATOPS expects total-aperture units and applies its own per-host
    renormalization (``triceratops.funcs.renorm_flux``), so
    :func:`run_triceratops_on_nemesis` converts them with
    :func:`convert_to_aperture_units` and NEMESIS' own 1 - FCR before
    TRICERATOPS sees them (10-14-2026). Passing them unchanged, the
    10-02-2026 choice, deblended twice.

    Stellar parameters are deliberately not assembled: ``tr.target`` queries
    the TIC directly at construction.

    Parameters
    ----------
    hdu : fits.HDUList
        TESScut HDUList for this star-sector.
    LC_df : pd.DataFrame
        NEMESIS light curve table. Requires ``Time`` and
        ``{flux_type} Flux``.
    bestfit_df : pd.DataFrame
        Single-row bestfit table. Always requires ``{search} depth [ppt]``.
        ``{search} Period [d]``, ``{search} TC [BTJD]`` and
        ``{search} Dur [hrs]`` are required only when ``posterior_path`` is
        None, and are used for a consistency check otherwise.
    savelcpath : str
        Directory holding the persisted ``*_pix_mask.pkl``.
    bestfit_path : str, optional
        Path the bestfit table came from. Strongly preferred: TIC and sector
        are parsed from its filename.
    tic_id, sector : int, optional
        Required only when ``bestfit_path`` is absent. Cross-checked against
        the parsed values when both are available.
    flux_type : str
        ``'SAP'``, ``'PLD'`` or ``'Detrended'``. Default ``'Detrended'``,
        which is what the transit search operated on.
    search : str
        ``'TLS'`` or ``'BLS'``. Supplies the depth always, and the
        ephemeris only when ``posterior_path`` is None.
    n_durations : int
        Half-width of the retained fold window, in transit durations.
    n_bins : int or None
        Passed through; ``None`` means no binning.
    verbose : bool
        Print the assembled quantities.
    posterior_path : str, optional
        jaxoplanet ``*_jaxoplanet_posterior.csv`` for this star-sector. See
        :func:`read_jaxoplanet_posterior`.

    Returns
    -------
    dict
        ``time``, ``flux``, ``flux_err``, ``P_orb``, ``T0``,
        ``duration_days``, ``tdepth``, ``ap_pixels``, ``all_ap_pixels``,
        ``tic_id``, ``sector``, ``ephemeris_source`` (``'jaxoplanet'`` or
        the search name), ``posterior_path`` (absolute path or None),
        ``posterior_converged`` (bool or None), ``Rp_posterior_RE``
        (NaN without a posterior).

    Raises
    ------
    ValueError
        If TIC or sector cannot be determined, a passed value disagrees with
        the parsed one, or the posterior fails its identity or value checks.
    KeyError
        If a required bestfit, light curve or posterior column is absent.
    FileNotFoundError
        If the persisted aperture mask or the posterior cannot be read.

    Examples
    --------
    >>> inp = nemesis_to_triceratops_inputs(
    ...     hdu, LC_df, bestfit_df, savelcpath, bestfit_path=hits[0],
    ...     posterior_path=post_csv)
    >>> inp["ephemeris_source"]
    'jaxoplanet'
    """
    parsed_tic = parsed_sector = None
    if bestfit_path is not None:
        base = os.path.basename(bestfit_path)
        m_tic = re.search(r"TIC[_ ]?(\d+)", base)
        m_sec = re.search(r"Sector[_ ]?(\d+)", base)
        if m_tic:
            parsed_tic = int(m_tic.group(1))
        if m_sec:
            parsed_sector = int(m_sec.group(1))

    if parsed_tic is not None and tic_id is not None and int(tic_id) != parsed_tic:
        raise ValueError(
            f"TIC mismatch: tic_id={tic_id} but bestfit file names "
            f"TIC {parsed_tic} ({os.path.basename(bestfit_path)}). Pairing "
            f"one star's ephemeris with another's aperture yields a plausible "
            f"FPP for the wrong star. Refusing."
        )
    if parsed_sector is not None and sector is not None and int(sector) != parsed_sector:
        raise ValueError(
            f"Sector mismatch: sector={sector} but bestfit file names "
            f"Sector {parsed_sector} ({os.path.basename(bestfit_path)})."
        )

    tic_id = int(parsed_tic if parsed_tic is not None else (tic_id or 0))
    sector = int(parsed_sector if parsed_sector is not None
                 else (sector if sector is not None else -1))
    if tic_id <= 0 or sector < 0:
        raise ValueError(
            "could not determine TIC and sector. Pass bestfit_path, or both "
            "tic_id and sector explicitly."
        )

    depth_col = f"{search} depth [ppt]"
    eph_cols = [f"{search} Period [d]", f"{search} TC [BTJD]",
                f"{search} Dur [hrs]"]
    need = [depth_col] + (eph_cols if posterior_path is None else [])
    missing = [c for c in need if c not in bestfit_df.columns]
    if missing:
        raise KeyError(
            f"bestfit_df is missing {missing}. Present: "
            f"{list(bestfit_df.columns)[:12]} ..."
        )

    row = bestfit_df.iloc[0]
    tdepth = float(row[depth_col]) / 1000.0

    posterior_converged: bool | None = None
    rp_posterior = float("nan")
    if posterior_path is None:
        period = float(row[eph_cols[0]])
        t0 = float(row[eph_cols[1]])
        duration_days = float(row[eph_cols[2]]) / 24.0
        ephemeris_source = search
    else:
        post = read_jaxoplanet_posterior(posterior_path, tic_id=tic_id,
                                         sector=sector, verbose=verbose)
        period = post["P_orb"]
        t0 = post["T0"]
        duration_days = post["duration_days"]
        rp_posterior = post["Rp_RE"]
        posterior_converged = post["converged"]
        posterior_path = post["path"]
        ephemeris_source = "jaxoplanet"

        if all(c in bestfit_df.columns for c in eph_cols[:2]):
            try:
                p_search = float(row[eph_cols[0]])
                t0_search = float(row[eph_cols[1]])
            except (TypeError, ValueError):
                p_search = t0_search = float("nan")
            if np.isfinite(p_search) and p_search > 0:
                frac = abs(period - p_search) / p_search
                if verbose:
                    print(f"{search} vs jaxoplanet: dP/P = {frac:.2e}, "
                          f"dT0 = {(t0 - t0_search) * 24.0:+.3f} hr")
                if frac > POSTERIOR_PERIOD_TOL:
                    warnings.warn(
                        f"TIC {tic_id} S{sector:04d}: jaxoplanet period "
                        f"{period:.6f} d differs from the {search} period "
                        f"{p_search:.6f} d by {frac:.1%}, above "
                        f"POSTERIOR_PERIOD_TOL. Check that the posterior "
                        f"belongs to this signal.", RuntimeWarning)

    flux_col = f"{flux_type} Flux"
    for c in ("Time", flux_col):
        if c not in LC_df.columns:
            raise KeyError(
                f"LC_df is missing {c!r}. Present: {list(LC_df.columns)}")

    t_out, f_out, flux_err = prepare_triceratops_lightcurve(
        time=LC_df["Time"].to_numpy(dtype=float),
        flux=LC_df[flux_col].to_numpy(dtype=float),
        period=period, t0=t0, duration=duration_days,
        n_durations=n_durations, n_bins=n_bins, verbose=verbose,)
    savelcpath = os.path.join(savelcpath, "")
    sector_tag = f"Sector_{sector:04d}"
    mask_stem = f"TIC_{tic_id}_{sector_tag}_pix_mask"
    try:
        pix_mask = read_ndarray(savelcpath, mask_stem)
    except Exception as exc:
        raise FileNotFoundError(
            f"could not read {mask_stem} from {savelcpath}: {exc}. NEMESIS "
            f"cleanup_files deletes '*pix_mask.pkl'; re-run full_pipeline "
            f"with keep_extra_files=True and keep_imagedata=True, or "
            f"regenerate the mask for this star-sector."
        ) from exc

    pix_mask = np.asarray(pix_mask)
    if pix_mask.ndim != 2:
        pix_mask = pix_mask[0]

    ap_pixels = nemesis_mask_to_ccd_pixels(hdu, pix_mask, verbose=verbose)

    if verbose:
        print(f"\nTIC {tic_id}  Sector {sector}  "
              f"(ephemeris: {ephemeris_source}, depth: {search})")
        print(f"P_orb         : {period:.7f} d")
        print(f"T0            : {t0:.6f} BTJD")
        print(f"duration      : {duration_days * 24.0:.3f} hr")
        print(f"tdepth        : {tdepth:.6f}  ({tdepth * 1e3:.3f} ppt)")
        print(f"aperture      : {len(ap_pixels)} pixels")
        print(f"LC points     : {t_out.size}")

    return {
        "time": t_out, "flux": f_out, "flux_err": flux_err,
        "P_orb": period, "T0": t0, "duration_days": duration_days,
        "tdepth": tdepth, "ap_pixels": ap_pixels,
        "all_ap_pixels": np.array([ap_pixels]),
        "tic_id": int(tic_id), "sector": int(sector),
        "ephemeris_source": ephemeris_source,
        "posterior_path": posterior_path,
        "posterior_converged": posterior_converged,
        "Rp_posterior_RE": rp_posterior,
    }

def nemesis_to_triceratops_inputs_OLD(
    hdu: fits.HDUList,
    LC_df: pd.DataFrame,
    bestfit_df: pd.DataFrame,
    savelcpath: str,
    bestfit_path: str | None = None,
    tic_id: int | None = None,
    sector: int | None = None,
    flux_type: str = "Detrended",
    search: str = "TLS",
    n_durations: int = 5,
    n_bins: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Assemble every TRICERATOPS input from the products of one NEMESIS run on
    a single star-sector.

    Identity is enforced rather than assumed: ``tic_id`` and ``sector`` are
    preferentially parsed from ``bestfit_path``, and any explicitly passed
    value that disagrees raises. This matters more than it looks. Pairing one
    star's ephemeris with another star's aperture produces a plausible FPP
    with no error, and in a batch over a shortlist every mask file exists, so
    nothing would fail loudly.

    Stellar parameters are deliberately not assembled: ``tr.target`` queries
    the TIC directly at construction.

    Parameters
    ----------
    hdu : fits.HDUList
        TESScut HDUList for this star-sector.
    LC_df : pd.DataFrame
        NEMESIS light curve table. Requires ``Time`` and
        ``{flux_type} Flux``.
    bestfit_df : pd.DataFrame
        Single-row bestfit table. Requires ``{search} Period [d]``,
        ``{search} TC [BTJD]``, ``{search} Dur [hrs]``,
        ``{search} depth [ppt]``.
    savelcpath : str
        Directory holding the persisted ``*_pix_mask.pkl``.
    bestfit_path : str, optional
        Path the bestfit table came from. Strongly preferred: TIC and sector
        are parsed from its filename.
    tic_id, sector : int, optional
        Required only when ``bestfit_path`` is absent. Cross-checked against
        the parsed values when both are available.
    flux_type : str
        ``'SAP'``, ``'PLD'`` or ``'Detrended'``. Default ``'Detrended'``,
        which is what the transit search operated on.
    search : str
        ``'TLS'`` or ``'BLS'``.
    n_durations : int
        Half-width of the retained fold window, in transit durations.
    n_bins : int or None
        Passed through; ``None`` means no binning.
    verbose : bool
        Print the assembled quantities.

    Returns
    -------
    dict
        ``time``, ``flux``, ``flux_err``, ``P_orb``, ``T0``,
        ``duration_days``, ``tdepth``, ``ap_pixels``, ``all_ap_pixels``,
        ``tic_id``, ``sector``.

    Raises
    ------
    ValueError
        If TIC or sector cannot be determined, or a passed value disagrees
        with the parsed one.
    KeyError
        If a required bestfit or light curve column is absent.
    FileNotFoundError
        If the persisted aperture mask cannot be read.

    Examples
    --------
    >>> inp = nemesis_to_triceratops_inputs(
    ...     hdu, LC_df, bestfit_df, savelcpath, bestfit_path=hits[0])
    """
    parsed_tic = parsed_sector = None
    if bestfit_path is not None:
        base = os.path.basename(bestfit_path)
        m_tic = re.search(r"TIC[_ ]?(\d+)", base)
        m_sec = re.search(r"Sector[_ ]?(\d+)", base)
        if m_tic:
            parsed_tic = int(m_tic.group(1))
        if m_sec:
            parsed_sector = int(m_sec.group(1))

    if parsed_tic is not None and tic_id is not None and int(tic_id) != parsed_tic:
        raise ValueError(
            f"TIC mismatch: tic_id={tic_id} but bestfit file names "
            f"TIC {parsed_tic} ({os.path.basename(bestfit_path)}). Pairing "
            f"one star's ephemeris with another's aperture yields a plausible "
            f"FPP for the wrong star. Refusing."
        )
    if parsed_sector is not None and sector is not None and int(sector) != parsed_sector:
        raise ValueError(
            f"Sector mismatch: sector={sector} but bestfit file names "
            f"Sector {parsed_sector} ({os.path.basename(bestfit_path)})."
        )

    tic_id = int(parsed_tic if parsed_tic is not None else (tic_id or 0))
    sector = int(parsed_sector if parsed_sector is not None
                 else (sector if sector is not None else -1))
    if tic_id <= 0 or sector < 0:
        raise ValueError(
            "could not determine TIC and sector. Pass bestfit_path, or both "
            "tic_id and sector explicitly."
        )

    need = [f"{search} Period [d]", f"{search} TC [BTJD]",
            f"{search} Dur [hrs]", f"{search} depth [ppt]"]
    missing = [c for c in need if c not in bestfit_df.columns]
    if missing:
        raise KeyError(
            f"bestfit_df is missing {missing}. Present: "
            f"{list(bestfit_df.columns)[:12]} ..."
        )

    row = bestfit_df.iloc[0]
    period = float(row[f"{search} Period [d]"])
    t0 = float(row[f"{search} TC [BTJD]"])
    duration_days = float(row[f"{search} Dur [hrs]"]) / 24.0
    tdepth = float(row[f"{search} depth [ppt]"]) / 1000.0

    flux_col = f"{flux_type} Flux"
    for c in ("Time", flux_col):
        if c not in LC_df.columns:
            raise KeyError(
                f"LC_df is missing {c!r}. Present: {list(LC_df.columns)}")

    t_out, f_out, flux_err = prepare_triceratops_lightcurve(
        time=LC_df["Time"].to_numpy(dtype=float),
        flux=LC_df[flux_col].to_numpy(dtype=float),
        period=period, t0=t0, duration=duration_days,
        n_durations=n_durations, n_bins=n_bins, verbose=verbose,
            posterior_path=posterior_path,
        )
    savelcpath = os.path.join(savelcpath, "")
    sector_tag = f"Sector_{sector:04d}"
    mask_stem = f"TIC_{tic_id}_{sector_tag}_pix_mask"
    try:
        pix_mask = read_ndarray(savelcpath, mask_stem)
    except Exception as exc:
        raise FileNotFoundError(
            f"could not read {mask_stem} from {savelcpath}: {exc}. NEMESIS "
            f"cleanup_files deletes '*pix_mask.pkl'; re-run full_pipeline "
            f"with keep_extra_files=True and keep_imagedata=True, or "
            f"regenerate the mask for this star-sector."
        ) from exc

    pix_mask = np.asarray(pix_mask)
    if pix_mask.ndim != 2:
        pix_mask = pix_mask[0]

    ap_pixels = nemesis_mask_to_ccd_pixels(hdu, pix_mask, verbose=verbose)

    if verbose:
        print(f"\nTIC {tic_id}  Sector {sector}  ({search})")
        print(f"P_orb         : {period:.7f} d")
        print(f"T0            : {t0:.6f} BTJD")
        print(f"duration      : {duration_days * 24.0:.3f} hr")
        print(f"tdepth        : {tdepth:.6f}  ({tdepth * 1e3:.3f} ppt)")
        print(f"aperture      : {len(ap_pixels)} pixels")
        print(f"LC points     : {t_out.size}")

    return {
        "time": t_out, "flux": f_out, "flux_err": flux_err,
        "P_orb": period, "T0": t0, "duration_days": duration_days,
        "tdepth": tdepth, "ap_pixels": ap_pixels,
        "all_ap_pixels": np.array([ap_pixels]),
        "tic_id": int(tic_id), "sector": int(sector),
    }


# --------------------------------------------------------------------------
# Main driver
# --------------------------------------------------------------------------
def summarize_trials(x: np.ndarray, label: str = "FPP") -> dict[str, float]:
    """
    Summarize a set of Monte Carlo trial values with quantiles.

    ``calc_probs`` output at 30-minute FFI cadence is not Gaussian and can be
    multi-modal: most draws cluster low while a minority land near unity,
    because a poorly constrained transit shape lets an eclipsing-binary
    scenario occasionally fit as well as a planet. A mean and standard
    deviation over such a sample describe a distribution that does not
    exist, so quantiles are reported instead. Mean and standard deviation are
    retained only for continuity with earlier runs and should not drive a
    disposition.

    ``frac_high`` is the diagnostic that matters: a nonzero value means some
    draws favored a false-positive scenario outright, and the median alone
    conceals that.

    Parameters
    ----------
    x : np.ndarray
        Finite trial values, one per successful ``calc_probs`` call.
    label : str
        Key prefix, normally ``'FPP'`` or ``'NFPP'``.

    Returns
    -------
    dict
        ``{label}_p16``, ``_p50``, ``_p84``, ``_p95``, ``_min``, ``_max``,
        ``_mean``, ``_std``, ``_frac_high`` (fraction above 0.5), and
        ``_n``. Empty input yields NaN throughout with ``_n`` zero.

    Examples
    --------
    >>> s = summarize_trials(np.array([0.01, 0.02, 0.9]), "FPP")
    >>> round(s["FPP_p50"], 3)
    0.02
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        keys = ("p16", "p50", "p84", "p95", "min", "max", "mean", "std",
                "frac_high")
        out = {f"{label}_{k}": float("nan") for k in keys}
        out[f"{label}_n"] = 0
        return out

    p16, p50, p84, p95 = np.percentile(x, [16.0, 50.0, 84.0, 95.0])
    return {
        f"{label}_p16": float(p16),
        f"{label}_p50": float(p50),
        f"{label}_p84": float(p84),
        f"{label}_p95": float(p95),
        f"{label}_min": float(x.min()),
        f"{label}_max": float(x.max()),
        f"{label}_mean": float(x.mean()),
        f"{label}_std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
        f"{label}_frac_high": float((x > 0.5).mean()),
        f"{label}_n": int(x.size),
    }

def render_to_file(
    plot_call,
    path: str,
    dpi: int = 150,
    verbose: bool = True,
) -> str | None:
    """
    Run a plotting call and save whatever figure it leaves open.

    ``plt.savefig`` writes the *current* figure. Under the IPython inline
    backend, ``plt.show()`` displays all open figures and then destroys
    them, so a library function that calls ``plt.show()`` internally leaves
    no current figure behind. The subsequent ``plt.savefig`` silently
    creates a fresh empty one and writes a blank white image. Some plotting
    helpers also call ``plt.close()`` themselves when asked not to save,
    with the same effect.

    This suppresses ``plt.show`` for the duration of the call, then saves
    the most recently created figure directly through its own ``savefig``
    method rather than through the pyplot state machine. A figure with no
    axes is reported rather than written, so a blank result surfaces as a
    warning instead of a white PNG.

    Parameters
    ----------
    plot_call : callable
        Zero-argument callable that draws a figure, e.g.
        ``lambda: target.plot_field(sector=4, ap_pixels=ap)``.
    path : str
        Output path. The extension determines the format.
    dpi : int
        Output resolution.
    verbose : bool
        Print a warning when nothing usable was produced.

    Returns
    -------
    str or None
        ``path`` on success, None if the call produced no figure, produced
        one with no axes, or raised.

    Examples
    --------
    >>> render_to_file(lambda: target.plot_field(sector=4, ap_pixels=ap),
    ...                "field.png")
    """
    original_show = plt.show
    plt.close("all")
    plt.show = lambda *args, **kwargs: None
    try:
        plot_call()
        nums = plt.get_fignums()
        if not nums:
            if verbose:
                print(f"[warn] no figure produced for {os.path.basename(path)}")
            return None
        fig = plt.figure(nums[-1])
        if not fig.get_axes():
            if verbose:
                print(f"[warn] figure for {os.path.basename(path)} has no "
                      f"axes; not written")
            return None
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        return path
    except Exception as exc:
        if verbose:
            print(f"[warn] {os.path.basename(path)} failed: "
                  f"{type(exc).__name__}: {exc}")
        return None
    finally:
        plt.show = original_show
        plt.close("all")
def pdf_page_size_px(
    pdf_path: str,
    dpi: int = 150,
    page_index: int = 0,
    verbose: bool = True,
) -> tuple[int, int] | None:
    """
    Read a PDF page's dimensions and convert them to pixels.

    PDF geometry is stored in points, 1/72 inch, in the page MediaBox. New
    pages appended to the NEMESIS vetting report should be composed at the
    same physical size as the existing pages so the document has a uniform
    trim, which means reading the size rather than assuming it.

    Parameters
    ----------
    pdf_path : str
        Existing multi-page PDF.
    dpi : int
        Resolution used to convert points to pixels.
    page_index : int
        Which page to measure. Page 1 is index 0.
    verbose : bool
        Print the measured size in inches and pixels.

    Returns
    -------
    (width_px, height_px) or None
        None if the file is missing, has no pages, or PyPDF2 is
        unavailable.

    Examples
    --------
    >>> pdf_page_size_px("TIC_..._TLS_Report_multi_page.pdf")
    (1500, 1200)
    """
    if not os.path.isfile(pdf_path):
        if verbose:
            print(f"[warn] pdf_page_size_px: no PDF at {pdf_path}")
        return None
    try:
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        if not len(reader.pages):
            return None
        page = reader.pages[page_index]
        box = getattr(page, "mediabox", None) or getattr(page, "mediaBox")
        w_pt, h_pt = float(box.width), float(box.height)
    except Exception as exc:
        if verbose:
            print(f"[warn] pdf_page_size_px: {type(exc).__name__}: {exc}")
        return None

    w_px = int(round(w_pt / 72.0 * dpi))
    h_px = int(round(h_pt / 72.0 * dpi))
    if verbose:
        print(f"report page   : {w_pt / 72.0:.2f} x {h_pt / 72.0:.2f} in  "
              f"-> {w_px} x {h_px} px at {dpi} dpi")
    return w_px, h_px

def compose_triceratops_page(
    field_png: str,
    fits_png: str,
    out_stem: str,
    page_size_px: tuple[int, int],
    header_lines: list[str] | None = None,
    layout: str = "auto",
    split: bool = False,
    dpi: int = 150,
    verbose: bool = True,
) -> list[str]:
    """
    Compose the TRICERATOPS figures onto page(s) matching a fixed canvas.

    The two inputs have opposing aspect ratios: the field plot is wide and
    short (two panels side by side) while the scenario grid is tall and
    narrow (fifteen panels, three across by five down). Fitting both onto
    one page of fixed size therefore requires substantial downscaling of at
    least one, so the achieved linear scale of each is reported.

    Two arrangements are evaluated. ``stacked`` gives both figures a common
    width and places them one above the other; ``side_by_side`` gives both a
    common height and places them left and right. Under ``layout='auto'``
    the arrangement that maximizes the smaller of the two scale factors is
    chosen, which avoids shrinking either figure far more than the other.

    With ``split=True`` two pages are emitted instead, each holding one
    figure at close to full page size. Use this when the scenario grid's
    axis labels need to be readable rather than merely its panel shapes.

    Axes are placed in figure-fraction coordinates rather than with a
    gridspec so nothing is clipped by automatic padding, and the output
    pixel dimensions match ``page_size_px`` exactly.

    Parameters
    ----------
    field_png : str
        Path to ``*_TRICERATOPS_field.png``.
    fits_png : str
        Path to ``*_TRICERATOPS_fits.png``.
    out_stem : str
        Output path without extension. ``'.png'`` is appended, or
        ``'_1.png'`` and ``'_2.png'`` when ``split`` is True.
    page_size_px : tuple of int
        ``(width, height)`` in pixels, normally from
        :func:`pdf_page_size_px` so the new page matches the existing ones.
    header_lines : list of str, optional
        Monospace text rendered at the top of the first page. Each line is
        auto-shrunk to fit the page width.
    layout : str
        ``'auto'``, ``'stacked'`` or ``'side_by_side'``. Ignored when
        ``split`` is True.
    split : bool
        Emit two pages instead of one.
    dpi : int
        Output resolution. Should match ``pdf_page_size_px``.
    verbose : bool
        Print the chosen layout and the achieved scale factors.

    Returns
    -------
    list of str
        Written paths, in page order. Empty if either input is missing or
        unreadable.

    Examples
    --------
    >>> size = pdf_page_size_px(report_pdf)
    >>> compose_triceratops_page(field, fits, stem, size,
    ...                          header_lines=hdr, split=False)
    """
    for p in (field_png, fits_png):
        if not p or not os.path.isfile(p):
            if verbose:
                print(f"[warn] compose_triceratops_page: missing {p!r}")
            return []
    try:
        im_f = np.asarray(Image.open(field_png).convert("RGB"))
        im_t = np.asarray(Image.open(fits_png).convert("RGB"))
    except Exception as exc:
        if verbose:
            print(f"[warn] compose_triceratops_page: "
                  f"{type(exc).__name__}: {exc}")
        return []

    W, H = int(page_size_px[0]), int(page_size_px[1])
    fh, fw = im_f.shape[0], im_f.shape[1]
    th, tw = im_t.shape[0], im_t.shape[1]
    pad = max(8, int(round(0.012 * min(W, H))))
    gap = pad

    def _new_page() -> "plt.Figure":
        fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
        fig.patch.set_facecolor("white")
        return fig

    def _place(fig, img, x, y, w, h) -> None:
        """x, y, w, h in pixels; y measured from the page bottom."""
        ax = fig.add_axes([x / W, y / H, w / W, h / H])
        ax.imshow(img, interpolation="antialiased", aspect="auto")
        ax.axis("off")

    def _header(fig, lines, top_y, band_h) -> None:
        ax = fig.add_axes([pad / W, top_y / H,
                           (W - 2 * pad) / W, band_h / H])
        ax.axis("off")
        longest = max((len(s) for s in lines), default=1)
        # Monospace glyph advance is about 0.6 em; solve for a size that
        # keeps the longest line inside the text band.
        fs = min(12.0, 0.95 * (W - 2 * pad) / (0.6 * longest) * 72.0 / dpi)
        ax.text(0.0, 1.0, "\n".join(lines), family="monospace",
                fontsize=max(5.0, fs), va="top", ha="left", linespacing=1.45)

    out: list[str] = []

    if split:
        for idx, (img, ih, iw) in enumerate(
                ((im_f, fh, fw), (im_t, th, tw)), start=1):
            fig = _new_page()
            band = 0
            if header_lines and idx == 1:
                band = int(round(len(header_lines) * 0.030 * H + 0.02 * H))
                _header(fig, header_lines, H - pad - band, band)
            avail_h = H - 2 * pad - band - (gap if band else 0)
            avail_w = W - 2 * pad
            s = min(avail_w / iw, avail_h / ih)
            w, h = s * iw, s * ih
            _place(fig, img, (W - w) / 2.0, pad + (avail_h - h) / 2.0, w, h)
            path = f"{out_stem}_{idx}.png"
            fig.savefig(path, dpi=dpi, facecolor="white")
            plt.close(fig)
            out.append(path)
            if verbose:
                print(f"page {idx}        : scale {s:.3f} of native")
        return out

    band = 0
    if header_lines:
        band = int(round(len(header_lines) * 0.030 * H + 0.02 * H))
    avail_w = W - 2 * pad
    avail_h = H - 2 * pad - band - (gap if band else 0)

    # stacked: common width, one above the other
    w_st = min(avail_w, (avail_h - gap) / (fh / fw + th / tw))
    s_st = min(w_st / fw, w_st / tw)

    # side_by_side: common height, left and right
    h_sb = min(avail_h, (avail_w - gap) / (fw / fh + tw / th))
    s_sb = min(h_sb / fh, h_sb / th)

    chosen = layout
    if layout == "auto":
        chosen = "stacked" if s_st >= s_sb else "side_by_side"

    fig = _new_page()
    if header_lines:
        _header(fig, header_lines, H - pad - band, band)

    if chosen == "stacked":
        w = w_st
        hf, ht = w * fh / fw, w * th / tw
        y_top = pad + avail_h
        _place(fig, im_f, (W - w) / 2.0, y_top - hf, w, hf)
        _place(fig, im_t, (W - w) / 2.0, y_top - hf - gap - ht, w, ht)
        sf, st = w / fw, w / tw
    else:
        h = h_sb
        wf, wt = h * fw / fh, h * tw / th
        x0 = (W - (wf + gap + wt)) / 2.0
        y0 = pad + (avail_h - h) / 2.0
        _place(fig, im_f, x0, y0, wf, h)
        _place(fig, im_t, x0 + wf + gap, y0, wt, h)
        sf, st = wf / fw, wt / tw

    path = f"{out_stem}.png"
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    out.append(path)

    if verbose:
        print(f"layout        : {chosen}  "
              f"(stacked {s_st:.3f} vs side_by_side {s_sb:.3f})")
        print(f"scale         : field {sf:.3f}, fits {st:.3f} of native")
        print(f"fits panel    : ~{int(st * tw / 3)} x {int(st * th / 5)} px "
              f"per scenario")
    return out

def compose_triceratops_report_page(
    field_png: str,
    fits_png: str,
    out_path: str,
    page_size_px: tuple[int, int],
    title: str = "TRICERATOPS Statistical Validation",
    subtitle: str | None = None,
    params: list[tuple[str, str]] | None = None,
    stats: list[tuple[str, str]] | None = None,
    scenarios: list[tuple[str, float, float]] | None = None,
    disposition: str | None = None,
    footnote: str | None = None,
    grid_width_frac: float = 0.46,
    dpi: int = 150,
    verbose: bool = True,
) -> str | None:
    """
    Compose a single NEMESIS-style vetting report page for TRICERATOPS.

    Purpose-built for the landscape report pages, unlike the generic
    :func:`compose_triceratops_page`. The two figures have opposing aspect
    ratios: the field plot is roughly 2.4:1 wide, the scenario grid roughly
    0.58:1 tall. Stacking them or placing them side by side at a common
    scale both waste most of a landscape page. Instead the portrait scenario
    grid takes a full-height right column while the wide field plot and all
    text share a left column, which roughly doubles the achieved scale of
    the grid.

    Text follows the conventions of report pages 1 and 4: a large centred
    title, a subtitle identifying the target, two-column label-value blocks
    rather than run-on lines, and a colour-coded verdict box mirroring the
    red box used for a failed symmetry test.

    Font sizes are derived from the page height in points, so the page looks
    the same whatever dpi or canvas size is passed.

    Parameters
    ----------
    field_png : str
        Path to ``*_TRICERATOPS_field.png``.
    fits_png : str
        Path to ``*_TRICERATOPS_fits.png``.
    out_path : str
        Output PNG path.
    page_size_px : tuple of int
        ``(width, height)`` in pixels, from :func:`pdf_page_size_px`, so the
        new page matches the existing report pages.
    title : str
        Large centred header.
    subtitle : str, optional
        Identifying line under the title, e.g.
        ``"TIC 270380593  |  Sector 4  |  TLS"``.
    params : list of (str, str), optional
        Left label-value block, normally the transit parameters and run
        configuration.
    stats : list of (str, str), optional
        Right label-value block, normally the FPP and NFPP quantiles.
    scenarios : list of (str, float, float), optional
        ``(abbreviation, median probability, trial scatter)`` in descending
        probability. Rendered as a ranked table; pass only the top handful.
    disposition : str, optional
        Verdict text. Box colour is inferred from the wording: green for a
        validated planet, amber-green for a likely planet, red for a likely
        nearby false positive, grey for unconverged, uncertain or
        inconclusive.
    footnote : str, optional
        Small italic line at the bottom of the left column, e.g. the
        citation for the classification criteria.
    grid_width_frac : float
        Maximum fraction of page width given to the scenario grid column.
        Raise it to enlarge the grid at the cost of the left column.
    dpi : int
        Output resolution.
    verbose : bool
        Print the achieved scale of each figure and the per-panel size.

    Returns
    -------
    str or None
        ``out_path`` on success, None if either figure is missing or
        unreadable.

    Examples
    --------
    >>> compose_triceratops_report_page(
    ...     field, fits, "page7.png", pdf_page_size_px(rep),
    ...     subtitle="TIC 270380593  |  Sector 4  |  TLS",
    ...     params=[("P [d]", "3.833913")],
    ...     stats=[("FPP p50", "0.00880")],
    ...     scenarios=[("TP", 0.7455, 0.28)],
    ...     disposition="likely planet")
    """
    for p in (field_png, fits_png):
        if not p or not os.path.isfile(p):
            if verbose:
                print(f"[warn] compose_triceratops_report_page: missing {p!r}")
            return None
    try:
        im_f = np.asarray(Image.open(field_png).convert("RGB"))
        im_t = np.asarray(Image.open(fits_png).convert("RGB"))
    except Exception as exc:
        if verbose:
            print(f"[warn] compose_triceratops_report_page: "
                  f"{type(exc).__name__}: {exc}")
        return None

    W, H = int(page_size_px[0]), int(page_size_px[1])
    fh, fw = im_f.shape[0], im_f.shape[1]
    th, tw = im_t.shape[0], im_t.shape[1]

    px_to_pt = 72.0 / dpi
    pad = max(10, int(round(0.014 * min(W, H))))
    gap = pad

    fs_title = max(9.0, 0.036 * H * px_to_pt)
    fs_sub = max(7.0, 0.021 * H * px_to_pt)
    fs_head = max(6.5, 0.019 * H * px_to_pt)
    fs_body = max(6.0, 0.0165 * H * px_to_pt)
    fs_cap = max(5.5, 0.014 * H * px_to_pt)

    line_px = fs_body / px_to_pt * 1.42
    cap_px = fs_cap / px_to_pt * 1.6

    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    fig.patch.set_facecolor("white")

    def _ax(x, y, w, h):
        """Axes from pixel coordinates; y measured from the page bottom."""
        a = fig.add_axes([x / W, y / H, w / W, h / H])
        a.axis("off")
        return a

    # ---- title band -----------------------------------------------------
    title_h = fs_title / px_to_pt * 1.5
    sub_h = (fs_sub / px_to_pt * 1.6) if subtitle else 0.0
    band = title_h + sub_h

    ax_t = _ax(pad, H - pad - band, W - 2 * pad, band)
    ax_t.set_xlim(0, 1)
    ax_t.set_ylim(0, 1)
    ax_t.text(0.5, 1.0, title, ha="center", va="top",
              fontsize=fs_title, fontweight="bold")
    if subtitle:
        ax_t.text(0.5, 1.0 - title_h / band, subtitle,
                  ha="center", va="top", fontsize=fs_sub, color="0.25")
    ax_t.plot([0, 1], [0.02, 0.02], color="0.6", lw=0.8)

    body_top = H - pad - band - gap
    body_h = body_top - pad

    # ---- right column: scenario grid at full body height ----------------
    cap_grid = cap_px if True else 0.0
    gh = body_h - cap_grid
    gw = gh * (tw / th)
    if gw > grid_width_frac * W:
        gw = grid_width_frac * W
        gh = gw * (th / tw)
    s_grid = gw / tw

    x_grid = W - pad - gw
    _ax(x_grid, body_top - gh, gw, gh).imshow(
        im_t, interpolation="antialiased", aspect="auto")
    a = _ax(x_grid, body_top - gh - cap_grid, gw, cap_grid)
    a.set_xlim(0, 1); a.set_ylim(0, 1)
    a.text(0.5, 0.6, "Scenario fits (15 configurations)",
           ha="center", va="center", fontsize=fs_cap, color="0.3")

    # ---- left column ----------------------------------------------------
    left_w = x_grid - gap - pad
    lfw = left_w
    lfh = lfw * (fh / fw)
    s_field = lfw / fw

    y = body_top - lfh
    _ax(pad, y, lfw, lfh).imshow(im_f, interpolation="antialiased",
                                 aspect="auto")
    y -= cap_px
    a = _ax(pad, y, lfw, cap_px)
    a.set_xlim(0, 1); a.set_ylim(0, 1)
    a.text(0.5, 0.6, "Aperture and field sources "
                     "(NEMESIS aperture in red)",
           ha="center", va="center", fontsize=fs_cap, color="0.3")
    y -= gap

    text_h = y - pad
    ax_txt = _ax(pad, pad, lfw, text_h)
    ax_txt.set_xlim(0, lfw)
    ax_txt.set_ylim(0, text_h)
    ty = text_h

    def _kv_block(x0, col_w, heading, rows, y_start):
        """Render a heading plus label-value rows; return the new y."""
        yy = y_start
        ax_txt.text(x0, yy, heading, fontsize=fs_head, fontweight="bold",
                    va="top", ha="left")
        yy -= line_px * 1.35
        for label, value in rows:
            ax_txt.text(x0, yy, label, fontsize=fs_body, va="top", ha="left",
                        color="0.25")
            ax_txt.text(x0 + col_w, yy, str(value), fontsize=fs_body,
                        va="top", ha="right", family="monospace")
            yy -= line_px
        return yy

    col_w = (lfw - gap) / 2.0
    y_left = y_right = ty
    if params:
        y_left = _kv_block(0.0, col_w * 0.95, "Transit Parameters",
                           params, ty)
    if stats:
        y_right = _kv_block(col_w + gap, col_w * 0.95,
                            "False Positive Probability", stats, ty)
    ty = min(y_left, y_right) - line_px * 0.6

    if scenarios:
        ax_txt.text(0.0, ty, "Scenario Probabilities",
                    fontsize=fs_head, fontweight="bold", va="top", ha="left")
        ty -= line_px * 1.35
        n_col = 2 if len(scenarios) > 4 else 1
        per = int(np.ceil(len(scenarios) / n_col))
        for k, (name, prob, std) in enumerate(scenarios):
            cx = (k // per) * (col_w + gap)
            cy = ty - (k % per) * line_px
            ax_txt.text(cx, cy, name, fontsize=fs_body, va="top", ha="left",
                        family="monospace", fontweight="bold")
            ax_txt.text(cx + col_w * 0.95, cy,
                        f"{prob:.4f} +/- {std:.4f}",
                        fontsize=fs_body, va="top", ha="right",
                        family="monospace")
        ty -= per * line_px + line_px * 0.6

    if disposition:
        d = disposition.lower()
        if "validated" in d:
            face, edge = "#d6efd6", "#2e7d32"
        elif "nearby false positive" in d:
            face, edge = "#f8d7d7", "#c62828"
        elif "likely planet" in d:
            face, edge = "#eaf3d9", "#7a9a2e"
        else:
            face, edge = "#eeeeee", "#757575"
        box_h = line_px * 2.1
        ty -= line_px * 0.2
        ax_txt.add_patch(plt.Rectangle(
            (0.0, ty - box_h), lfw, box_h,
            facecolor=face, edgecolor=edge, lw=1.6, zorder=0))
        ax_txt.text(lfw / 2.0, ty - box_h / 2.0, disposition.upper(),
                    fontsize=fs_head, fontweight="bold", color=edge,
                    va="center", ha="center", zorder=1)
        ty -= box_h + line_px * 0.4

    if footnote:
        ax_txt.text(0.0, ty, footnote, fontsize=fs_cap, style="italic",
                    color="0.4", va="top", ha="left")

    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)

    if verbose:
        print(f"report page   : {W} x {H} px")
        print(f"scale         : field {s_field:.3f}, grid {s_grid:.3f} "
              f"of native")
        print(f"fits panel    : ~{int(s_grid * tw / 3)} x "
              f"{int(s_grid * th / 5)} px per scenario")
    return out_path

import shutil
from PIL import Image

def append_png_to_pdf(
    pdf_path: str,
    png_path: str | list[str],
    out_path: str | None = None,
    in_place: bool = True,
    keep_pages: int | None = 6,
    dpi: int = 150,
    verbose: bool = True,
) -> str | None:
    """
    Append a PNG as a new final page of an existing multi-page PDF.

    Deliberately does NOT use the NEMESIS ``append_PDF`` helper. That
    function rebuilds the multi-page PDF from a list of page PNGs, writes to
    a filename it constructs itself, and deletes every input except the
    first. Passing it a single extra page would overwrite the six-page
    vetting report with a one-page PDF and remove the source image. It also
    cannot be used incrementally, because the page-2 through page-6 PNGs are
    already deleted by the time the report exists.

    Default behavior is non-destructive: the original PDF is left untouched
    and a new file with a ``_TRICERATOPS`` suffix is written. Because that
    derived file is always rebuilt from the pristine source, repeated calls
    are idempotent and cannot accumulate duplicate pages.

    In-place mode is available but should be paired with ``keep_pages`` so a
    second call truncates before appending rather than adding another page.

    Parameters
    ----------
    pdf_path : str
        Existing multi-page PDF, normally
        ``TIC_{ID}_Sector_{S:04d}_{TLS|BLS}_Report_multi_page.pdf``.
    png_path : str
        Page image to append.
    out_path : str, optional
        Explicit output path. Defaults to ``pdf_path`` with
        ``_TRICERATOPS`` inserted before the extension, or ``pdf_path``
        itself when ``in_place`` is True.
    in_place : bool
        Overwrite ``pdf_path``, which keeps the NEMESIS filename convention
        intact. Written to a temporary file and renamed, so an interruption
        cannot truncate the report. Default True.
    keep_pages : int, optional
        Retain only the first this many pages of the existing PDF before
        appending. Default 6, the NEMESIS vetting report page count, which
        makes repeated calls idempotent: a second run truncates back to the
        six original pages instead of adding a duplicate. Pass None to
        append unconditionally.
    dpi : int
        Resolution recorded when converting the PNG to PDF.
    verbose : bool
        Print page counts and the output path.

    Returns
    -------
    str or None
        Output path on success, None if either input is missing or PyPDF2
        is unavailable.

    Raises
    ------
    ValueError
        If ``keep_pages`` exceeds the page count of the existing PDF.

    Examples
    --------
    >>> append_png_to_pdf("TIC_..._TLS_Report_multi_page.pdf", "page7.png")
    """
    if not os.path.isfile(pdf_path):
        if verbose:
            print(f"[warn] append_png_to_pdf: no PDF at {pdf_path}")
        return None
    pngs = [png_path] if isinstance(png_path, str) else list(png_path)
    if not pngs:
        if verbose:
            print("[warn] append_png_to_pdf: no PNGs given")
        return None
    for p in pngs:
        if not os.path.isfile(p):
            if verbose:
                print(f"[warn] append_png_to_pdf: no PNG at {p}")
            return None

    try:
        from PyPDF2 import PdfReader, PdfWriter
    except ImportError:
        try:
            from pypdf import PdfReader, PdfWriter
        except ImportError:
            if verbose:
                print("[warn] append_png_to_pdf: neither PyPDF2 nor pypdf "
                      "is importable")
            return None

    if out_path is None:
        stem, ext = os.path.splitext(pdf_path)
        out_path = pdf_path if in_place else f"{stem}_TRICERATOPS{ext}"

    tmp_pdfs = [os.path.splitext(p)[0] + "_page.pdf" for p in pngs]
    try:
        for src, dst in zip(pngs, tmp_pdfs):
            Image.open(src).convert("RGB").save(
                dst, "PDF", resolution=float(dpi))

        reader = PdfReader(pdf_path)
        n_existing = len(reader.pages)
        if keep_pages is not None:
            if keep_pages > n_existing:
                raise ValueError(
                    f"keep_pages={keep_pages} exceeds the {n_existing} pages "
                    f"in {os.path.basename(pdf_path)}")
            pages = list(reader.pages)[:keep_pages]
        else:
            pages = list(reader.pages)

        writer = PdfWriter()
        for page in pages:
            writer.add_page(page)
        for tmp in tmp_pdfs:
            for page in PdfReader(tmp).pages:
                writer.add_page(page)

        tmp_out = out_path + ".tmp"
        with open(tmp_out, "wb") as fh:
            writer.write(fh)
        os.replace(tmp_out, out_path)

        if verbose:
            print(f"vetting report: {n_existing} -> "
                  f"{len(pages) + len(tmp_pdfs)} pages, {out_path}")
        return out_path

    except Exception as exc:
        if verbose:
            print(f"[warn] append_png_to_pdf: {type(exc).__name__}: {exc}")
        return None
    finally:
        for tmp in tmp_pdfs:
            if os.path.isfile(tmp):
                os.remove(tmp)

# --------------------------------------------------------------------------
# Converted inputs and the target-host outcome (1.3.0)
# --------------------------------------------------------------------------

def read_nemesis_fcr(bestfit_df: pd.DataFrame) -> tuple[float, str]:
    """
    Read the NEMESIS flux contamination ratio that deblended this light
    curve, from the bestfit table.

    NEMESIS deblends in its SAP step by subtracting FCR from the normalized
    flux and dividing by the new median, 1 - FCR (NEMESIS_pipeline.py,
    "Normalize and deblend"). Since 1.3.0 this number also converts the
    TRICERATOPS inputs back to aperture units, so it is required: a missing
    or unusable value raises instead of falling back.

    Parameters
    ----------
    bestfit_df : pd.DataFrame
        Single-row NEMESIS bestfit table.

    Returns
    -------
    (fcr, column) : tuple
        The ratio, and the column it came from (first match in
        :data:`FCR_COLUMN_CANDIDATES`, the same order 1.2.0 used).

    Raises
    ------
    KeyError
        If no candidate column is present.
    ValueError
        If the value is not a number in [0, 1).

    Examples
    --------
    >>> read_nemesis_fcr(pd.DataFrame({"Flux Contamination Ratio": [0.25]}))
    (0.25, 'Flux Contamination Ratio')
    >>> read_nemesis_fcr(pd.DataFrame({"x": [1.0]}))  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    KeyError: no FCR column
    """
    for col in FCR_COLUMN_CANDIDATES:
        if col in bestfit_df.columns:
            raw = bestfit_df.iloc[0][col]
            try:
                fcr = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"bestfit {col} is not a number: {raw!r}") from exc
            if not (np.isfinite(fcr) and 0.0 <= fcr < 1.0):
                raise ValueError(
                    f"bestfit {col} = {fcr}; need 0 <= FCR < 1 to convert "
                    f"the TRICERATOPS inputs")
            return fcr, col
    raise KeyError(
        f"no NEMESIS FCR column in the bestfit table; tried "
        f"{FCR_COLUMN_CANDIDATES}")


def convert_to_aperture_units(
    inp: dict[str, Any],
    one_minus_fcr: float,
) -> dict[str, Any]:
    """
    Convert the TRICERATOPS inputs from NEMESIS' deblended units (the
    target's own flux) to the total-aperture units TRICERATOPS expects.

    With s = 1 - FCR, the number NEMESIS deblended with::

        flux     -> 1 - s * (1 - flux)
        flux_err -> s * flux_err
        tdepth   -> s * tdepth

    This exactly reverses NEMESIS' deblend step, so TRICERATOPS receives
    the dip the aperture recorded and applies its own light model
    (``renorm_flux``) to every possible host, the target included
    (10-14-2026 decision). Shape, timing and signal-to-noise of every point
    are unchanged. The survey values are kept under ``flux_survey``,
    ``flux_err_survey`` and ``tdepth_survey``, and ``conversion_share``
    records s. The input dict is not modified.

    Parameters
    ----------
    inp : dict
        Output of :func:`nemesis_to_triceratops_inputs`.
    one_minus_fcr : float
        s = 1 - FCR, in (0, 1].

    Returns
    -------
    dict
        A copy of ``inp`` with converted ``flux``, ``flux_err`` and
        ``tdepth``, plus the survey values and ``conversion_share``.

    Raises
    ------
    ValueError
        If s is not in (0, 1].

    Examples
    --------
    >>> inp = {"flux": np.array([1.0, 0.9]), "flux_err": 0.01, "tdepth": 0.1}
    >>> out = convert_to_aperture_units(inp, 0.25)
    >>> out["flux"].round(6).tolist(), round(out["flux_err"], 6), round(out["tdepth"], 6)
    ([1.0, 0.975], 0.0025, 0.025)
    >>> inp["flux"].tolist(), out["flux_survey"].tolist(), out["conversion_share"]
    ([1.0, 0.9], [1.0, 0.9], 0.25)
    >>> fcr = 0.75; raw = np.array([1.0, 0.975])
    >>> deblended = {"flux": (raw - fcr) / (1 - fcr), "flux_err": 0.01, "tdepth": 0.1}
    >>> bool(np.allclose(convert_to_aperture_units(deblended, 1 - fcr)["flux"], raw))
    True
    >>> convert_to_aperture_units(inp, 0.0)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: share must be in (0, 1]
    """
    s = float(one_minus_fcr)
    if not (np.isfinite(s) and 0.0 < s <= 1.0):
        raise ValueError(
            f"conversion share (1 - FCR) must be in (0, 1]; got {s}")
    f = np.asarray(inp["flux"], dtype=float)
    out = dict(inp)
    out.update({
        "flux": 1.0 - s * (1.0 - f),
        "flux_err": s * float(inp["flux_err"]),
        "tdepth": s * float(inp["tdepth"]),
        "flux_survey": f,
        "flux_err_survey": float(inp["flux_err"]),
        "tdepth_survey": float(inp["tdepth"]),
        "conversion_share": s,
    })
    return out


def target_host_check(
    stars: pd.DataFrame,
    tic_id: int,
    depth_handed: float,
) -> dict[str, Any]:
    """
    Check the stars table after ``calc_depths`` and report whether the
    target can make the dip.

    ``calc_probs`` treats the first star with ``tdepth > 0`` as the target
    (its ``i == 0`` branch, triceratops.py line 765) and assumes that star
    is row 0 of ``target.stars``. This confirms the target is row 0 and
    reads whether ``calc_depths`` kept it as a possible host. A target
    depth of 0 means it would need more than 100% of its light; stock
    ``calc_probs`` would then evaluate the next possible host as if it were
    the target, so the caller must not call it.

    Parameters
    ----------
    stars : pd.DataFrame
        ``target.stars`` after ``calc_depths``: ``ID``, ``fluxratio``,
        ``tdepth``.
    tic_id : int
        The target.
    depth_handed : float
        Depth passed to ``calc_depths``, in aperture units.

    Returns
    -------
    dict
        ``fluxratio_target``; ``tdepth_target`` as ``calc_depths`` set it;
        ``needed_depth`` = depth_handed / fluxratio_target, before
        ``calc_depths`` caps it at 1; ``can_host`` (``tdepth_target > 0``,
        triceratops' own verdict); ``other_hosts``, the IDs of every other
        star with ``tdepth > 0``, in table order.

    Raises
    ------
    ValueError
        If the target is missing, appears more than once, or is not row 0.

    Examples
    --------
    >>> s = pd.DataFrame({"ID": ["1", "2"], "fluxratio": [0.005, 0.9],
    ...                   "tdepth": [0.0, 0.0075]})
    >>> c = target_host_check(s, 1, 0.00675)
    >>> c["can_host"], round(c["needed_depth"], 2), c["other_hosts"]
    (False, 1.35, ['2'])
    >>> target_host_check(s.iloc[::-1], 1, 0.00675)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: target is not row 0
    """
    ids_str = stars["ID"].astype(str).to_numpy()
    ids_num = pd.to_numeric(stars["ID"], errors="coerce").to_numpy(dtype=float)
    is_t = ids_num == float(tic_id)
    if int(is_t.sum()) != 1:
        raise ValueError(
            f"target TIC {tic_id} appears {int(is_t.sum())} times in "
            f"target.stars; expected once")
    if not is_t[0]:
        raise ValueError(
            f"row 0 of target.stars is {ids_str[0]}, not the target "
            f"{tic_id}; calc_probs would evaluate {ids_str[0]} as the "
            f"target. Refusing.")
    fr = float(stars["fluxratio"].to_numpy(dtype=float)[0])
    td = stars["tdepth"].to_numpy(dtype=float)
    need = float(depth_handed) / fr if fr > 0 else float("inf")
    others = [str(i) for i, t in zip(ids_str[1:], td[1:]) if t > 0]
    return {"fluxratio_target": fr, "tdepth_target": float(td[0]),
            "needed_depth": need, "can_host": bool(td[0] > 0),
            "other_hosts": others}


def check_trial_probs(
    probs: pd.DataFrame,
    tic_id: int,
    fpp: float,
    tol: float = 1e-6,
) -> None:
    """
    Guard one ``calc_probs`` result before it is recorded.

    The first 15 rows of ``target.probs`` are the target's own scenarios
    (TP through BEBx2P, the ``i == 0`` branch at triceratops.py line 765),
    and FPP is computed from rows 0, 3 and 9. If those rows carry another
    star's ID, the result describes the wrong star. A finite result whose
    probabilities do not sum to 1 is corrupt. Either raises, so no result
    is written for the star-sector.

    Parameters
    ----------
    probs : pd.DataFrame
        ``target.probs`` from one trial.
    tic_id : int
        The target.
    fpp : float
        ``target.FPP`` from the same trial. The sum is checked only when it
        is finite; the caller records a NaN trial as such.
    tol : float
        Allowed distance of the probability sum from 1.

    Raises
    ------
    RuntimeError
        On either failure.

    Examples
    --------
    >>> p = pd.DataFrame({"ID": [7] * 15 + [8] * 3, "prob": [1 / 18] * 18})
    >>> check_trial_probs(p, 7, 0.5) is None
    True
    >>> check_trial_probs(p.assign(ID=8), 7, 0.5)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    RuntimeError: target scenarios carry another ID
    """
    ids = pd.to_numeric(probs["ID"], errors="coerce").to_numpy(dtype=float)
    if ids.size < 15 or not np.all(ids[:15] == float(tic_id)):
        raise RuntimeError(
            f"the target's 15 scenario rows carry IDs "
            f"{sorted(set(probs['ID'].astype(str).tolist()[:15]))}, not "
            f"{tic_id}; this result describes another star. Refusing.")
    if np.isfinite(fpp):
        total = float(np.nansum(probs["prob"].to_numpy(dtype=float)))
        if abs(total - 1.0) > tol:
            raise RuntimeError(
                f"scenario probabilities sum to {total:.8f}, not 1. "
                f"Refusing.")


def cannot_host_trials(n_trials: int, other_hosts: list[str]) -> pd.DataFrame:
    """
    Per-trial table recorded, without calling ``calc_probs``, when
    ``calc_depths`` has excluded the target as a host.

    None of the target's own scenarios can occur under triceratops' light
    model, so by triceratops' definitions (FPP = 1 - (TP + PTP + DTP);
    NFPP = the sum of the other stars' scenarios) FPP = 1, and NFPP = 1
    when at least one other star can make the dip. When none can, NFPP is
    NaN. These are set by definition, not computed, and every row says so
    in ``status`` (decided 10-14-2026).

    Parameters
    ----------
    n_trials : int
        Rows to write, one per requested trial, so per-trial tables keep
        their usual length.
    other_hosts : list of str
        IDs of the other stars that can make the dip.

    Returns
    -------
    pd.DataFrame
        Columns ``trial``, ``FPP``, ``NFPP``, ``status``.

    Examples
    --------
    >>> t = cannot_host_trials(2, ["9590216"])
    >>> t["FPP"].tolist(), t["NFPP"].tolist(), t["status"].iloc[0]
    ([1.0, 1.0], [1.0, 1.0], 'set by definition')
    >>> bool(np.isnan(cannot_host_trials(1, [])["NFPP"].iloc[0]))
    True
    """
    n = int(n_trials)
    nfpp = 1.0 if other_hosts else float("nan")
    return pd.DataFrame({"trial": list(range(n)), "FPP": [1.0] * n,
                         "NFPP": [nfpp] * n,
                         "status": ["set by definition"] * n})
def count_negative_flux(flux: np.ndarray) -> dict[str, Any]:
    """
    Count the points below zero in the light curve handed to TRICERATOPS,
    and report its lowest value. A flag only: nothing is removed.

    Use it on the converted, whole-aperture light curve (``flux`` from
    :func:`convert_to_aperture_units`), never on the survey light curve.
    In the survey's deblended units the flux is the target's light alone,
    and when the target supplies only a few percent of the aperture light,
    NEMESIS' division by a small 1 - FCR magnifies ordinary scatter enough
    to cross zero. In whole-aperture units the flux is all the light the
    aperture recorded, relative to its median. No star, eclipse or blend
    can make that negative, so a point below zero is a data-quality
    problem, and no TRICERATOPS scenario can fit it (decided 10-15-2026).

    A survey point counts only if it lies below 1 - 1 / (1 - FCR).
    Worked example, TIC 401320705 S25, 1 - FCR = 0.035495, so the line is
    at -27.17: the survey point at -35.955 becomes
    1 - 0.035495 x 36.955 = -0.3117 and counts; the point at -23 becomes
    +0.148 and does not.

    Parameters
    ----------
    flux : np.ndarray
        The converted flux. NaN values are ignored.

    Returns
    -------
    dict
        ``n_below_zero``: number of points strictly below 0.
        ``min_flux``: the lowest finite value, NaN if there is none.

    Examples
    --------
    >>> count_negative_flux(np.array([1.0, 0.99, -0.31, 0.15]))
    {'n_below_zero': 1, 'min_flux': -0.31}
    >>> count_negative_flux(np.array([1.0, 0.0, np.nan]))
    {'n_below_zero': 0, 'min_flux': 0.0}
    >>> bool(np.isnan(count_negative_flux(np.array([np.nan]))["min_flux"]))
    True
    >>> inp = {"flux": np.array([1.0, -35.955, -23.0]), "flux_err": 0.3,
    ...        "tdepth": 0.744}
    >>> c = count_negative_flux(convert_to_aperture_units(inp, 0.035495)["flux"])
    >>> c["n_below_zero"], round(c["min_flux"], 4)
    (1, -0.3117)
    """
    f = np.asarray(flux, dtype=float).ravel()
    f = f[np.isfinite(f)]
    if f.size == 0:
        return {"n_below_zero": 0, "min_flux": float("nan")}
    return {"n_below_zero": int(np.count_nonzero(f < 0.0)),
            "min_flux": float(f.min())}

def summarize_scenarios(probs_all: pd.DataFrame) -> dict[str, Any]:
    """
    Summarize scenario probabilities over trials without pooling stars.

    TRICERATOPS gives each possible host its own rows: the target 15
    scenario types, every other possible host NTP, NEB and NEBx2P, so the
    same type name appears once per neighbor. The 1.2.0 summary grouped
    rows by name alone, putting every neighbor's rows in one pile and
    taking its middle value: one neighbor at 0.982 and two at or below
    0.00092 came out as 0.00092 (TIC 159344139 S26 at +/-5). Decided
    10-15-2026:

    1. Within each trial, add each scenario type over all stars.
    2. Over the target's own trials, take the median of those totals,
       with the mean, the standard deviation and the trial count.
    3. Keep the same statistics star by star, in TRICERATOPS' star order.
    4. The top scenario is the type with the largest median total (ties:
       TRICERATOPS order). Its leading star is the star with the largest
       own median for that type (ties: first in star order). This is the
       by-type rule.

    Trials with any non-finite probability (evidence underflow) are left
    out before adding: pandas adds NaN as 0, which would report every
    scenario of such a trial as 0.

    Medians do not add across types; means do. With many trials the
    ``prob_median`` column need not sum to 1, while ``prob_mean`` does.

    Parameters
    ----------
    probs_all : pd.DataFrame
        Per-trial rows with ``trial``, ``ID``, ``scenario`` and ``prob``,
        as in ``*_TRICERATOPS_scenarios_per_trial.csv``. ``trial`` must
        name each trial uniquely; merge_triceratops_pieces.read_piece
        renumbers each piece's trials to their place in the full sequence,
        so merged tables qualify.

    Returns
    -------
    dict
        ``by_type``: one row per scenario type with the columns of
        :data:`SCENARIO_STATS_COLUMNS`, largest median first. ``by_star``:
        ``ID`` plus the same columns, one row per star and type, in
        TRICERATOPS order. ``top_scenario``, ``top_scenario_prob``,
        ``top_scenario_star`` (TIC ID as text), ``top_scenario_star_prob``:
        None or NaN when no trial is usable. ``n_trials_used``.

    Raises
    ------
    KeyError
        If a required column is missing.
    ValueError
        If a star and scenario repeat within one trial (trial numbers not
        unique), or if the usable trials do not all carry the same stars
        and scenarios (they cannot come from one set of inputs).

    Examples
    --------
    >>> t = pd.DataFrame({"trial": [0] * 4 + [1] * 4,
    ...                   "ID": [5, 7, 8, 9] * 2,
    ...                   "scenario": ["TP", "NEBx2P", "NEBx2P", "NEBx2P"] * 2,
    ...                   "prob": [0.019, 0.98, 0.001, 0.0,
    ...                            0.027, 0.97, 0.003, 0.0]})
    >>> s = summarize_scenarios(t)
    >>> [(r.scenario, round(float(r.prob_median), 4)) for r in s["by_type"].itertuples()]
    [('NEBx2P', 0.977), ('TP', 0.023)]
    >>> s["top_scenario"], s["top_scenario_star"], round(s["top_scenario_star_prob"], 4)
    ('NEBx2P', '7', 0.975)
    >>> len(s["by_star"]), s["n_trials_used"]
    (4, 2)
    >>> nan_trial = t.assign(prob=np.where(t["trial"] == 1, np.nan, t["prob"]))
    >>> summarize_scenarios(nan_trial)["n_trials_used"]
    1
    >>> s0 = summarize_scenarios(t.iloc[:0])
    >>> s0["top_scenario"] is None, len(s0["by_type"]), s0["n_trials_used"]
    (True, 0, 0)
    >>> two = pd.concat([t[t["trial"] == 0], t[t["trial"] == 1].assign(trial=0)])
    >>> summarize_scenarios(two)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: trial numbers repeat
    """
    need = ["trial", "ID", "scenario", "prob"]
    missing = [c for c in need if c not in probs_all.columns]
    if missing:
        raise KeyError(f"per-trial scenario table is missing {missing}")
    stats = list(SCENARIO_STATS_COLUMNS)
    dtypes = {"ID": "int64", "scenario": object, "prob_median": float,
              "prob_mean": float, "prob_std": float, "n_trials": "int64"}
    empty: dict[str, Any] = {
        "by_type": pd.DataFrame({c: pd.Series(dtype=dtypes[c]) for c in stats}),
        "by_star": pd.DataFrame(
            {c: pd.Series(dtype=dtypes[c]) for c in ["ID"] + stats}),
        "top_scenario": None, "top_scenario_prob": float("nan"),
        "top_scenario_star": None, "top_scenario_star_prob": float("nan"),
        "n_trials_used": 0}

    df = probs_all.loc[:, need].copy()
    if df.empty:
        return empty
    df["ID"] = pd.to_numeric(df["ID"], errors="raise").astype("int64")
    df["scenario"] = df["scenario"].astype(str)
    df["prob"] = pd.to_numeric(df["prob"], errors="coerce").astype(float)

    dup = df.duplicated(subset=["trial", "ID", "scenario"], keep=False)
    if dup.any():
        raise ValueError(
            f"{int(dup.sum())} rows repeat a star and scenario within one "
            f"trial; trial numbers must be unique (renumber merged pieces "
            f"as merge_triceratops_pieces.read_piece does)")

    bad = pd.Series(~np.isfinite(df["prob"].to_numpy()), index=df.index)
    df = df[~bad.groupby(df["trial"]).transform("any").to_numpy(dtype=bool)]
    if df.empty:
        return empty

    n_used = int(df["trial"].nunique())
    per_pair = df.groupby(["ID", "scenario"], sort=False).size()
    if bool((per_pair != n_used).any()):
        raise ValueError(
            "the usable trials do not all carry the same stars and "
            "scenarios, so they cannot come from one set of inputs")

    totals = (df.groupby(["trial", "scenario"], sort=False)["prob"].sum()
              .reset_index())
    by_type = (totals.groupby("scenario", sort=False)["prob"]
               .agg(prob_median="median", prob_mean="mean",
                    prob_std="std", n_trials="count")
               .reset_index())
    by_type["prob_std"] = by_type["prob_std"].fillna(0.0)
    by_type = (by_type.sort_values("prob_median", ascending=False,
                                   kind="mergesort")
               .reset_index(drop=True)[stats])

    by_star = (df.groupby(["ID", "scenario"], sort=False)["prob"]
               .agg(prob_median="median", prob_mean="mean",
                    prob_std="std", n_trials="count")
               .reset_index())
    by_star["prob_std"] = by_star["prob_std"].fillna(0.0)
    by_star = by_star[["ID"] + stats]

    top_name = str(by_type["scenario"].iloc[0])
    cand = by_star[by_star["scenario"] == top_name].reset_index(drop=True)
    k = int(np.argmax(cand["prob_median"].to_numpy(dtype=float)))
    return {"by_type": by_type, "by_star": by_star,
            "top_scenario": top_name,
            "top_scenario_prob": float(by_type["prob_median"].iloc[0]),
            "top_scenario_star": str(int(cand["ID"].iloc[k])),
            "top_scenario_star_prob": float(cand["prob_median"].iloc[k]),
            "n_trials_used": n_used}

def classify_triceratops(
    fpp: Any,
    nfpp: Any,
    n_below_zero: int,
    can_host: bool,
    n_other_hosts: int,
    exptime_d: float,
) -> dict[str, Any]:
    """
    Outcome, label and poorly-vetted flag for one target, from all of its
    trials. One function, so a single run and the merge of its pieces
    label a target identically (rules decided 10-15-2026).

    Rules, in this order (percentiles as in :func:`summarize_trials`,
    numpy's default linear interpolation):

    1. Every trial NaN, a point below zero in the handed light curve:
       "inconclusive (evidence of negative flux)", outcome "poorly vetted".
    2. Every trial NaN, no such point: "inconclusive (NaN FPP: evidence
       underflow)", outcome :data:`OUTCOME_ALL_NAN`.
    3. Target cannot make the dip: "likely nearby false positive (target
       cannot make the dip)", or "inconclusive (no star in the field can
       make the dip)" when no other star can either.
    4. Median NFPP above 0.1: "likely nearby false positive".
    5. 16th-percentile FPP above 0.5 (about 17 of 20 trials above 0.5):
       "false positive favored".
    6. 84th-percentile FPP below 0.5 and 84th-percentile NFPP below 0.001:
       "likely planet". Above :data:`VALIDATION_EXPTIME_MAX_D` the label
       says the validation tier is withheld (Giacalone et al. 2021
       Section 4.2); at shorter cadence the 1.2.0 tier is kept, "validated
       planet" when 84th-percentile FPP is also below 0.015.
    7. 84th-percentile FPP below 0.5 otherwise: "inconclusive (nearby star
       not ruled out)". Every such target is reviewed by hand before the
       catalog is final; none is discarded without that review.
    8. Everything else: "inconclusive (trials split)".

    The poorly-vetted flag is True whenever a point lies below zero,
    whatever the label.

    Parameters
    ----------
    fpp, nfpp : array-like
        Per-trial FPP and NFPP; NaN for NaN or crashed trials. Only trials
        with finite FPP enter the percentiles.
    n_below_zero : int
        From :func:`count_negative_flux` on the handed light curve.
    can_host : bool
        ``target_host_check(...)["can_host"]``.
    n_other_hosts : int
        Other stars that can make the dip.
    exptime_d : float
        Exposure time in days.

    Returns
    -------
    dict
        ``outcome``, ``disposition``, ``poorly_vetted``.

    Examples
    --------
    >>> nan2 = [np.nan, np.nan]
    >>> classify_triceratops(nan2, nan2, 1, True, 6, 0.0208)["disposition"]
    'inconclusive (evidence of negative flux)'
    >>> c = classify_triceratops(nan2, nan2, 0, True, 6, 0.0208)
    >>> c["outcome"], c["disposition"]
    ('every trial NaN, no below-zero points', 'inconclusive (NaN FPP: evidence underflow)')
    >>> classify_triceratops([1.0, 1.0], [1.0, 1.0], 0, False, 2, 0.0208)["disposition"]
    'likely nearby false positive (target cannot make the dip)'
    >>> classify_triceratops([1.0], [np.nan], 0, False, 0, 0.0208)["disposition"]
    'inconclusive (no star in the field can make the dip)'
    >>> classify_triceratops([0.993, 0.993], [0.990, 0.953], 0, True, 1, 0.0208)["disposition"]
    'likely nearby false positive'
    >>> classify_triceratops([0.59537, 0.67453], [0.04592, 0.05334], 0, True, 4, 0.0208)["disposition"]
    'false positive favored'
    >>> classify_triceratops([0.0521, 0.01077], [0.0, 0.0], 0, True, 0, 0.0208)["disposition"]
    'likely planet (validation tier withheld at 30 min cadence)'
    >>> classify_triceratops([0.05, 0.10], [0.002, 0.003], 0, True, 1, 0.0208)["disposition"]
    'inconclusive (nearby star not ruled out)'
    >>> classify_triceratops([0.2, 0.9], [0.0, 0.0], 0, True, 0, 0.0208)["disposition"]
    'inconclusive (trials split)'
    >>> c = classify_triceratops([0.05, 0.10], [0.0, 0.0], 2, True, 0, 0.0208)
    >>> c["outcome"], c["poorly_vetted"], c["disposition"]
    ('computed', True, 'likely planet (validation tier withheld at 30 min cadence)')
    """
    f = np.asarray(fpp, dtype=float).ravel()
    n = np.asarray(nfpp, dtype=float).ravel()
    flagged = bool(int(n_below_zero) > 0)
    finite = np.isfinite(f)
    if can_host and not finite.any():
        if flagged:
            return {"outcome": OUTCOME_POORLY_VETTED,
                    "disposition": "inconclusive (evidence of negative flux)",
                    "poorly_vetted": True}
        return {"outcome": OUTCOME_ALL_NAN,
                "disposition": "inconclusive (NaN FPP: evidence underflow)",
                "poorly_vetted": False}
    if not can_host:
        label = ("likely nearby false positive (target cannot make the dip)"
                 if int(n_other_hosts) > 0 else
                 "inconclusive (no star in the field can make the dip)")
        return {"outcome": OUTCOME_CANNOT_HOST, "disposition": label,
                "poorly_vetted": flagged}
    fs = summarize_trials(f[finite], "FPP")
    ns = summarize_trials(n[finite], "NFPP")
    if ns["NFPP_p50"] > 0.1:
        label = "likely nearby false positive"
    elif fs["FPP_p16"] > 0.5:
        label = "false positive favored"
    elif fs["FPP_p84"] < 0.5 and ns["NFPP_p84"] < 1e-3:
        if float(exptime_d) > VALIDATION_EXPTIME_MAX_D:
            label = ("likely planet (validation tier withheld at "
                     f"{float(exptime_d) * 1440.0:.0f} min cadence)")
        elif fs["FPP_p84"] < 0.015:
            label = "validated planet"
        else:
            label = "likely planet"
    elif fs["FPP_p84"] < 0.5:
        label = "inconclusive (nearby star not ruled out)"
    else:
        label = "inconclusive (trials split)"
    return {"outcome": OUTCOME_COMPUTED, "disposition": label,
            "poorly_vetted": flagged}


def target_star_params(stars: pd.DataFrame,
                       require: bool = True) -> dict[str, float]:
    """
    The target's mass, radius, Teff and parallax as TRICERATOPS will use
    them: row 0 of ``target.stars``, which :func:`target_host_check` has
    confirmed is the target.

    TRICERATOPS' calc_probs stops early when the target lacks any of the
    four, then reports equal placeholder probabilities (FPP 0.8 with 15
    scenarios), a silent wrong result. With ``require`` True this refuses
    before calc_probs instead. The values are recorded in the summary
    either way.

    Parameters
    ----------
    stars : pd.DataFrame
        ``target.stars`` after ``calc_depths``.
    require : bool
        Raise if any of :data:`STAR_PARAM_COLUMNS` is missing or not finite.

    Returns
    -------
    dict
        ``mass``, ``rad``, ``Teff``, ``plx`` (NaN where missing).

    Raises
    ------
    ValueError
        If ``require`` and a value is missing.

    Examples
    --------
    >>> s = pd.DataFrame({"ID": [7, 8], "mass": [0.4, np.nan],
    ...                   "rad": [0.38, 1.0], "Teff": [3500.0, 5000.0],
    ...                   "plx": [25.0, 1.0]})
    >>> target_star_params(s)
    {'mass': 0.4, 'rad': 0.38, 'Teff': 3500.0, 'plx': 25.0}
    >>> bool(np.isnan(target_star_params(s.assign(plx=[np.nan, 1.0]), require=False)["plx"]))
    True
    >>> target_star_params(s.assign(plx=[np.nan, 1.0]))  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: target lacks plx
    """
    row = stars.iloc[0]
    out: dict[str, float] = {}
    for col in STAR_PARAM_COLUMNS:
        try:
            out[col] = (float(row[col]) if col in stars.columns
                        else float("nan"))
        except (TypeError, ValueError):
            out[col] = float("nan")
    missing = [c for c, v in out.items() if not np.isfinite(v)]
    if require and missing:
        raise ValueError(
            f"target {row.get('ID', '?')} lacks {missing} in target.stars; "
            f"triceratops would stop early and report equal placeholder "
            f"probabilities. Refusing.")
    return out


def run_triceratops_on_nemesis(
    tic_id: int | None = None,
    sector: int | None = None,
    report_dir: str | None = None,
    savelcpath: str | None = None,
    bestfit_path: str | None = None,
    hdu: Any = None,
    LC_df: pd.DataFrame | None = None,
    lightkurve_cache_dir: str | None = None,
    outdir: str | None = None,
    trilegal_dir: str | None = None,
    n_trials: int = 20,
    flux_type: str = "Detrended",
    search: str = "TLS",
    n_durations: int = N_DURATIONS_DEFAULT,
    n_bins: int | None = None,
    exptime: float | None = None,
    mc_samples: int = 5_000_000,
    parallel: bool = False,
    rp_max_earth: float | None = None,
    rp_column: str | None = None,
    nemesis_fcr: float | None = None,
    update_bestfit: bool = True,
    save_plots: bool = True,
    append_to_vetting_report: bool = True,
    vetting_report_path: str | None = None,
    report_layout: str = "auto",
    report_split: bool = False,
    report_base_pages: int | None = 6,
    seed: int | None = None,
    verbose: bool = True,
    posterior_path: str | None = None,
) -> dict[str, Any]:
    """
    Run TRICERATOPS statistical validation on one NEMESIS star-sector, from
    existing transit-search products through to catalog-ready columns.

    Four design choices worth knowing.

    **Inputs are converted to aperture units (1.3.0).** The NEMESIS light
    curve is deblended: its dip is in units of the target's own flux.
    TRICERATOPS expects the dip as the whole aperture recorded it and
    divides by each host's share itself, so the light curve, its error bar
    and the depth are multiplied back by NEMESIS' own 1 - FCR
    (:func:`convert_to_aperture_units`) before TRICERATOPS sees them. If
    TRICERATOPS' light model then gives the target a depth of 0 (it would
    need more than 100% of its light), ``calc_probs`` is not called,
    because it would treat the next possible host as the target. The
    result is FPP = 1 and NFPP = 1 by definition, and
    ``TRICERATOPS_outcome`` says so (:func:`cannot_host_trials`,
    10-14-2026).

    **Median is primary.** ``calc_probs`` evaluates scenario likelihoods by
    Monte Carlo, so repeated runs on identical inputs differ. The median
    across trials is the headline value, matching the manuscript
    specification, with mean and trial-to-trial standard deviation recorded
    alongside.

    **All scenarios are kept.** The per-scenario ``prob`` values are
    posterior probabilities, not false-positive probabilities, so the most
    likely scenario is the one with the HIGHEST probability. Rather than
    reduce to a winner, the median probability of every scenario is pivoted
    into ``TRI_prob_<SCENARIO>`` columns, with the argmax recorded
    separately for convenience.

    **TRILEGAL is cached, never re-queried.** The background population
    depends on the galactic sightline, not the candidate. A file at
    ``{trilegal_dir}/{ID}_TRILEGAL.csv`` is reused when present; only when
    absent is the network touched, and then once. This is what makes the
    function usable on compute nodes with no outbound network: pre-stage the
    files and no trial hits the web.

    Parameters
    ----------
    tic_id, sector : int, optional
        Target identity. Required unless ``bestfit_path`` is given, in which
        case both are parsed from its filename.
    report_dir : str, optional
        Directory holding bestfit CSVs, normally
        ``settings['saveReportpath_TLS']``. Required unless ``bestfit_path``
        is given.
    savelcpath : str
        Directory holding the persisted masks and, if ``LC_df`` is None, the
        final light curve CSV. Required.
    bestfit_path : str, optional
        Explicit path to the bestfit CSV, bypassing the search.
    hdu : fits.HDUList or str, optional
        Open HDUList or path to the TESScut FITS. Required: the CCD offset
        keywords cannot be recovered from anything else.
    LC_df : pd.DataFrame, optional
        Light curve table. Loaded from ``savelcpath`` when None.
    lightkurve_cache_dir : str, optional
        Cache directory for the TESScut cutout TRICERATOPS downloads for
        ITSELF. This is a different file from ``hdu``: ``hdu`` supplies four
        header keywords for the aperture conversion and never reaches
        TRICERATOPS, while TRICERATOPS needs pixel data of its own for the
        PSF flux fractions in ``calc_depths`` and the image panel in
        ``plot_field``. Point this at ``settings['downloadpath']`` to reuse
        the NEMESIS cutout, which works only when the requested cutout sizes
        match and ``shard_cache_by_tic`` is False. Otherwise give it a
        dedicated directory. Omitting it means a fresh download on every
        target construction, which fails outright on compute nodes with no
        outbound network.
    outdir : str, optional
    trilegal_dir : str, optional
        Directory for cached TRILEGAL files. Defaults to ``outdir``.
    n_trials : int
        Independent ``calc_probs`` realizations. Manuscript macro ``NTRI``.
    flux_type : str
        ``'SAP'``, ``'PLD'`` or ``'Detrended'``.
    search : str
        ``'TLS'`` or ``'BLS'``.
    n_durations : int
        Half-width of the fold window, in transit durations. Default
        :data:`N_DURATIONS_DEFAULT` (3, decided 10-14-2026). Recorded as
        ``TRICERATOPS_n_durations``.
    n_bins : int or None
        ``None`` means no binning, correct for 30-minute FFI cadence.
    exptime : float, optional
        Exposure time in DAYS. Derived from the median cadence spacing of
        the raw time array when omitted, which is the production setting.
    mc_samples : int
        ``calc_probs`` parameter ``N`` per trial. Default 5,000,000.
    parallel : bool
        Passed to ``calc_probs``. Keep False under disBatch: each call
        spawns a worker pool, and many tasks per node each spawning one
        reproduces the thread-oversubscription pattern that amplified the
        July 2026 heap-corruption crashes. Parallelize across candidates.
    rp_max_earth : float or None
        Radius ceiling in Earth radii. Candidates above it return early
        with ``skipped='radius_ceiling'``. Default None: no ceiling, every
        candidate is evaluated (09-29 decision). Candidates above
        :data:`RP_FLAG_RE` are flagged in ``TRICERATOPS_Rp_flag_gt_8``,
        because above roughly that radius giant planets, brown dwarfs and
        the lowest-mass stars overlap and depth alone no longer
        discriminates.
    rp_column : str, optional
        Bestfit radius column. Auto-detected from
        :data:`RP_COLUMN_CANDIDATES`.
    nemesis_fcr : float, optional
        NEMESIS flux contamination ratio. Since 1.3.0 it converts the
        inputs (1 - FCR) as well as feeding the contamination cross-check.
        Read from the bestfit table with :func:`read_nemesis_fcr` when
        omitted; a missing or unusable value raises.
    update_bestfit : bool
        Append ``TRICERATOPS_*`` and ``TRI_prob_*`` columns to the bestfit
        CSV, written to a temp file and renamed so an interrupt cannot
        truncate a primary survey product.
    save_plots : bool
        Write the field and scenario-fit figures. The scenario-fit figure
        is skipped when the target cannot make the dip.
    seed : int, optional
        Seeds ``numpy.random`` before the trial loop. TRICERATOPS also
        samples internally, so this constrains reproducibility only
        partially. Recorded for provenance either way.
    verbose : bool
        Print progress.
    posterior_path : str, optional
        jaxoplanet ``*_jaxoplanet_posterior.csv`` for this star-sector.
        When given, period, mid-transit time and duration come from its
        posterior medians and the radius from ``MCMC Planet Radius [RE]``.
        The depth stays the search depth from the bestfit. When omitted the
        search ephemeris is used, which keeps library calls such as the
        TOI-465 reference reproduction working.

    Returns
    -------
    dict
        ``tic_id``, ``sector``, ``skipped``, ``FPP_median``, ``FPP_mean``,
        ``FPP_std``, ``NFPP_median``, ``NFPP_mean``, ``NFPP_std``,
        ``n_trials``, ``disposition``, ``top_scenario``,
        ``top_scenario_prob``, ``scenario_probs``, ``trials_df``,
        ``probs_df``, ``fluxratio_target``, ``nemesis_fcr``,
        ``fcr_residual``, ``Rp_RE``, ``files``, ``inputs``, ``target``,
        since 1.3.0 ``outcome`` and ``host_check``, and since part 2
        ``classification``, ``scenario_summary``, ``negative_flux`` and
        ``star_params``.

    Raises
    ------
    ValueError
        If required paths are missing, no bestfit is found, no trial is
        finite while at least one raised (every trial NaN is recorded as a
        result since part 2), the bestfit FCR is unusable, the target is
        not row 0 of ``target.stars``, or the target lacks mass, radius,
        Teff or parallax.
    KeyError
        If the bestfit table has no FCR column.
    RuntimeError
        If a trial's target scenarios carry another star's ID, or its
        probabilities do not sum to 1.
    FileNotFoundError
        If the bestfit CSV, light curve or aperture mask is absent.

    Examples
    --------
    >>> res = run_triceratops_on_nemesis(
    ...     tic_id=270380593, sector=4,
    ...     report_dir=settings['saveReportpath_TLS'],
    ...     savelcpath=settings['savelcpath'],
    ...     hdu=hdu, n_trials=20)
    >>> print(res["FPP_median"], res["disposition"])
    """
    # import triceratops.triceratops as tr
    from nemesis_tess.compat import load_triceratops
    tr = load_triceratops()
#     from triceratops.funcs import save_trilegal

    t_start = time.time()
    mc_samples = int(mc_samples)

    if savelcpath is None:
        raise ValueError("savelcpath is required")
    if hdu is None:
        raise ValueError(
            "hdu is required: the cutout-to-CCD pixel offset lives in the "
            "TESScut header and cannot be recovered from anything else. Pass "
            "the HDUList from full_pipeline, or a path to the cached FITS."
        )

    if bestfit_path is None:
        if report_dir is None or tic_id is None:
            raise ValueError(
                "pass bestfit_path, or both report_dir and tic_id")
        hits = find_nemesis_bestfit(report_dir, tic_id, sector=sector,
                                    search=search, verbose=verbose)
        if not hits:
            raise ValueError(
                f"no {search} bestfit found for TIC {tic_id} "
                f"(sector={sector}) under {report_dir}")
        if len(hits) > 1 and sector is None:
            raise ValueError(
                f"{len(hits)} bestfit files match TIC {tic_id}; pass sector= "
                f"or bestfit_path= to disambiguate:\n  " + "\n  ".join(hits))
        bestfit_path = hits[0]

    bestfit_path = os.path.abspath(bestfit_path)
    if not os.path.isfile(bestfit_path):
        raise FileNotFoundError(f"bestfit CSV not found: {bestfit_path}")
    bestfit_df = pd.read_csv(bestfit_path)

    base = os.path.basename(bestfit_path)
    m_tic = re.search(r"TIC[_ ]?(\d+)", base)
    m_sec = re.search(r"Sector[_ ]?(\d+)", base)
    if m_tic:
        tic_id = int(m_tic.group(1))
    if m_sec:
        sector = int(m_sec.group(1))

    if LC_df is None:
        LC_df = find_nemesis_lightcurve(savelcpath, tic_id, sector,
                                        verbose=verbose)

    hdulist, opened_here = open_hdu(hdu)

    try:
        if outdir is None:
            outdir = os.path.join(os.path.dirname(bestfit_path), "TRICERATOPS")
        os.makedirs(outdir, exist_ok=True)
        if trilegal_dir is None:
            trilegal_dir = outdir
        os.makedirs(trilegal_dir, exist_ok=True)

        inp = nemesis_to_triceratops_inputs(
            hdu=hdulist, LC_df=LC_df, bestfit_df=bestfit_df,
            savelcpath=savelcpath, bestfit_path=bestfit_path,
            flux_type=flux_type, search=search,
            n_durations=n_durations, n_bins=n_bins, verbose=verbose,
            posterior_path=posterior_path,
        )

        tic_id, sector = inp["tic_id"], inp["sector"]
        tag = f"TIC_{tic_id}_Sector_{sector:04d}"

        # ------------------------------------------- conversion (1.3.0)
        # Undo NEMESIS' deblend with the number it deblended with, so
        # TRICERATOPS sees the dip the aperture recorded (10-14-2026).
        if nemesis_fcr is None:
            nemesis_fcr, fcr_column = read_nemesis_fcr(bestfit_df)
        else:
            nemesis_fcr, fcr_column = float(nemesis_fcr), "passed in"
        inp = convert_to_aperture_units(inp, 1.0 - nemesis_fcr)
        # 1.3.0 part 2 (10-15-2026): count points below zero in the
        # handed light curve. A flag only; nothing is removed.
        neg = count_negative_flux(inp["flux"])
        if verbose:
            print(f"conversion    : 1 - FCR = {inp['conversion_share']:.6f} "
                  f"({fcr_column}); dip handed "
                  f"{inp['tdepth'] * 1e3:.4f} ppt (survey "
                  f"{inp['tdepth_survey'] * 1e3:.3f} ppt); window "
                  f"+/- {n_durations} durations")

        # TRICERATOPS defaults exptime to 0.00139 d, which is 2 minutes: the
        # SPOC short-cadence value. At 30-minute FFI cadence a transit is
        # smeared across a full cadence, so comparing it against sharp-edged
        # 2-minute models breaks the shape discriminator that separates a
        # planet from an eclipsing binary, and FPP becomes multi-modal.
        # Derived from the raw, unfolded time array rather than assumed.
        if exptime is None:
            t_raw = np.asarray(LC_df["Time"], dtype=float)
            t_raw = t_raw[np.isfinite(t_raw)]
            dt = np.median(np.diff(np.sort(t_raw))) if t_raw.size > 2 else np.nan
            exptime = float(dt) if np.isfinite(dt) and dt > 0 else 30.0 / 1440.0
        if verbose:
            print(f"exptime       : {exptime:.6f} d "
                  f"({exptime * 1440.0:.2f} min)")
            print(f"MC samples    : {mc_samples:,}")

        # ------------------------------------------------------------ radius
        # The jaxoplanet radius is preferred when a posterior was passed;
        # the bestfit search radius is used otherwise. An explicit
        # rp_column still wins. rp_column records which source was used.
        rp_value = np.nan
        post_rp = float(inp.get("Rp_posterior_RE", np.nan))
        if rp_column is None and np.isfinite(post_rp):
            rp_column = "MCMC Planet Radius [RE]"
            rp_value = post_rp
        else:
            if rp_column is None:
                for c in RP_COLUMN_CANDIDATES:
                    if c in bestfit_df.columns:
                        rp_column = c
                        break
            if rp_column is not None and rp_column in bestfit_df.columns:
                try:
                    rp_value = float(bestfit_df.iloc[0][rp_column])
                except (TypeError, ValueError):
                    rp_value = np.nan
                    #
        if rp_max_earth is not None:
            if not np.isfinite(rp_value):
                if verbose:
                    print(f"[warn] no usable radius column "
                          f"(tried {RP_COLUMN_CANDIDATES}); ceiling NOT "
                          f"applied")
            elif rp_value > rp_max_earth:
                if verbose:
                    print(f"SKIP {tag}: Rp = {rp_value:.2f} Re exceeds the "
                          f"{rp_max_earth:.1f} Re ceiling.")
                return {
                    "tic_id": tic_id, "sector": sector,
                    "skipped": "radius_ceiling", "Rp_RE": rp_value,
                    "FPP_median": np.nan, "FPP_mean": np.nan,
                    "FPP_std": np.nan, "NFPP_median": np.nan,
                    "NFPP_mean": np.nan, "NFPP_std": np.nan, "n_trials": 0,
                    "disposition": "not evaluated (Rp ceiling)",
                    "top_scenario": None, "top_scenario_prob": np.nan,
                    "scenario_probs": {}, "trials_df": None, "probs_df": None,
                    "fluxratio_target": np.nan, "nemesis_fcr": np.nan,
                    "fcr_residual": np.nan, "files": {}, "inputs": inp,
                    "target": None,
                }

        # ---------------------------------------------------------- TRILEGAL
        # The canonical location, plus the places an earlier interactive
        # save_trilegal call may have dropped the file. That helper takes no
        # path argument and writes {ID}_TRILEGAL.csv to the working
        # directory, so a file fetched by hand rarely sits where the default
        # cache lookup expects it.
        trilegal_name = f"{tic_id}_TRILEGAL.csv"
        trilegal_path = os.path.join(trilegal_dir, trilegal_name)

        search_dirs = [trilegal_dir, os.getcwd(), outdir,
                       os.path.dirname(bestfit_path)]
        found = None
        for cand_dir in search_dirs:
            cand = os.path.join(cand_dir, trilegal_name)
            if os.path.isfile(cand) and os.path.getsize(cand) > 0:
                found = os.path.abspath(cand)
                break

        # Copy rather than move. os.replace would relocate the canonical
        # cached file into whatever outdir this run happens to use, so a
        # throwaway run (a convergence test, a parameter sweep) would steal
        # the cache and force the next production run back onto the network.
        if found is not None and os.path.abspath(trilegal_path) != found:
            shutil.copy2(found, trilegal_path)
            if verbose:
                print(f"TRILEGAL      : copied {found} -> {trilegal_path}")
            found = trilegal_path

        cached = found is not None

        # Built once and shared by both construction paths. Omitted rather
        # than passed as None so triceratops falls back to its own default
        # when no cache directory is given.
        target_kwargs: dict[str, Any] = {}
        if lightkurve_cache_dir is not None:
            target_kwargs["lightkurve_cache_dir"] = lightkurve_cache_dir
            if verbose:
                print(f"lk cache dir  : {lightkurve_cache_dir}")

        if cached:
            if verbose:
                print(f"TRILEGAL      : cached, {trilegal_path}")
            target = tr.target(ID=tic_id, sectors=np.array([sector]),
                               trilegal_fname=trilegal_path, **target_kwargs)
            target.trilegal_fname = trilegal_path
        else:
            if verbose:
                print(f"TRILEGAL      : not cached in "
                      f"{[os.path.abspath(p) for p in search_dirs]}, "
                      f"submitting one job")
            with trilegal_ssl_workaround(verbose=verbose):
                target = tr.target(ID=tic_id, sectors=np.array([sector]),
                                   **target_kwargs)
            # Outside the workaround block above, because
            # save_trilegal_when_ready opens its own. The job needs time to
            # run; reading the URL immediately returns an empty document.
            written = save_trilegal_when_ready(
                getattr(target, "trilegal_url", None), tic_id,
                verbose=verbose)
            if written and os.path.isfile(str(written)):
                if os.path.abspath(str(written)) != os.path.abspath(trilegal_path):
                    os.replace(str(written), trilegal_path)
                target.trilegal_fname = trilegal_path
                if verbose:
                    print(f"TRILEGAL      : saved to {trilegal_path}")
            else:
                trilegal_path = str(written)
                if verbose:
                    print(f"[warn] save_trilegal returned {written!r}; later "
                          f"trials may re-query the network")

        if verbose:
            print(f"stars in field: {len(target.stars)}")

        # ------------------------------------------------------------ depths
        # 1.3.0: the depth handed is in aperture units (converted above).
        target.calc_depths(tdepth=inp["tdepth"],
                           all_ap_pixels=inp["all_ap_pixels"])

        stars = target.stars.copy()
        id_col = "ID" if "ID" in stars.columns else stars.columns[0]

        # 1.3.0 guard: the target must be row 0, and triceratops decides
        # whether it can make the dip. Raises if the target is not row 0.
        host = target_host_check(stars, tic_id, inp["tdepth"])
        # 1.3.0 part 2: triceratops stops early, with equal placeholder
        # probabilities, if the target lacks mass, radius, Teff or
        # parallax. Refuse before calc_probs instead; record the values.
        star_params = target_star_params(stars, require=host["can_host"])
        fluxratio_target = host["fluxratio_target"]
        fcr_residual = fluxratio_target - (1.0 - nemesis_fcr)
        outcome = OUTCOME_COMPUTED if host["can_host"] else OUTCOME_CANNOT_HOST

        if verbose:
            print(f"\nfluxratio(target)     : {fluxratio_target:.6f}")
            print(f"1 - NEMESIS FCR       : {1.0 - nemesis_fcr:.6f}")
            print(f"residual              : {fcr_residual:+.6f}  "
                  f"(independent contamination cross-check)")
            print(f"needed on target      : "
                  f"{host['needed_depth'] * 1e3:.3f} ppt "
                  f"(dip handed / fluxratio(target))")
            print(f"outcome               : {outcome}; other stars that "
                  f"can make the dip: {len(host['other_hosts'])}")
            if "fluxratio" in stars.columns:
                contam = stars[stars["fluxratio"] > 0.001]
                cols = [c for c in (id_col, "fluxratio", "tdepth")
                        if c in contam.columns]
                print("\ncontributing sources:")
                print(contam[cols].to_string(index=False))

        # -------------------------------------------------------- field plot
        files: dict[str, str] = {}
        if save_plots:
            f_field = os.path.join(outdir, f"{tag}_TRICERATOPS_field.png")
            written_fig = render_to_file(
                lambda: target.plot_field(
                    sector=sector, ap_pixels=inp["ap_pixels"],
                    ap_color="red", save=False, fname=None),
                f_field, verbose=verbose)
            if written_fig:
                files["field_plot"] = written_fig

        # ------------------------------------------------------------ trials
        if seed is not None:
            np.random.seed(seed)

        if host["can_host"]:
            rows: list[dict[str, Any]] = []
            scen_frames: list[pd.DataFrame] = []

            for i in range(n_trials):
                try:
                    target.calc_probs(
                        time=inp["time"], flux_0=inp["flux"],
                        flux_err_0=inp["flux_err"], P_orb=inp["P_orb"],
                        exptime=exptime, N=mc_samples,
                        parallel=parallel, verbose=0,
                    )
                except Exception as exc:
                    if verbose:
                        print(f"  trial {i + 1}/{n_trials}: FAILED ({exc})")
                    rows.append({"trial": i, "FPP": np.nan, "NFPP": np.nan,
                                 "status": f"exception: "
                                           f"{type(exc).__name__}: {exc}"})
                    continue

                fpp_i, nfpp_i = float(target.FPP), float(target.NFPP)
                p = target.probs.copy()
                # 1.3.0 guard: raises (so nothing is written) if the
                # target's scenarios describe another star, or a finite
                # result does not sum to 1.
                check_trial_probs(p, tic_id, fpp_i)
                rows.append({"trial": i, "FPP": fpp_i, "NFPP": nfpp_i,
                             "status": "ok" if np.isfinite(fpp_i) else "nan"})
                p["trial"] = i
                scen_frames.append(p)
                if verbose:
                    print(f"  trial {i + 1}/{n_trials}: FPP {fpp_i:.5f}  "
                          f"NFPP {nfpp_i:.5f}")

            trials_df = pd.DataFrame(rows)
            ok = trials_df["FPP"].notna()
            n_nan = int((trials_df["status"] == "nan").sum())
            all_nan = len(trials_df) > 0 and n_nan == len(trials_df)
            # 1.3.0 part 2 (10-15-2026): every trial NaN is a result,
            # labeled by classify_triceratops. No finite trial with at
            # least one crash still stops, so the retry pass reruns the
            # piece: a crash is not an answer.
            if not ok.any() and not all_nan:
                raise ValueError(
                    f"no finite calc_probs trial for {tag}: {n_nan} "
                    f"returned NaN FPP, {len(trials_df) - n_nan} raised")
            probs_all = (pd.concat(scen_frames, ignore_index=True)
                         if scen_frames else
                         pd.DataFrame(columns=list(TRIAL_PROBS_COLUMNS)))
        else:
            # 1.3.0: calc_probs is not called; see cannot_host_trials.
            trials_df = cannot_host_trials(n_trials, host["other_hosts"])
            ok = trials_df["FPP"].notna()
            probs_all = pd.DataFrame(columns=list(TRIAL_PROBS_COLUMNS))
            if verbose:
                print(f"  calc_probs not called: {OUTCOME_CANNOT_HOST}. "
                      f"FPP 1 and NFPP {trials_df['NFPP'].iloc[0]} set by "
                      f"definition for {n_trials} trial(s).")

        # 1.3.0 part 2 (10-15-2026): scenario types summed over stars
        # within each trial, medians over trials, a per-star table, and
        # the top type with its leading star (summarize_scenarios).
        scen = summarize_scenarios(probs_all)
        agg = scen["by_type"]
        f_star = os.path.join(
            outdir, f"{tag}_TRICERATOPS_scenarios_per_star.csv")
        scen["by_star"].to_csv(f_star, index=False)
        files["scenarios_per_star_csv"] = f_star

        fpp = trials_df.loc[ok, "FPP"].to_numpy(dtype=float)
        nfpp = trials_df.loc[ok, "NFPP"].to_numpy(dtype=float)

        fpp_stats = summarize_trials(fpp, "FPP")
        nfpp_stats = summarize_trials(nfpp, "NFPP")

        fpp_med = fpp_stats["FPP_p50"]
        fpp_mean = fpp_stats["FPP_mean"]
        fpp_std = fpp_stats["FPP_std"]
        nfpp_med = nfpp_stats["NFPP_p50"]
        nfpp_mean = nfpp_stats["NFPP_mean"]
        nfpp_std = nfpp_stats["NFPP_std"]

        # --------------------------------------------------------- scenarios
        finite = agg[np.isfinite(agg["prob_median"])]
        top_scenario = str(finite.iloc[0]["scenario"]) if len(finite) else None
        top_prob = float(finite.iloc[0]["prob_median"]) if len(finite) else np.nan
        scenario_probs = {str(r.scenario): float(r.prob_median)
                          for r in agg.itertuples()
                          if np.isfinite(r.prob_median)}

        # Labels (1.3.0 part 2, 10-15-2026): classify_triceratops applies
        # the eight rules in order to all of this run's trials. The
        # validated-planet tier is withheld above VALIDATION_EXPTIME_MAX_D:
        # Giacalone et al. (2021) Section 4.2 found no region of the
        # FPP-NFPP plane in which planets could be validated at 30-minute
        # cadence, so long-cadence results are for vetting and follow-up
        # prioritization, not validation.
        long_cadence = float(exptime) > VALIDATION_EXPTIME_MAX_D
        cls = classify_triceratops(
            trials_df["FPP"].to_numpy(dtype=float),
            trials_df["NFPP"].to_numpy(dtype=float),
            n_below_zero=neg["n_below_zero"], can_host=host["can_host"],
            n_other_hosts=len(host["other_hosts"]),
            exptime_d=float(exptime))
        disposition = cls["disposition"]
        outcome = cls["outcome"]

        # --------------------------------------------------------- fits plot
        # 1.3.0: plotted from the converted light curve calc_probs saw;
        # skipped when calc_probs was not called.
        if save_plots and host["can_host"]:
            f_fits = os.path.join(outdir, f"{tag}_TRICERATOPS_fits.png")
            written_fig = render_to_file(
                lambda: target.plot_fits(
                    time=inp["time"], flux_0=inp["flux"],
                    flux_err_0=inp["flux_err"]),
                f_fits, verbose=verbose)
            if written_fig:
                files["fits_plot"] = written_fig
        #
        # ------------------------------------------- vetting report page
        # Appended in place so the NEMESIS filename convention and location
        # are unchanged. Idempotent via keep_pages=6: a repeat run truncates
        # back to the six original pages rather than growing the PDF.
        #
        # Deliberately not routed through NEMESIS append_PDF: that helper
        # rebuilds the PDF from a list of page PNGs, writes to a filename it
        # constructs itself, and deletes every input but the first. It also
        # cannot run incrementally, since the page2-page6 PNGs are already
        # deleted by the time the report exists.
        if (append_to_vetting_report and save_plots
                and files.get("field_plot") and files.get("fits_plot")):
            if vetting_report_path is None:
                vetting_report_path = os.path.join(
                    os.path.dirname(bestfit_path),
                    f"{tag}_{search}_Report_multi_page.pdf")

            if not os.path.isfile(vetting_report_path):
                if verbose:
                    print(f"[warn] no vetting report at "
                          f"{vetting_report_path}; skipping PDF append")
            else:
                page_size = pdf_page_size_px(vetting_report_path,
                                             dpi=150, verbose=verbose)
                if page_size is None:
                    if verbose:
                        print("[warn] could not read report page size; "
                              "skipping PDF append")
                else:
                    params = [
                        ("Period [d]", f"{inp['P_orb']:.6f}"),
                        ("T0 [BTJD]", f"{inp['T0']:.5f}"),
                        ("Duration [hr]", f"{inp['duration_days'] * 24:.3f}"),
                        ("Depth [ppt]", f"{inp['tdepth_survey'] * 1e3:.3f}"),
                        ("Rp [RE]", f"{rp_value:.3f}"
                                    if np.isfinite(rp_value) else "n/a"),
                        ("Aperture [px]", f"{len(inp['ap_pixels'])}"),
                        ("Flux ratio", f"{fluxratio_target:.5f}"),
                        ("LC points", f"{inp['time'].size}"),
                    ]
                    stats_rows = [
                        ("FPP p50", f"{fpp_med:.5f}"),
                        ("FPP p16", f"{fpp_stats['FPP_p16']:.5f}"),
                        ("FPP p84", f"{fpp_stats['FPP_p84']:.5f}"),
                        ("FPP max", f"{fpp_stats['FPP_max']:.5f}"),
                        ("NFPP p50", f"{nfpp_med:.5f}"),
                        ("NFPP max", f"{nfpp_stats['NFPP_max']:.5f}"),
                        ("Trials", f"{int(ok.sum())}"),
                        ("MC samples", f"{mc_samples:,}"),
                    ]
                    top_scen = [
                        (str(r.scenario), float(r.prob_median),
                         float(r.prob_std))
                        for r in agg.itertuples()
                        if np.isfinite(r.prob_median)
                        and float(r.prob_median) > 1e-4
                    ][:6]

                    page = compose_triceratops_report_page(
                        files["field_plot"], files["fits_plot"],
                        os.path.join(outdir, f"{tag}_TRICERATOPS_page.png"),
                        page_size_px=page_size,
                        subtitle=(f"TIC {tic_id}   |   Sector {sector}   |   "
                                  f"{inp['ephemeris_source']}   |   exptime "
                                  f"{exptime * 1440:.0f} min   |   "
                                  f"module v{__version__}"),
                        params=params,
                        stats=stats_rows,
                        scenarios=top_scen,
                        disposition=disposition.split(" (")[0],
                        footnote=("Per-scenario values are independent "
                                  "medians over trials and do not sum to "
                                  "unity.\nFPP is computed per trial, then "
                                  "summarized.\nClassification criteria from "
                                  "Giacalone et al. (2021);\nthe "
                                  "validated-planet tier is unavailable at "
                                  "FFI cadence (their Section 4.2)."),
                        verbose=verbose)

                    if page:
                        files["report_page_png"] = page
                        appended = append_png_to_pdf(
                            vetting_report_path, page,
                            in_place=True,
                            keep_pages=report_base_pages,
                            verbose=verbose)
                        if appended:
                            files["vetting_report_pdf"] = appended
        #
        # ------------------------------------------------------------ tables
        f_trials = os.path.join(outdir, f"{tag}_TRICERATOPS_trials.csv")
        trials_df.to_csv(f_trials, index=False)
        files["trials_csv"] = f_trials

        f_probs = os.path.join(outdir, f"{tag}_TRICERATOPS_scenarios.csv")
        agg.to_csv(f_probs, index=False)
        files["scenarios_csv"] = f_probs

        f_all = os.path.join(
            outdir, f"{tag}_TRICERATOPS_scenarios_per_trial.csv")
        probs_all.to_csv(f_all, index=False)
        files["scenarios_per_trial_csv"] = f_all

        summary: dict[str, Any] = {
            "TIC_ID": tic_id, "Sector": sector,
            **{f"TRICERATOPS_{k}": v for k, v in fpp_stats.items()},
            **{f"TRICERATOPS_{k}": v for k, v in nfpp_stats.items()},
            "TRICERATOPS_exptime_d": float(exptime),
            "TRICERATOPS_mc_samples": int(mc_samples),
            "TRICERATOPS_long_cadence": bool(long_cadence),
            "TRICERATOPS_validation_tier_available": bool(not long_cadence),
            "TRICERATOPS_median_FPP": fpp_med,
            "TRICERATOPS_mean_FPP": fpp_mean,
            "TRICERATOPS_std_FPP": fpp_std,
            "TRICERATOPS_median_NFPP": nfpp_med,
            "TRICERATOPS_mean_NFPP": nfpp_mean,
            "TRICERATOPS_std_NFPP": nfpp_std,
            "TRICERATOPS_N_trials": int(ok.sum()),
            "TRICERATOPS_N_trials_failed": int((~ok).sum()),
            "TRICERATOPS_disposition": disposition,
            "TRICERATOPS_top_scenario": top_scenario,
            "TRICERATOPS_top_scenario_prob": top_prob,
            "TRICERATOPS_aperture_npix": int(len(inp["ap_pixels"])),
            "TRICERATOPS_LC_points": int(inp["time"].size),
            # 1.3.0: these two keep their 1.2.0 meaning, the survey values;
            # what triceratops received is in the *_handed columns below.
            "TRICERATOPS_flux_err": float(inp["flux_err_survey"]),
            "TRICERATOPS_tdepth_in": float(inp["tdepth_survey"]),
            "TRICERATOPS_fluxratio_target": fluxratio_target,
            "TRICERATOPS_NEMESIS_FCR": float(nemesis_fcr),
            "TRICERATOPS_FCR_residual": fcr_residual,
            "TRICERATOPS_outcome": outcome,
            "TRICERATOPS_n_durations": int(n_durations),
            "TRICERATOPS_conversion_share": float(inp["conversion_share"]),
            "TRICERATOPS_flux_err_handed": float(inp["flux_err"]),
            "TRICERATOPS_tdepth_handed": float(inp["tdepth"]),
            "TRICERATOPS_tdepth_needed_target": float(host["needed_depth"]),
            "TRICERATOPS_n_other_hosts": int(len(host["other_hosts"])),
            "TRICERATOPS_other_hosts": " ".join(host["other_hosts"]),
            "TRICERATOPS_N_trials_nan": int(
                (trials_df["status"] == "nan").sum()),
            "TRICERATOPS_n_flux_below_zero": int(neg["n_below_zero"]),
            "TRICERATOPS_min_flux_handed": float(neg["min_flux"]),
            "TRICERATOPS_poorly_vetted": bool(cls["poorly_vetted"]),
            "TRICERATOPS_top_scenario_star": scen["top_scenario_star"],
            "TRICERATOPS_top_scenario_star_prob":
                scen["top_scenario_star_prob"],
            "TRICERATOPS_N_trials_scenarios": int(scen["n_trials_used"]),
            "TRICERATOPS_star_mass": star_params["mass"],
            "TRICERATOPS_star_rad": star_params["rad"],
            "TRICERATOPS_star_Teff": star_params["Teff"],
            "TRICERATOPS_star_plx": star_params["plx"],
            "TRICERATOPS_Rp_RE": rp_value,
            "TRICERATOPS_Rp_column": rp_column or "",
            "TRICERATOPS_Rp_flag_gt_8": bool(np.isfinite(rp_value)
                                             and rp_value > RP_FLAG_RE),
            "TRICERATOPS_ephemeris_source": inp["ephemeris_source"],
            "TRICERATOPS_posterior_file": os.path.basename(
                inp["posterior_path"] or ""),
            "TRICERATOPS_posterior_converged": inp["posterior_converged"],
            "TRICERATOPS_P_orb_d": float(inp["P_orb"]),
            "TRICERATOPS_T0_BTJD": float(inp["T0"]),
            "TRICERATOPS_duration_hr": float(inp["duration_days"]) * 24.0,
            "TRICERATOPS_trilegal_file": os.path.basename(str(trilegal_path)),
            "TRICERATOPS_lk_cache_dir": (lightkurve_cache_dir or ""),
            "TRICERATOPS_seed": (seed if seed is not None else -1),
            "TRICERATOPS_module_version": __version__,
            "TRICERATOPS_runtime_s": round(time.time() - t_start, 1),
            "TRICERATOPS_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        # Asymmetric uncertainties from the trial quantiles. The trial
        # distribution is right-skewed, so a single +/- value would
        # misrepresent it; these are the p16 and p84 offsets from the median
        # and are what should be quoted in a table.
        summary["TRICERATOPS_FPP_err_lo"] = fpp_med - fpp_stats["FPP_p16"]
        summary["TRICERATOPS_FPP_err_hi"] = fpp_stats["FPP_p84"] - fpp_med
        summary["TRICERATOPS_NFPP_err_lo"] = nfpp_med - nfpp_stats["NFPP_p16"]
        summary["TRICERATOPS_NFPP_err_hi"] = nfpp_stats["NFPP_p84"] - nfpp_med

        # Per-scenario median probability plus its trial-to-trial scatter.
        # The scatter matters: a scenario whose probability swings between
        # trials is a scenario the data does not constrain, which is the
        # case for STP at FFI cadence.
        for r in agg.itertuples():
            if not np.isfinite(r.prob_median):
                continue
            summary[f"TRI_prob_{r.scenario}"] = float(r.prob_median)
            summary[f"TRI_prob_{r.scenario}_std"] = float(r.prob_std)

        # Second and third most probable scenarios, so the catalog shows
        # what the top pick was competing against without a join back to
        # the scenarios CSV.
        ranked = agg[np.isfinite(agg["prob_median"])].reset_index(drop=True)
        for rank in (2, 3):
            if len(ranked) >= rank:
                summary[f"TRICERATOPS_scenario_{rank}"] = \
                    str(ranked.iloc[rank - 1]["scenario"])
                summary[f"TRICERATOPS_scenario_{rank}_prob"] = \
                    float(ranked.iloc[rank - 1]["prob_median"])
            else:
                summary[f"TRICERATOPS_scenario_{rank}"] = None
                summary[f"TRICERATOPS_scenario_{rank}_prob"] = np.nan

        f_summary = os.path.join(outdir, f"{tag}_TRICERATOPS_summary.csv")
        pd.DataFrame([summary]).to_csv(f_summary, index=False)
        files["summary_csv"] = f_summary

        f_json = os.path.join(outdir, f"{tag}_TRICERATOPS_summary.json")
        with open(f_json, "w") as fh:
            json.dump(
                {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                 for k, v in summary.items()},
                fh, indent=2, default=str)
        files["summary_json"] = f_json

        # ------------------------------------------------- bestfit update
        if update_bestfit:
            try:
                current = pd.read_csv(bestfit_path)
                for k, v in summary.items():
                    if k in ("TIC_ID", "Sector") and k in current.columns:
                        continue
                    current[k] = v
                tmp = bestfit_path + ".tmp"
                current.to_csv(tmp, index=False)
                os.replace(tmp, bestfit_path)
                files["bestfit_csv"] = bestfit_path
                if verbose:
                    print(f"\nbestfit updated: {bestfit_path}")
            except Exception as exc:
                if verbose:
                    print(f"[warn] bestfit update failed, standalone files "
                          f"are intact: {exc}")

        if verbose:
            print(f"\n{'=' * 62}")
            print(f"{tag}")
            print(f"  FPP   p50 {fpp_med:.5f}   "
                  f"[p16 {fpp_stats['FPP_p16']:.5f}, "
                  f"p84 {fpp_stats['FPP_p84']:.5f}]   "
                  f"p95 {fpp_stats['FPP_p95']:.5f}")
            print(f"        range {fpp_stats['FPP_min']:.5f} to "
                  f"{fpp_stats['FPP_max']:.5f}, "
                  f"{fpp_stats['FPP_frac_high'] * 100:.0f}% above 0.5   "
                  f"({int(ok.sum())} trials)")
            print(f"  NFPP  p50 {nfpp_med:.5f}   "
                  f"p84 {nfpp_stats['NFPP_p84']:.5f}   "
                  f"p95 {nfpp_stats['NFPP_p95']:.5f}")
            print(f"  exptime {exptime * 1440.0:.1f} min, "
                  f"N = {mc_samples:,}")
            print(f"  outcome     : {outcome}")
            print(f"  below zero  : {neg['n_below_zero']} point(s), lowest "
                  f"{neg['min_flux']:.4f}; poorly vetted flag "
                  f"{cls['poorly_vetted']}")
            print(f"  leading star: {scen['top_scenario_star']}  p = "
                  f"{scen['top_scenario_star_prob']:.4e}")
            print(f"  top scenario: {top_scenario}  p = {top_prob:.4e}")
            print(f"  disposition : {disposition}")
            print(f"  runtime     : {summary['TRICERATOPS_runtime_s']:.1f} s")
            print(f"{'=' * 62}")

        return {
            "tic_id": tic_id, "sector": sector, "skipped": None,
            "FPP_median": fpp_med, "FPP_mean": fpp_mean, "FPP_std": fpp_std,
            "NFPP_median": nfpp_med, "NFPP_mean": nfpp_mean,
            "NFPP_std": nfpp_std, "n_trials": int(ok.sum()),
            "disposition": disposition, "top_scenario": top_scenario,
            "top_scenario_prob": top_prob, "scenario_probs": scenario_probs,
            "trials_df": trials_df, "probs_df": agg,
            "fluxratio_target": fluxratio_target, "nemesis_fcr": nemesis_fcr,
            "fcr_residual": fcr_residual, "Rp_RE": rp_value,
            "files": files, "inputs": inp, "target": target,
            "outcome": outcome, "host_check": host,
            "classification": cls, "scenario_summary": scen,
            "negative_flux": neg, "star_params": star_params,
        }

    finally:
        if opened_here:
            hdulist.close()

def run_triceratops_on_nemesis_OLD(
    tic_id: int | None = None,
    sector: int | None = None,
    report_dir: str | None = None,
    savelcpath: str | None = None,
    bestfit_path: str | None = None,
    hdu: Any = None,
    LC_df: pd.DataFrame | None = None,
    lightkurve_cache_dir: str | None = None,
    outdir: str | None = None,
    trilegal_dir: str | None = None,
    n_trials: int = 20,
    flux_type: str = "Detrended",
    search: str = "TLS",
    n_durations: int = 5,
    n_bins: int | None = None,
    exptime: float | None = None,
    mc_samples: int = 5_000_000,
    parallel: bool = False,
    rp_max_earth: float | None = None,
    rp_column: str | None = None,
    nemesis_fcr: float | None = None,
    update_bestfit: bool = True,
    save_plots: bool = True,
    append_to_vetting_report: bool = True,
    vetting_report_path: str | None = None,
    report_layout: str = "auto",
    report_split: bool = False,
    report_base_pages: int | None = 6,
    seed: int | None = None,
    verbose: bool = True,
    posterior_path: str | None = None,
) -> dict[str, Any]:
    """
    Run TRICERATOPS statistical validation on one NEMESIS star-sector, from
    existing transit-search products through to catalog-ready columns.

    Three design choices worth knowing.

    **Median is primary.** ``calc_probs`` evaluates scenario likelihoods by
    Monte Carlo, so repeated runs on identical inputs differ. The median
    across trials is the headline value, matching the manuscript
    specification, with mean and trial-to-trial standard deviation recorded
    alongside.

    **All scenarios are kept.** The per-scenario ``prob`` values are
    posterior probabilities, not false-positive probabilities, so the most
    likely scenario is the one with the HIGHEST probability. Rather than
    reduce to a winner, the median probability of every scenario is pivoted
    into ``TRI_prob_<SCENARIO>`` columns, with the argmax recorded
    separately for convenience.

    **TRILEGAL is cached, never re-queried.** The background population
    depends on the galactic sightline, not the candidate. A file at
    ``{trilegal_dir}/{ID}_TRILEGAL.csv`` is reused when present; only when
    absent is the network touched, and then once. This is what makes the
    function usable on compute nodes with no outbound network: pre-stage the
    files and no trial hits the web.

    Parameters
    ----------
    tic_id, sector : int, optional
        Target identity. Required unless ``bestfit_path`` is given, in which
        case both are parsed from its filename.
    report_dir : str, optional
        Directory holding bestfit CSVs, normally
        ``settings['saveReportpath_TLS']``. Required unless ``bestfit_path``
        is given.
    savelcpath : str
        Directory holding the persisted masks and, if ``LC_df`` is None, the
        final light curve CSV. Required.
    bestfit_path : str, optional
        Explicit path to the bestfit CSV, bypassing the search.
    hdu : fits.HDUList or str, optional
        Open HDUList or path to the TESScut FITS. Required: the CCD offset
        keywords cannot be recovered from anything else.
    LC_df : pd.DataFrame, optional
        Light curve table. Loaded from ``savelcpath`` when None.
    lightkurve_cache_dir : str, optional
        Cache directory for the TESScut cutout TRICERATOPS downloads for
        ITSELF. This is a different file from ``hdu``: ``hdu`` supplies four
        header keywords for the aperture conversion and never reaches
        TRICERATOPS, while TRICERATOPS needs pixel data of its own for the
        PSF flux fractions in ``calc_depths`` and the image panel in
        ``plot_field``. Point this at ``settings['downloadpath']`` to reuse
        the NEMESIS cutout, which works only when the requested cutout sizes
        match and ``shard_cache_by_tic`` is False. Otherwise give it a
        dedicated directory. Omitting it means a fresh download on every
        target construction, which fails outright on compute nodes with no
        outbound network.
    outdir : str, optional
    trilegal_dir : str, optional
        Directory for cached TRILEGAL files. Defaults to ``outdir``.
    n_trials : int
        Independent ``calc_probs`` realizations. Manuscript macro ``NTRI``.
    flux_type : str
        ``'SAP'``, ``'PLD'`` or ``'Detrended'``.
    search : str
        ``'TLS'`` or ``'BLS'``.
    n_durations : int
        Half-width of the fold window, in transit durations.
    n_bins : int or None
        ``None`` means no binning, correct for 30-minute FFI cadence.
    exptime : float, optional
        Exposure time in DAYS. Derived from the median cadence spacing of
        the raw time array when omitted, which is the production setting.
    mc_samples : int
        ``calc_probs`` parameter ``N`` per trial. Default 5,000,000.
    parallel : bool
        Passed to ``calc_probs``. Keep False under disBatch: each call
        spawns a worker pool, and many tasks per node each spawning one
        reproduces the thread-oversubscription pattern that amplified the
        July 2026 heap-corruption crashes. Parallelize across candidates.
    rp_max_earth : float or None
        Radius ceiling in Earth radii. Candidates above it return early
        with ``skipped='radius_ceiling'``. Default None: no ceiling, every
        candidate is evaluated (09-29 decision). Candidates above
        :data:`RP_FLAG_RE` are flagged in ``TRICERATOPS_Rp_flag_gt_8``,
        because above roughly that radius giant planets, brown dwarfs and
        the lowest-mass stars overlap and depth alone no longer
        discriminates.
    rp_column : str, optional
        Bestfit radius column. Auto-detected from
        :data:`RP_COLUMN_CANDIDATES`.
    nemesis_fcr : float, optional
        NEMESIS flux contamination ratio for the independent cross-check.
        Auto-detected from the bestfit table when omitted.
    update_bestfit : bool
        Append ``TRICERATOPS_*`` and ``TRI_prob_*`` columns to the bestfit
        CSV, written to a temp file and renamed so an interrupt cannot
        truncate a primary survey product.
    save_plots : bool
        Write the field and scenario-fit figures.
    seed : int, optional
        Seeds ``numpy.random`` before the trial loop. TRICERATOPS also
        samples internally, so this constrains reproducibility only
        partially. Recorded for provenance either way.
    verbose : bool
        Print progress.
    posterior_path : str, optional
        jaxoplanet ``*_jaxoplanet_posterior.csv`` for this star-sector.
        When given, period, mid-transit time and duration come from its
        posterior medians and the radius from ``MCMC Planet Radius [RE]``.
        The depth stays the search depth from the bestfit. When omitted the
        search ephemeris is used, which keeps library calls such as the
        TOI-465 reference reproduction working.

    Returns
    -------
    dict
        ``tic_id``, ``sector``, ``skipped``, ``FPP_median``, ``FPP_mean``,
        ``FPP_std``, ``NFPP_median``, ``NFPP_mean``, ``NFPP_std``,
        ``n_trials``, ``disposition``, ``top_scenario``,
        ``top_scenario_prob``, ``scenario_probs``, ``trials_df``,
        ``probs_df``, ``fluxratio_target``, ``nemesis_fcr``,
        ``fcr_residual``, ``Rp_RE``, ``files``, ``inputs``, ``target``.

    Raises
    ------
    ValueError
        If required paths are missing, no bestfit is found, or every trial
        fails.
    FileNotFoundError
        If the bestfit CSV, light curve or aperture mask is absent.

    Examples
    --------
    >>> res = run_triceratops_on_nemesis(
    ...     tic_id=270380593, sector=4,
    ...     report_dir=settings['saveReportpath_TLS'],
    ...     savelcpath=settings['savelcpath'],
    ...     hdu=hdu, n_trials=20)
    >>> print(res["FPP_median"], res["disposition"])
    """
    import triceratops.triceratops as tr
#     from triceratops.funcs import save_trilegal

    t_start = time.time()
    mc_samples = int(mc_samples)

    if savelcpath is None:
        raise ValueError("savelcpath is required")
    if hdu is None:
        raise ValueError(
            "hdu is required: the cutout-to-CCD pixel offset lives in the "
            "TESScut header and cannot be recovered from anything else. Pass "
            "the HDUList from full_pipeline, or a path to the cached FITS."
        )

    if bestfit_path is None:
        if report_dir is None or tic_id is None:
            raise ValueError(
                "pass bestfit_path, or both report_dir and tic_id")
        hits = find_nemesis_bestfit(report_dir, tic_id, sector=sector,
                                    search=search, verbose=verbose)
        if not hits:
            raise ValueError(
                f"no {search} bestfit found for TIC {tic_id} "
                f"(sector={sector}) under {report_dir}")
        if len(hits) > 1 and sector is None:
            raise ValueError(
                f"{len(hits)} bestfit files match TIC {tic_id}; pass sector= "
                f"or bestfit_path= to disambiguate:\n  " + "\n  ".join(hits))
        bestfit_path = hits[0]

    bestfit_path = os.path.abspath(bestfit_path)
    if not os.path.isfile(bestfit_path):
        raise FileNotFoundError(f"bestfit CSV not found: {bestfit_path}")
    bestfit_df = pd.read_csv(bestfit_path)

    base = os.path.basename(bestfit_path)
    m_tic = re.search(r"TIC[_ ]?(\d+)", base)
    m_sec = re.search(r"Sector[_ ]?(\d+)", base)
    if m_tic:
        tic_id = int(m_tic.group(1))
    if m_sec:
        sector = int(m_sec.group(1))

    if LC_df is None:
        LC_df = find_nemesis_lightcurve(savelcpath, tic_id, sector,
                                        verbose=verbose)

    hdulist, opened_here = open_hdu(hdu)

    try:
        if outdir is None:
            outdir = os.path.join(os.path.dirname(bestfit_path), "TRICERATOPS")
        os.makedirs(outdir, exist_ok=True)
        if trilegal_dir is None:
            trilegal_dir = outdir
        os.makedirs(trilegal_dir, exist_ok=True)

        inp = nemesis_to_triceratops_inputs(
            hdu=hdulist, LC_df=LC_df, bestfit_df=bestfit_df,
            savelcpath=savelcpath, bestfit_path=bestfit_path,
            flux_type=flux_type, search=search,
            n_durations=n_durations, n_bins=n_bins, verbose=verbose,
            posterior_path=posterior_path,
        )

        tic_id, sector = inp["tic_id"], inp["sector"]
        tag = f"TIC_{tic_id}_Sector_{sector:04d}"

        # TRICERATOPS defaults exptime to 0.00139 d, which is 2 minutes: the
        # SPOC short-cadence value. At 30-minute FFI cadence a transit is
        # smeared across a full cadence, so comparing it against sharp-edged
        # 2-minute models breaks the shape discriminator that separates a
        # planet from an eclipsing binary, and FPP becomes multi-modal.
        # Derived from the raw, unfolded time array rather than assumed.
        if exptime is None:
            t_raw = np.asarray(LC_df["Time"], dtype=float)
            t_raw = t_raw[np.isfinite(t_raw)]
            dt = np.median(np.diff(np.sort(t_raw))) if t_raw.size > 2 else np.nan
            exptime = float(dt) if np.isfinite(dt) and dt > 0 else 30.0 / 1440.0
        if verbose:
            print(f"exptime       : {exptime:.6f} d "
                  f"({exptime * 1440.0:.2f} min)")
            print(f"MC samples    : {mc_samples:,}")

        # ------------------------------------------------------------ radius
        # The jaxoplanet radius is preferred when a posterior was passed;
        # the bestfit search radius is used otherwise. An explicit
        # rp_column still wins. rp_column records which source was used.
        rp_value = np.nan
        post_rp = float(inp.get("Rp_posterior_RE", np.nan))
        if rp_column is None and np.isfinite(post_rp):
            rp_column = "MCMC Planet Radius [RE]"
            rp_value = post_rp
        else:
            if rp_column is None:
                for c in RP_COLUMN_CANDIDATES:
                    if c in bestfit_df.columns:
                        rp_column = c
                        break
            if rp_column is not None and rp_column in bestfit_df.columns:
                try:
                    rp_value = float(bestfit_df.iloc[0][rp_column])
                except (TypeError, ValueError):
                    rp_value = np.nan
                    #
        if rp_max_earth is not None:
            if not np.isfinite(rp_value):
                if verbose:
                    print(f"[warn] no usable radius column "
                          f"(tried {RP_COLUMN_CANDIDATES}); ceiling NOT "
                          f"applied")
            elif rp_value > rp_max_earth:
                if verbose:
                    print(f"SKIP {tag}: Rp = {rp_value:.2f} Re exceeds the "
                          f"{rp_max_earth:.1f} Re ceiling.")
                return {
                    "tic_id": tic_id, "sector": sector,
                    "skipped": "radius_ceiling", "Rp_RE": rp_value,
                    "FPP_median": np.nan, "FPP_mean": np.nan,
                    "FPP_std": np.nan, "NFPP_median": np.nan,
                    "NFPP_mean": np.nan, "NFPP_std": np.nan, "n_trials": 0,
                    "disposition": "not evaluated (Rp ceiling)",
                    "top_scenario": None, "top_scenario_prob": np.nan,
                    "scenario_probs": {}, "trials_df": None, "probs_df": None,
                    "fluxratio_target": np.nan, "nemesis_fcr": np.nan,
                    "fcr_residual": np.nan, "files": {}, "inputs": inp,
                    "target": None,
                }

        # ---------------------------------------------------------- TRILEGAL
        # The canonical location, plus the places an earlier interactive
        # save_trilegal call may have dropped the file. That helper takes no
        # path argument and writes {ID}_TRILEGAL.csv to the working
        # directory, so a file fetched by hand rarely sits where the default
        # cache lookup expects it.
        trilegal_name = f"{tic_id}_TRILEGAL.csv"
        trilegal_path = os.path.join(trilegal_dir, trilegal_name)

        search_dirs = [trilegal_dir, os.getcwd(), outdir,
                       os.path.dirname(bestfit_path)]
        found = None
        for cand_dir in search_dirs:
            cand = os.path.join(cand_dir, trilegal_name)
            if os.path.isfile(cand) and os.path.getsize(cand) > 0:
                found = os.path.abspath(cand)
                break

        # Copy rather than move. os.replace would relocate the canonical
        # cached file into whatever outdir this run happens to use, so a
        # throwaway run (a convergence test, a parameter sweep) would steal
        # the cache and force the next production run back onto the network.
        if found is not None and os.path.abspath(trilegal_path) != found:
            shutil.copy2(found, trilegal_path)
            if verbose:
                print(f"TRILEGAL      : copied {found} -> {trilegal_path}")
            found = trilegal_path

        cached = found is not None

        # Built once and shared by both construction paths. Omitted rather
        # than passed as None so triceratops falls back to its own default
        # when no cache directory is given.
        target_kwargs: dict[str, Any] = {}
        if lightkurve_cache_dir is not None:
            target_kwargs["lightkurve_cache_dir"] = lightkurve_cache_dir
            if verbose:
                print(f"lk cache dir  : {lightkurve_cache_dir}")

        if cached:
            if verbose:
                print(f"TRILEGAL      : cached, {trilegal_path}")
            target = tr.target(ID=tic_id, sectors=np.array([sector]),
                               trilegal_fname=trilegal_path, **target_kwargs)
            target.trilegal_fname = trilegal_path
        else:
            if verbose:
                print(f"TRILEGAL      : not cached in "
                      f"{[os.path.abspath(p) for p in search_dirs]}, "
                      f"submitting one job")
            with trilegal_ssl_workaround(verbose=verbose):
                target = tr.target(ID=tic_id, sectors=np.array([sector]),
                                   **target_kwargs)
            # Outside the workaround block above, because
            # save_trilegal_when_ready opens its own. The job needs time to
            # run; reading the URL immediately returns an empty document.
            written = save_trilegal_when_ready(
                getattr(target, "trilegal_url", None), tic_id,
                verbose=verbose)
            if written and os.path.isfile(str(written)):
                if os.path.abspath(str(written)) != os.path.abspath(trilegal_path):
                    os.replace(str(written), trilegal_path)
                target.trilegal_fname = trilegal_path
                if verbose:
                    print(f"TRILEGAL      : saved to {trilegal_path}")
            else:
                trilegal_path = str(written)
                if verbose:
                    print(f"[warn] save_trilegal returned {written!r}; later "
                          f"trials may re-query the network")

        if verbose:
            print(f"stars in field: {len(target.stars)}")

        # ------------------------------------------------------------ depths
        target.calc_depths(tdepth=inp["tdepth"],
                           all_ap_pixels=inp["all_ap_pixels"])

        stars = target.stars.copy()
        id_col = "ID" if "ID" in stars.columns else stars.columns[0]
        fluxratio_target = np.nan
        if "fluxratio" in stars.columns:
            match = stars[stars[id_col].astype(str) == str(tic_id)]
            if len(match):
                fluxratio_target = float(match["fluxratio"].iloc[0])

        if nemesis_fcr is None:
            for c in FCR_COLUMN_CANDIDATES:
                if c in bestfit_df.columns:
                    try:
                        nemesis_fcr = float(bestfit_df.iloc[0][c])
                    except (TypeError, ValueError):
                        nemesis_fcr = None
                    break

        fcr_residual = np.nan
        if (nemesis_fcr is not None and np.isfinite(nemesis_fcr)
                and np.isfinite(fluxratio_target)):
            fcr_residual = fluxratio_target - (1.0 - nemesis_fcr)

        if verbose:
            print(f"\nfluxratio(target)     : {fluxratio_target:.6f}")
            if np.isfinite(fcr_residual):
                print(f"1 - NEMESIS FCR       : {1.0 - nemesis_fcr:.6f}")
                print(f"residual              : {fcr_residual:+.6f}  "
                      f"(independent contamination cross-check)")
            else:
                print("NEMESIS FCR           : unavailable, cross-check "
                      "skipped")
            if "fluxratio" in stars.columns:
                contam = stars[stars["fluxratio"] > 0.001]
                cols = [c for c in (id_col, "fluxratio", "tdepth")
                        if c in contam.columns]
                print("\ncontributing sources:")
                print(contam[cols].to_string(index=False))

#         # -------------------------------------------------------- field plot
#         files: dict[str, str] = {}
#         if save_plots:
#             try:
#                 f_field = os.path.join(outdir, f"{tag}_TRICERATOPS_field.png")
#                 target.plot_field(sector=sector, ap_pixels=inp["ap_pixels"],
#                                   ap_color="red", save=False, fname=None)
#                 plt.savefig(f_field, dpi=150, bbox_inches="tight")
#                 plt.close("all")
#                 files["field_plot"] = f_field
#             except Exception as exc:
#                 if verbose:
#                     print(f"[warn] field plot failed: {exc}")
        # -------------------------------------------------------- field plot
        files: dict[str, str] = {}
        if save_plots:
            f_field = os.path.join(outdir, f"{tag}_TRICERATOPS_field.png")
            written_fig = render_to_file(
                lambda: target.plot_field(
                    sector=sector, ap_pixels=inp["ap_pixels"],
                    ap_color="red", save=False, fname=None),
                f_field, verbose=verbose)
            if written_fig:
                files["field_plot"] = written_fig

        # ------------------------------------------------------------ trials
        if seed is not None:
            np.random.seed(seed)

        rows: list[dict[str, float]] = []
        scen_frames: list[pd.DataFrame] = []

        for i in range(n_trials):
            try:
                target.calc_probs(
                    time=inp["time"], flux_0=inp["flux"],
                    flux_err_0=inp["flux_err"], P_orb=inp["P_orb"],
                    exptime=exptime, N=mc_samples,
                    parallel=parallel, verbose=0,
                )
            except Exception as exc:
                if verbose:
                    print(f"  trial {i + 1}/{n_trials}: FAILED ({exc})")
                rows.append({"trial": i, "FPP": np.nan, "NFPP": np.nan})
                continue

            rows.append({"trial": i, "FPP": float(target.FPP),
                         "NFPP": float(target.NFPP)})
            p = target.probs.copy()
            p["trial"] = i
            scen_frames.append(p)
            if verbose:
                print(f"  trial {i + 1}/{n_trials}: FPP {target.FPP:.5f}  "
                      f"NFPP {target.NFPP:.5f}")

        trials_df = pd.DataFrame(rows)
        ok = trials_df["FPP"].notna()
        if not ok.any():
            raise ValueError(
                f"all {n_trials} calc_probs trials failed for {tag}")

        fpp = trials_df.loc[ok, "FPP"].to_numpy(dtype=float)
        nfpp = trials_df.loc[ok, "NFPP"].to_numpy(dtype=float)

        fpp_stats = summarize_trials(fpp, "FPP")
        nfpp_stats = summarize_trials(nfpp, "NFPP")

        fpp_med = fpp_stats["FPP_p50"]
        fpp_mean = fpp_stats["FPP_mean"]
        fpp_std = fpp_stats["FPP_std"]
        nfpp_med = nfpp_stats["NFPP_p50"]
        nfpp_mean = nfpp_stats["NFPP_mean"]
        nfpp_std = nfpp_stats["NFPP_std"]

        # --------------------------------------------------------- scenarios
        probs_all = pd.concat(scen_frames, ignore_index=True)
        agg = (probs_all.groupby("scenario")["prob"]
               .agg(prob_median="median", prob_mean="mean", prob_std="std",
                    n_trials="count")
               .reset_index()
               .sort_values("prob_median", ascending=False)
               .reset_index(drop=True))
        agg["prob_std"] = agg["prob_std"].fillna(0.0)

        finite = agg[np.isfinite(agg["prob_median"])]
        top_scenario = str(finite.iloc[0]["scenario"]) if len(finite) else None
        top_prob = float(finite.iloc[0]["prob_median"]) if len(finite) else np.nan
        scenario_probs = {str(r.scenario): float(r.prob_median)
                          for r in agg.itertuples()
                          if np.isfinite(r.prob_median)}

        # Disposition is driven by the 84th percentile, not the median. With
        # a skewed or multi-modal trial distribution a median sitting just
        # under 0.015 flips on a single draw, so "validated" requires the
        # upper end of the spread to clear the threshold as well. Runs where
        # a meaningful fraction of draws land above 0.5 are reported as
        # unconverged rather than given a disposition at all: that pattern
        # means the transit shape did not constrain the scenario, and no
        # single number summarizes it honestly.
        # Tiers follow Giacalone et al. (2021): validated planet requires
        # FPP < 0.015 and NFPP < 1e-3; likely planet FPP < 0.5 and
        # NFPP < 1e-3; likely nearby false positive NFPP > 0.1. Two
        # departures from a naive implementation.
        #
        # First, the decision uses the 84th percentile of the trial
        # distribution rather than a single realization, because that
        # distribution is right-skewed and a point estimate near a
        # threshold flips between runs.
        #
        # Second, the validated-planet tier is withheld above
        # VALIDATION_EXPTIME_MAX_D. Section 4.2 of that paper repeated the
        # whole analysis on 30-minute FFI light curves and found no region
        # of the FPP-NFPP plane in which planets could be validated, while
        # the likely-planet and nearby-false-positive regions both held up.
        # Long-cadence results are for vetting and follow-up prioritization,
        # not validation.
        fpp_p84 = fpp_stats["FPP_p84"]
        nfpp_p84 = nfpp_stats["NFPP_p84"]
        fpp_frac_high = fpp_stats["FPP_frac_high"]
        long_cadence = float(exptime) > VALIDATION_EXPTIME_MAX_D

        if not np.isfinite(fpp_med):
            disposition = "inconclusive (NaN FPP)"
        elif nfpp_med > 0.1:
            disposition = "likely nearby false positive"
        elif fpp_frac_high >= 0.25:
            disposition = (f"unconverged ({fpp_frac_high * 100:.0f}% of "
                           f"trials above FPP 0.5)")
        elif fpp_p84 < 0.5 and nfpp_p84 < 1e-3:
            if long_cadence:
                disposition = ("likely planet (validation tier withheld at "
                               f"{exptime * 1440.0:.0f} min cadence)")
            elif fpp_p84 < 0.015:
                disposition = "validated planet"
            else:
                disposition = "likely planet"
        else:
            disposition = "uncertain"

#         # --------------------------------------------------------- fits plot
#         if save_plots:
#             try:
#                 f_fits = os.path.join(outdir, f"{tag}_TRICERATOPS_fits.png")
#                 target.plot_fits(time=inp["time"], flux_0=inp["flux"],
#                                  flux_err_0=inp["flux_err"])
#                 plt.savefig(f_fits, dpi=150, bbox_inches="tight")
#                 plt.close("all")
#                 files["fits_plot"] = f_fits
#             except Exception as exc:
#                 if verbose:
#                     print(f"[warn] fits plot failed: {exc}")
        # --------------------------------------------------------- fits plot
        if save_plots:
            f_fits = os.path.join(outdir, f"{tag}_TRICERATOPS_fits.png")
            written_fig = render_to_file(
                lambda: target.plot_fits(
                    time=inp["time"], flux_0=inp["flux"],
                    flux_err_0=inp["flux_err"]),
                f_fits, verbose=verbose)
            if written_fig:
                files["fits_plot"] = written_fig
        #
        # ------------------------------------------- vetting report page
        # Appended in place so the NEMESIS filename convention and location
        # are unchanged. Idempotent via keep_pages=6: a repeat run truncates
        # back to the six original pages rather than growing the PDF.
        #
        # Deliberately not routed through NEMESIS append_PDF: that helper
        # rebuilds the PDF from a list of page PNGs, writes to a filename it
        # constructs itself, and deletes every input but the first. It also
        # cannot run incrementally, since the page2-page6 PNGs are already
        # deleted by the time the report exists.
        if (append_to_vetting_report and save_plots
                and files.get("field_plot") and files.get("fits_plot")):
            if vetting_report_path is None:
                vetting_report_path = os.path.join(
                    os.path.dirname(bestfit_path),
                    f"{tag}_{search}_Report_multi_page.pdf")

            if not os.path.isfile(vetting_report_path):
                if verbose:
                    print(f"[warn] no vetting report at "
                          f"{vetting_report_path}; skipping PDF append")
            else:
                page_size = pdf_page_size_px(vetting_report_path,
                                             dpi=150, verbose=verbose)
                if page_size is None:
                    if verbose:
                        print("[warn] could not read report page size; "
                              "skipping PDF append")
                else:
                    params = [
                        ("Period [d]", f"{inp['P_orb']:.6f}"),
                        ("T0 [BTJD]", f"{inp['T0']:.5f}"),
                        ("Duration [hr]", f"{inp['duration_days'] * 24:.3f}"),
                        ("Depth [ppt]", f"{inp['tdepth'] * 1e3:.3f}"),
                        ("Rp [RE]", f"{rp_value:.3f}"
                                    if np.isfinite(rp_value) else "n/a"),
                        ("Aperture [px]", f"{len(inp['ap_pixels'])}"),
                        ("Flux ratio", f"{fluxratio_target:.5f}"),
                        ("LC points", f"{inp['time'].size}"),
                    ]
                    stats_rows = [
                        ("FPP p50", f"{fpp_med:.5f}"),
                        ("FPP p16", f"{fpp_stats['FPP_p16']:.5f}"),
                        ("FPP p84", f"{fpp_stats['FPP_p84']:.5f}"),
                        ("FPP max", f"{fpp_stats['FPP_max']:.5f}"),
                        ("NFPP p50", f"{nfpp_med:.5f}"),
                        ("NFPP max", f"{nfpp_stats['NFPP_max']:.5f}"),
                        ("Trials", f"{int(ok.sum())}"),
                        ("MC samples", f"{mc_samples:,}"),
                    ]
                    top_scen = [
                        (str(r.scenario), float(r.prob_median),
                         float(r.prob_std))
                        for r in agg.itertuples()
                        if np.isfinite(r.prob_median)
                        and float(r.prob_median) > 1e-4
                    ][:6]

                    page = compose_triceratops_report_page(
                        files["field_plot"], files["fits_plot"],
                        os.path.join(outdir, f"{tag}_TRICERATOPS_page.png"),
                        page_size_px=page_size,
                        subtitle=(f"TIC {tic_id}   |   Sector {sector}   |   "
                                  f"{inp['ephemeris_source']}   |   exptime "
                                  f"{exptime * 1440:.0f} min   |   "
                                  f"module v{__version__}"),
                        params=params,
                        stats=stats_rows,
                        scenarios=top_scen,
                        disposition=disposition.split(" (")[0],
                        footnote=("Per-scenario values are independent "
                                  "medians over trials and do not sum to "
                                  "unity.\nFPP is computed per trial, then "
                                  "summarized.\nClassification criteria from "
                                  "Giacalone et al. (2021);\nthe "
                                  "validated-planet tier is unavailable at "
                                  "FFI cadence (their Section 4.2)."),
                        verbose=verbose)

                    if page:
                        files["report_page_png"] = page
                        appended = append_png_to_pdf(
                            vetting_report_path, page,
                            in_place=True,
                            keep_pages=report_base_pages,
                            verbose=verbose)
                        if appended:
                            files["vetting_report_pdf"] = appended
        #
        # ------------------------------------------------------------ tables
        f_trials = os.path.join(outdir, f"{tag}_TRICERATOPS_trials.csv")
        trials_df.to_csv(f_trials, index=False)
        files["trials_csv"] = f_trials

        f_probs = os.path.join(outdir, f"{tag}_TRICERATOPS_scenarios.csv")
        agg.to_csv(f_probs, index=False)
        files["scenarios_csv"] = f_probs

        f_all = os.path.join(
            outdir, f"{tag}_TRICERATOPS_scenarios_per_trial.csv")
        probs_all.to_csv(f_all, index=False)
        files["scenarios_per_trial_csv"] = f_all

        summary: dict[str, Any] = {
            "TIC_ID": tic_id, "Sector": sector,
            **{f"TRICERATOPS_{k}": v for k, v in fpp_stats.items()},
            **{f"TRICERATOPS_{k}": v for k, v in nfpp_stats.items()},
            "TRICERATOPS_exptime_d": float(exptime),
            "TRICERATOPS_mc_samples": int(mc_samples),
            "TRICERATOPS_long_cadence": bool(long_cadence),
            "TRICERATOPS_validation_tier_available": bool(not long_cadence),
            "TRICERATOPS_median_FPP": fpp_med,
            "TRICERATOPS_mean_FPP": fpp_mean,
            "TRICERATOPS_std_FPP": fpp_std,
            "TRICERATOPS_median_NFPP": nfpp_med,
            "TRICERATOPS_mean_NFPP": nfpp_mean,
            "TRICERATOPS_std_NFPP": nfpp_std,
            "TRICERATOPS_N_trials": int(ok.sum()),
            "TRICERATOPS_N_trials_failed": int((~ok).sum()),
            "TRICERATOPS_disposition": disposition,
            "TRICERATOPS_top_scenario": top_scenario,
            "TRICERATOPS_top_scenario_prob": top_prob,
            "TRICERATOPS_aperture_npix": int(len(inp["ap_pixels"])),
            "TRICERATOPS_LC_points": int(inp["time"].size),
            "TRICERATOPS_flux_err": float(inp["flux_err"]),
            "TRICERATOPS_tdepth_in": float(inp["tdepth"]),
            "TRICERATOPS_fluxratio_target": fluxratio_target,
            "TRICERATOPS_NEMESIS_FCR": (float(nemesis_fcr)
                                        if nemesis_fcr is not None else np.nan),
            "TRICERATOPS_FCR_residual": fcr_residual,
            "TRICERATOPS_Rp_RE": rp_value,
            "TRICERATOPS_Rp_column": rp_column or "",
            "TRICERATOPS_Rp_flag_gt_8": bool(np.isfinite(rp_value)
                                             and rp_value > RP_FLAG_RE),
            "TRICERATOPS_ephemeris_source": inp["ephemeris_source"],
            "TRICERATOPS_posterior_file": os.path.basename(
                inp["posterior_path"] or ""),
            "TRICERATOPS_posterior_converged": inp["posterior_converged"],
            "TRICERATOPS_P_orb_d": float(inp["P_orb"]),
            "TRICERATOPS_T0_BTJD": float(inp["T0"]),
            "TRICERATOPS_duration_hr": float(inp["duration_days"]) * 24.0,
            "TRICERATOPS_trilegal_file": os.path.basename(str(trilegal_path)),
            "TRICERATOPS_lk_cache_dir": (lightkurve_cache_dir or ""),
            "TRICERATOPS_seed": (seed if seed is not None else -1),
            "TRICERATOPS_module_version": __version__,
            "TRICERATOPS_runtime_s": round(time.time() - t_start, 1),
            "TRICERATOPS_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        # Asymmetric uncertainties from the trial quantiles. The trial
        # distribution is right-skewed, so a single +/- value would
        # misrepresent it; these are the p16 and p84 offsets from the median
        # and are what should be quoted in a table.
        summary["TRICERATOPS_FPP_err_lo"] = fpp_med - fpp_stats["FPP_p16"]
        summary["TRICERATOPS_FPP_err_hi"] = fpp_stats["FPP_p84"] - fpp_med
        summary["TRICERATOPS_NFPP_err_lo"] = nfpp_med - nfpp_stats["NFPP_p16"]
        summary["TRICERATOPS_NFPP_err_hi"] = nfpp_stats["NFPP_p84"] - nfpp_med

        # Per-scenario median probability plus its trial-to-trial scatter.
        # The scatter matters: a scenario whose probability swings between
        # trials is a scenario the data does not constrain, which is the
        # case for STP at FFI cadence.
        for r in agg.itertuples():
            if not np.isfinite(r.prob_median):
                continue
            summary[f"TRI_prob_{r.scenario}"] = float(r.prob_median)
            summary[f"TRI_prob_{r.scenario}_std"] = float(r.prob_std)

        # Second and third most probable scenarios, so the catalog shows
        # what the top pick was competing against without a join back to
        # the scenarios CSV.
        ranked = agg[np.isfinite(agg["prob_median"])].reset_index(drop=True)
        for rank in (2, 3):
            if len(ranked) >= rank:
                summary[f"TRICERATOPS_scenario_{rank}"] = \
                    str(ranked.iloc[rank - 1]["scenario"])
                summary[f"TRICERATOPS_scenario_{rank}_prob"] = \
                    float(ranked.iloc[rank - 1]["prob_median"])
            else:
                summary[f"TRICERATOPS_scenario_{rank}"] = None
                summary[f"TRICERATOPS_scenario_{rank}_prob"] = np.nan

        f_summary = os.path.join(outdir, f"{tag}_TRICERATOPS_summary.csv")
        pd.DataFrame([summary]).to_csv(f_summary, index=False)
        files["summary_csv"] = f_summary

        f_json = os.path.join(outdir, f"{tag}_TRICERATOPS_summary.json")
        with open(f_json, "w") as fh:
            json.dump(
                {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                 for k, v in summary.items()},
                fh, indent=2, default=str)
        files["summary_json"] = f_json

        # ------------------------------------------------- bestfit update
        if update_bestfit:
            try:
                current = pd.read_csv(bestfit_path)
                for k, v in summary.items():
                    if k in ("TIC_ID", "Sector") and k in current.columns:
                        continue
                    current[k] = v
                tmp = bestfit_path + ".tmp"
                current.to_csv(tmp, index=False)
                os.replace(tmp, bestfit_path)
                files["bestfit_csv"] = bestfit_path
                if verbose:
                    print(f"\nbestfit updated: {bestfit_path}")
            except Exception as exc:
                if verbose:
                    print(f"[warn] bestfit update failed, standalone files "
                          f"are intact: {exc}")

        if verbose:
            print(f"\n{'=' * 62}")
            print(f"{tag}")
            print(f"  FPP   p50 {fpp_med:.5f}   "
                  f"[p16 {fpp_stats['FPP_p16']:.5f}, "
                  f"p84 {fpp_stats['FPP_p84']:.5f}]   "
                  f"p95 {fpp_stats['FPP_p95']:.5f}")
            print(f"        range {fpp_stats['FPP_min']:.5f} to "
                  f"{fpp_stats['FPP_max']:.5f}, "
                  f"{fpp_stats['FPP_frac_high'] * 100:.0f}% above 0.5   "
                  f"({int(ok.sum())} trials)")
            print(f"  NFPP  p50 {nfpp_med:.5f}   "
                  f"p84 {nfpp_stats['NFPP_p84']:.5f}   "
                  f"p95 {nfpp_stats['NFPP_p95']:.5f}")
            print(f"  exptime {exptime * 1440.0:.1f} min, "
                  f"N = {mc_samples:,}")
            print(f"  top scenario: {top_scenario}  p = {top_prob:.4e}")
            print(f"  disposition : {disposition}")
            print(f"  runtime     : {summary['TRICERATOPS_runtime_s']:.1f} s")
            print(f"{'=' * 62}")

        return {
            "tic_id": tic_id, "sector": sector, "skipped": None,
            "FPP_median": fpp_med, "FPP_mean": fpp_mean, "FPP_std": fpp_std,
            "NFPP_median": nfpp_med, "NFPP_mean": nfpp_mean,
            "NFPP_std": nfpp_std, "n_trials": int(ok.sum()),
            "disposition": disposition, "top_scenario": top_scenario,
            "top_scenario_prob": top_prob, "scenario_probs": scenario_probs,
            "trials_df": trials_df, "probs_df": agg,
            "fluxratio_target": fluxratio_target, "nemesis_fcr": nemesis_fcr,
            "fcr_residual": fcr_residual, "Rp_RE": rp_value,
            "files": files, "inputs": inp, "target": target,
        }

    finally:
        if opened_here:
            hdulist.close()


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def _float_or_none(text: str) -> float | None:
    """
    Parse a command-line value as a float, or None for a null word.

    Parameters
    ----------
    text : str
        Raw argument text. ``''``, ``'none'``, ``'null'`` and ``'off'``
        (any case) map to None.

    Returns
    -------
    float or None

    Raises
    ------
    argparse.ArgumentTypeError
        If the text is neither a number nor a null word.

    Examples
    --------
    >>> _float_or_none("8"), _float_or_none("none")
    (8.0, None)
    """
    t = str(text).strip().lower()
    if t in ("", "none", "null", "off"):
        return None
    try:
        return float(t)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected a number or 'none', got {text!r}") from exc

_PAGE7_KEYS: tuple[str, ...] = (
    "TRICERATOPS_P_orb_d", "TRICERATOPS_T0_BTJD", "TRICERATOPS_duration_hr",
    "TRICERATOPS_tdepth_in", "TRICERATOPS_Rp_RE", "TRICERATOPS_aperture_npix",
    "TRICERATOPS_fluxratio_target", "TRICERATOPS_LC_points",
    "TRICERATOPS_FPP_p50", "TRICERATOPS_FPP_p16", "TRICERATOPS_FPP_p84",
    "TRICERATOPS_FPP_max", "TRICERATOPS_NFPP_p50", "TRICERATOPS_NFPP_max",
    "TRICERATOPS_N_trials", "TRICERATOPS_mc_samples", "TRICERATOPS_exptime_d",
    "TRICERATOPS_disposition", "TRICERATOPS_ephemeris_source",
    "TRICERATOPS_module_version",
)


def _pdf_n_pages(path: str) -> int:
    """
    Page count of a PDF; -1 when it cannot be read.

    Examples
    --------
    >>> _pdf_n_pages("/no/such/file.pdf")
    -1
    """
    try:
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            from pypdf import PdfReader
        return len(PdfReader(path).pages)
    except Exception:
        return -1


def merged_top_scenarios(agg: pd.DataFrame, n: int = 6,
                         floor: float = 1e-4) -> list[tuple[str, float, float]]:
    """
    Top scenarios of a merged TRICERATOPS result, as report page 7 lists them.

    Same rule as the single-run page in run_triceratops_on_nemesis: finite
    median probability above ``floor``, highest first, at most ``n``.

    Parameters
    ----------
    agg : pd.DataFrame
        <tag>_TRICERATOPS_scenarios.csv with scenario, prob_median, prob_std.
    n : int
        Most scenarios returned.
    floor : float
        Smallest median probability kept.

    Returns
    -------
    list of (str, float, float)
        (scenario, median probability, scatter over trials).

    Raises
    ------
    KeyError
        A needed column is absent.

    Examples
    --------
    >>> a = pd.DataFrame({"scenario": ["TP", "EB", "NTP"],
    ...                   "prob_median": [0.2, 0.7, 1e-6],
    ...                   "prob_std": [0.1, 0.05, 0.0]})
    >>> merged_top_scenarios(a)
    [('EB', 0.7, 0.05), ('TP', 0.2, 0.1)]
    """
    need = ("scenario", "prob_median", "prob_std")
    miss = [c for c in need if c not in agg.columns]
    if miss:
        raise KeyError(f"scenario table lacks {miss}; columns are {list(agg.columns)}")
    df = agg.assign(pmed=pd.to_numeric(agg["prob_median"], errors="coerce"),
                    pstd=pd.to_numeric(agg["prob_std"], errors="coerce"))
    df = df[np.isfinite(df["pmed"]) & (df["pmed"] > float(floor))]
    df = df.sort_values("pmed", ascending=False, kind="mergesort").head(int(n))
    return [(str(r.scenario), float(r.pmed), float(r.pstd)) for r in df.itertuples()]


def write_merged_report_page(
    tic: int,
    sector: int,
    merged_root: str = "/mnt/home/dfeliz/ceph/NEMESIS_II/triceratops/production_v130",
    data_root: str = "/mnt/home/dfeliz/ceph/NEMESIS_II",
    append_to: str | None = None,
    backup_dir: str | None = None,
    fits_png: str | None = None,     
    keep_pages: int = 6,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Build vetting-report page 7 (TRICERATOPS) from a merged split-and-merge
    result, and optionally append it to a PDF. Nothing is rerun.

    Production passed --no-report-append to every piece and deferred the
    write-back, so no vetting report received this page. This builds the page
    the single-run path builds (run_triceratops_on_nemesis ->
    compose_triceratops_report_page: same rows, rounding and footnote, plus
    one line on the scenario-fit panels) from the merged 20-trial files in
    <merged_root>/merged/<tag>/:
        <tag>_TRICERATOPS_summary.json     inputs and FPP/NFPP statistics
        <tag>_TRICERATOPS_scenarios.csv    per-scenario medians over trials
        <tag>_TRICERATOPS_field.png        piece 0's field plot
        <tag>_TRICERATOPS_fits.png         piece 0's scenario fits (one trial)
    and writes <tag>_TRICERATOPS_page.png beside them, sized to the
    star-sector's vetting report (pdf_page_size_px).

    With ``append_to`` the page is appended to that PDF in place, keeping its
    first ``keep_pages`` pages (append_png_to_pdf), so a rerun replaces page 7
    rather than adding page 8. With ``backup_dir`` the PDF is first copied to
    <backup_dir>/Sector_SSSS/ once; an existing backup is never overwritten.
    Missing inputs return a status instead of raising, for batch use.

    Parameters
    ----------
    tic, sector : int
        Star-sector.
    merged_root : str
        Production root holding merged/<tag>/.
    data_root : str
        NEMESIS II data root holding Sector_SSSS/NEMESIS_FFI_TLS_Report/.
    append_to : str or None
        PDF to append the page to, in place; None builds the page only.
    backup_dir : str or None
        Folder for a one-time copy of ``append_to`` before it is changed.
    fits_png : str or None
        Scenario-fit image used instead of the merged one, e.g. a labeled
        stand-in when TRICERATOPS computed no fits; None uses the merged file.    
    keep_pages : int
        Pages of ``append_to`` kept before the new page (6 for a survey report).
    verbose : bool
        Pass through to the page and append functions.

    Returns
    -------
    dict
        TIC_ID, Sector, status ("ok", "page only", "missing: ..." or
        "failed: ..."), page_png, report_pdf, appended_to, backup,
        pages_before, pages_after, missing_keys.

    Examples
    --------
    >>> res = write_merged_report_page(103633434, 20)  # doctest: +SKIP
    """
    tag = f"TIC_{int(tic)}_Sector_{int(sector):04d}"
    mdir = os.path.join(merged_root, "merged", tag)
    paths = {
        "summary": os.path.join(mdir, f"{tag}_TRICERATOPS_summary.json"),
        "scenarios": os.path.join(mdir, f"{tag}_TRICERATOPS_scenarios.csv"),
        "field": os.path.join(mdir, f"{tag}_TRICERATOPS_field.png"),
        #"fits": os.path.join(mdir, f"{tag}_TRICERATOPS_fits.png"),
        "fits": fits_png or os.path.join(mdir, f"{tag}_TRICERATOPS_fits.png"),        
        "report": os.path.join(data_root, f"Sector_{int(sector):04d}",
                               "NEMESIS_FFI_TLS_Report",
                               f"{tag}_TLS_Report_multi_page.pdf"),
    }
    out: dict[str, Any] = {"TIC_ID": int(tic), "Sector": int(sector), "status": "",
                           "page_png": "", "report_pdf": paths["report"],
                           "appended_to": append_to or "", "backup": "",
                           "pages_before": -1, "pages_after": -1, "missing_keys": ""}
    missing = [k for k, p in paths.items() if not os.path.isfile(p)]
    if append_to and not os.path.isfile(append_to):
        missing.append("append_to")
    if missing:
        out["status"] = "missing: " + ", ".join(missing)
        return out

    with open(paths["summary"]) as fh:
        s = json.load(fh)
    out["missing_keys"] = ";".join(k for k in _PAGE7_KEYS if s.get(k) is None)

    def _f(key: str) -> float:
        """Summary value as float; NaN when absent or not numeric."""
        try:
            return float(s.get(key))
        except (TypeError, ValueError):
            return float("nan")

    def _fmt(key: str, spec: str, scale: float = 1.0) -> str:
        """Formatted summary value, 'n/a' when not finite."""
        v = _f(key) * scale
        return format(v, spec) if np.isfinite(v) else "n/a"

    params = [
        ("Period [d]", _fmt("TRICERATOPS_P_orb_d", ".6f")),
        ("T0 [BTJD]", _fmt("TRICERATOPS_T0_BTJD", ".5f")),
        ("Duration [hr]", _fmt("TRICERATOPS_duration_hr", ".3f")),
        ("Depth [ppt]", _fmt("TRICERATOPS_tdepth_in", ".3f", 1e3)),
        ("Rp [RE]", _fmt("TRICERATOPS_Rp_RE", ".3f")),
        ("Aperture [px]", _fmt("TRICERATOPS_aperture_npix", ".0f")),
        ("Flux ratio", _fmt("TRICERATOPS_fluxratio_target", ".5f")),
        ("LC points", _fmt("TRICERATOPS_LC_points", ".0f")),
    ]
    stats = [
        ("FPP p50", _fmt("TRICERATOPS_FPP_p50", ".5f")),
        ("FPP p16", _fmt("TRICERATOPS_FPP_p16", ".5f")),
        ("FPP p84", _fmt("TRICERATOPS_FPP_p84", ".5f")),
        ("FPP max", _fmt("TRICERATOPS_FPP_max", ".5f")),
        ("NFPP p50", _fmt("TRICERATOPS_NFPP_p50", ".5f")),
        ("NFPP max", _fmt("TRICERATOPS_NFPP_max", ".5f")),
        ("Trials", _fmt("TRICERATOPS_N_trials", ".0f")),
        ("MC samples", _fmt("TRICERATOPS_mc_samples", ",.0f")),
    ]
    scenarios = merged_top_scenarios(pd.read_csv(paths["scenarios"]))
    disposition = str(s.get("TRICERATOPS_disposition") or "").split(" (")[0] or None
    subtitle = (f"TIC {int(tic)}   |   Sector {int(sector)}   |   "
                f"{s.get('TRICERATOPS_ephemeris_source') or 'n/a'}   |   exptime "
                f"{_f('TRICERATOPS_exptime_d') * 1440.0:.0f} min   |   "
                f"module v{s.get('TRICERATOPS_module_version') or 'n/a'}")
    footnote = ("Per-scenario values are independent medians over trials and do not "
                "sum to unity.\nFPP is computed per trial, then summarized.\n"
                "Classification criteria from Giacalone et al. (2021);\nthe "
                "validated-planet tier is unavailable at FFI cadence (their Section 4.2).\n"
                "Scenario-fit panels show one trial; all values are over all trials.")

    size = pdf_page_size_px(paths["report"], verbose=False)
    if size is None:
        out["status"] = "failed: report page size unreadable"
        return out
    page = compose_triceratops_report_page(
        paths["field"], paths["fits"], os.path.join(mdir, f"{tag}_TRICERATOPS_page.png"),
        page_size_px=size, subtitle=subtitle, params=params, stats=stats,
        scenarios=scenarios, disposition=disposition, footnote=footnote, verbose=verbose)
    if not page:
        out["status"] = "failed: page not composed"
        return out
    out["page_png"] = page
    if not append_to:
        out["status"] = "page only"
        return out

    out["pages_before"] = _pdf_n_pages(append_to)
    if backup_dir:
        dest = os.path.join(backup_dir, f"Sector_{int(sector):04d}",
                            os.path.basename(append_to))
        if not os.path.exists(dest):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(append_to, dest)
        out["backup"] = dest
    appended = append_png_to_pdf(append_to, page, in_place=True,
                                 keep_pages=int(keep_pages), verbose=verbose)
    out["pages_after"] = _pdf_n_pages(append_to)
    ok = bool(appended) and out["pages_after"] == int(keep_pages) + 1
    out["status"] = ("ok" if ok else
                     f"failed: append returned {appended!r}, pages {out['pages_after']}")
    return out

def main(argv: list[str] | None = None) -> int:
    """
    Command-line entry point, one star-sector per invocation.

    Forces the Agg matplotlib backend, which is required on HPC compute
    nodes with no display. The backend is NOT touched at module import, so
    importing this module inside Jupyter leaves inline plotting alone.

    ``--posterior-path`` is required: period, mid-transit time and duration
    come from the jaxoplanet posterior, so a batch run can never fall back to
    the search ephemeris without saying so. The depth stays the search depth
    from the bestfit. The radius ceiling is off by default; candidates above
    :data:`RP_FLAG_RE` are flagged rather than skipped.

    Parameters
    ----------
    argv : list of str, optional
        Argument list. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        0 on success, or on a deliberate radius-ceiling skip when
        ``--rp-max`` is set; 1 on failure. Suitable as a disBatch exit
        status.

    Examples
    --------
    >>> main(["--tic", "259377017", "--sector", "4",
    ...       "--report-dir", "/path/TLS_Report/",
    ...       "--savelcpath", "/path/LCs/",
    ...       "--hdu", "/path/cutout.fits",
    ...       "--posterior-path",
    ...       "/path/TIC_259377017_Sector_0004_jaxoplanet_posterior.csv"])
    """
    import matplotlib
    matplotlib.use("Agg", force=True)

    p = argparse.ArgumentParser(
        description="TRICERATOPS validation of one NEMESIS star-sector.")
    p.add_argument("--tic", type=int, required=True)
    p.add_argument("--sector", type=int, required=True)
    p.add_argument("--report-dir", required=True)
    p.add_argument("--savelcpath", required=True)
    p.add_argument("--hdu", required=True,
                   help="path to the cached TESScut FITS")
    p.add_argument("--posterior-path", required=True,
                   help="jaxoplanet *_jaxoplanet_posterior.csv for this "
                        "star-sector; supplies period, T0 and duration")
    p.add_argument("--bestfit-path", default=None)
    p.add_argument("--outdir", default=None)
    p.add_argument("--trilegal-dir", default=None)
    p.add_argument("--lk-cache-dir", default=None,
                   help="cache dir for the 22x22 TESScut cutout TRICERATOPS "
                        "downloads for itself; must be the PARENT of "
                        "TESSCut/, not TESSCut/ itself.")
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--mc-samples", type=int, default=5_000_000,
                   help="calc_probs N per trial (09-29: 5,000,000)")
    p.add_argument("--search", default="TLS", choices=["TLS", "BLS"],
                   help="search whose bestfit supplies the depth")
    p.add_argument("--flux-type", default="Detrended",
                   choices=["SAP", "PLD", "Detrended"])
    p.add_argument("--n-durations", type=int, default=N_DURATIONS_DEFAULT,
                   help="fold half-width in transit durations "
                        "(10-14-2026: 3 for every target)")
    p.add_argument("--n-bins", type=int, default=None)
    p.add_argument("--rp-max", type=_float_or_none, default=None,
                   help="radius ceiling in Earth radii; candidates above it "
                        "are skipped. Default none: every candidate is "
                        "evaluated and Rp > 8 is flagged instead.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--parallel", action="store_true")
    p.add_argument("--no-bestfit-update", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no-report-append", action="store_true")
    p.add_argument("--report-split", action="store_true",
                   help="two pages instead of one, each figure near full size")
    p.add_argument("--report-layout", default="auto",
                   choices=["auto", "stacked", "side_by_side"])
    args = p.parse_args(argv)

    try:
        res = run_triceratops_on_nemesis(
            tic_id=args.tic, sector=args.sector,
            report_dir=args.report_dir, savelcpath=args.savelcpath,
            bestfit_path=args.bestfit_path, hdu=args.hdu,
            lightkurve_cache_dir=args.lk_cache_dir,
            outdir=args.outdir, trilegal_dir=args.trilegal_dir,
            n_trials=args.n_trials, search=args.search,
            flux_type=args.flux_type, n_durations=args.n_durations,
            n_bins=args.n_bins, mc_samples=args.mc_samples,
            parallel=args.parallel,
            rp_max_earth=args.rp_max, seed=args.seed,
            update_bestfit=not args.no_bestfit_update,
            save_plots=not args.no_plots, verbose=not args.quiet,
            append_to_vetting_report=not args.no_report_append,
            report_layout=args.report_layout,
            report_split=args.report_split,
            posterior_path=args.posterior_path,
        )
    except Exception as exc:
        print(f"FAILED TIC {args.tic} S{args.sector:04d}: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    eph = (res.get("inputs") or {}).get("ephemeris_source", "unknown")
    print(f"TIC {res['tic_id']} S{res['sector']:04d}  "
          f"FPP {res['FPP_median']:.4f} +/- {res['FPP_std']:.4f}  "
          f"{res['disposition']}  [ephemeris: {eph}]")
    return 0

def main_OLD(argv: list[str] | None = None) -> int:
    """
    Command-line entry point, one star-sector per invocation.

    Forces the Agg matplotlib backend, which is required on HPC compute
    nodes with no display. The backend is NOT touched at module import, so
    importing this module inside Jupyter leaves inline plotting alone.

    Parameters
    ----------
    argv : list of str, optional
        Argument list. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        0 on success or a deliberate radius-ceiling skip, 1 on failure.
        Suitable as a disBatch exit status.

    Examples
    --------
    >>> main(["--tic", "270380593", "--sector", "4",
    ...       "--report-dir", "/path/TLS_Report/",
    ...       "--savelcpath", "/path/LCs/",
    ...       "--hdu", "/path/cutout.fits"])
    """
    import matplotlib
    matplotlib.use("Agg", force=True)

    p = argparse.ArgumentParser(
        description="TRICERATOPS validation of one NEMESIS star-sector.")
    p.add_argument("--tic", type=int, required=True)
    p.add_argument("--sector", type=int, required=True)
    p.add_argument("--report-dir", required=True)
    p.add_argument("--savelcpath", required=True)
    p.add_argument("--hdu", required=True,
                   help="path to the cached TESScut FITS")
    p.add_argument("--bestfit-path", default=None)
    p.add_argument("--outdir", default=None)
    p.add_argument("--trilegal-dir", default=None)
    p.add_argument("--lk-cache-dir", default=None,
                   help="cache dir for the TESScut cutout TRICERATOPS "
                        "downloads for itself; normally "
                        "settings['downloadpath']. Required on compute "
                        "nodes with no outbound network.")
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--search", default="TLS", choices=["TLS", "BLS"])
    p.add_argument("--flux-type", default="Detrended",
                   choices=["SAP", "PLD", "Detrended"])
    p.add_argument("--n-durations", type=int, default=5)
    p.add_argument("--n-bins", type=int, default=None)
    p.add_argument("--rp-max", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--parallel", action="store_true")
    p.add_argument("--no-bestfit-update", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no-report-append", action="store_true")
    p.add_argument("--report-split", action="store_true",
                   help="two pages instead of one, each figure near full size")
    p.add_argument("--report-layout", default="auto",
                   choices=["auto", "stacked", "side_by_side"])
    args = p.parse_args(argv)

    try:
        res = run_triceratops_on_nemesis(
            tic_id=args.tic, sector=args.sector,
            report_dir=args.report_dir, savelcpath=args.savelcpath,
            bestfit_path=args.bestfit_path, hdu=args.hdu,
            lightkurve_cache_dir=args.lk_cache_dir,
            outdir=args.outdir, trilegal_dir=args.trilegal_dir,
            n_trials=args.n_trials, search=args.search,
            flux_type=args.flux_type, n_durations=args.n_durations,
            n_bins=args.n_bins, parallel=args.parallel,
            rp_max_earth=args.rp_max, seed=args.seed,
            update_bestfit=not args.no_bestfit_update,
            save_plots=not args.no_plots, verbose=not args.quiet,
#             vetting_report_in_place=args.report_in_place,
            append_to_vetting_report=not args.no_report_append,
            report_layout=args.report_layout,
            report_split=args.report_split,
        )
    except Exception as exc:
        print(f"FAILED TIC {args.tic} S{args.sector:04d}: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"TIC {res['tic_id']} S{res['sector']:04d}  "
          f"FPP {res['FPP_median']:.4f} +/- {res['FPP_std']:.4f}  "
          f"{res['disposition']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
