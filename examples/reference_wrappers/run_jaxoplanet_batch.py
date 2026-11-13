#!/usr/bin/env python
"""
run_jaxoplanet_batch.py

Worker for one jaxoplanet refit batch. Reads a batch CSV produced by
generate_jaxoplanet_manifest.py and calls refine_transit_parameters on each
row, saving both the summary CSV and the InferenceData.

Usage (called by disBatch, not directly):
    python run_jaxoplanet_batch.py <batch_csv> [--n_chains 4] [--save_idata] ...

Each row in the batch CSV has columns:
    TIC_ID, Sector, lc_path, bestfit_path, outdir, save_name, batch_key

Ordering constraint: argument parsing and environment configuration happen
before any JAX import. numpyro.set_host_device_count only takes effect if the
XLA backend has not yet initialized, and each host device otherwise spawns its
own BLAS thread pool, which oversubscribes the allocation.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
import traceback
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Machine detection. Same marker and failure mode as run_injection_batch.py.
# ---------------------------------------------------------------------------
_HPC_MARKER = "/mnt/home/dfeliz"
ON_HPC = os.path.isdir(_HPC_MARKER)

if ON_HPC:
    SCRIPTS_DIR = "/mnt/home/dfeliz/TESS_Mdwarf_Survey2/TESS_Mdwarf_Survey2_scripts/"
else:
    SCRIPTS_DIR = "/Users/daxfeliz/Desktop/TESS/00_Current_Pipeline/"

CODEPATH = os.environ.get("NEMESIS_CODEPATH", SCRIPTS_DIR)
if not os.path.isdir(CODEPATH):
    raise RuntimeError(
        f"NEMESIS_CODEPATH = {CODEPATH!r} does not exist. Detected machine: "
        f"{'HPC' if ON_HPC else 'local'} (marker {_HPC_MARKER!r})."
    )
if CODEPATH not in sys.path:
    sys.path.insert(0, CODEPATH)


class TargetTimeout(Exception):
    """Raised by the SIGALRM handler when one target exceeds its budget."""


def _alarm_handler(signum, frame):  # noqa: ANN001
    raise TargetTimeout("per-target wall-time budget exceeded")


def configure_runtime(n_chains: int, verbose: bool = True) -> int:
    """
    Pin threads, select the CPU backend, and allocate host devices for NUTS.

    Must run before the first JAX array is created. Called once per worker.

    Parameters
    ----------
    n_chains : int
        Number of NUTS chains, and therefore the number of XLA host devices to
        request. Should equal the SLURM cpus-per-worker.
    verbose : bool
        Print the resulting device count.

    Returns
    -------
    int
        jax.local_device_count() after configuration. Compare against n_chains:
        if smaller, chains will run sequentially and the batch takes n_chains
        times longer.

    Examples
    --------
    >>> configure_runtime(4)   # doctest: +SKIP
    4
    """
    os.environ.setdefault("MPLBACKEND", "Agg")
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"

    import numpyro
    numpyro.set_platform("cpu")
    numpyro.set_host_device_count(int(n_chains))

    import jax
    n_dev = int(jax.local_device_count())
    if verbose:
        print(f"[worker] XLA host devices: {n_dev} (requested {n_chains}); "
              f"MPLBACKEND={os.environ.get('MPLBACKEND')}", flush=True)
    if n_dev < n_chains:
        print(f"[worker] WARNING: only {n_dev} devices for {n_chains} chains. "
              f"NUTS will run chains sequentially.", flush=True)
    return n_dev


def save_inference_data(idata: Any, path_stem: str, thin: int = 1,
                        verbose: bool = True) -> Optional[str]:
    """
    Thin and write an ArviZ InferenceData, with a pickle fallback.

    NetCDF needs netcdf4 or h5netcdf in the venv. When neither is importable
    ArviZ raises, and losing every posterior over a missing optional dependency
    is not acceptable in a multi-day batch, so this falls back to pickle.

    Parameters
    ----------
    idata : arviz.InferenceData
        Posterior returned by refine_transit_parameters.
    path_stem : str
        Absolute path without extension. '.nc' or '.pkl' is appended.
    thin : int
        Keep every Nth draw. 1 keeps everything. Thinning by 5 on a
        4-chain 3000-draw run leaves 2400 draws, ample for percentiles,
        and cuts the file by roughly 5x.
    verbose : bool
        Print the written path.

    Returns
    -------
    str or None
        Path written, or None if both routes failed.

    Examples
    --------
    >>> save_inference_data(idata, '/ceph/fits/TIC_1_Sector_0004', thin=5)  # doctest: +SKIP
    '/ceph/fits/TIC_1_Sector_0004_jaxoplanet_idata.nc'
    """
    thin = int(max(1, thin))
    try:
        obj = idata.sel(draw=slice(None, None, thin)) if thin > 1 else idata
    except Exception as exc:
        print(f"  [idata] thinning failed ({type(exc).__name__}: {exc}); "
              f"saving unthinned", flush=True)
        obj = idata

    nc_path = f"{path_stem}_jaxoplanet_idata.nc"
    try:
        obj.to_netcdf(nc_path)
        if verbose:
            mb = os.path.getsize(nc_path) / 1e6
            print(f"  [idata] {nc_path} ({mb:.1f} MB)", flush=True)
        return nc_path
    except Exception as exc:
        print(f"  [idata] to_netcdf failed ({type(exc).__name__}: {exc}); "
              f"falling back to pickle", flush=True)

    pkl_path = f"{path_stem}_jaxoplanet_idata.pkl"
    try:
        import pickle
        with open(pkl_path, "wb") as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
        if verbose:
            mb = os.path.getsize(pkl_path) / 1e6
            print(f"  [idata] {pkl_path} ({mb:.1f} MB)", flush=True)
        return pkl_path
    except Exception as exc:
        print(f"  [idata] pickle fallback also failed "
              f"({type(exc).__name__}: {exc}). Posterior NOT saved.",
              flush=True)
        return None


# ---------------------------------------------------------------------------
# Per-target status table and quality-aware resume. Pure numpy and pandas,
# no JAX import, so --summarize is safe to run on a login node.
# ---------------------------------------------------------------------------
STATUS_COLUMNS: tuple = (
    "TIC_ID", "Sector", "save_name", "batch_csv", "attempt_utc", "status",
    "over_budget", "wall_minutes", "n_cadences_fit", "r_hat_max",
    "ess_bulk_min", "ess_tail_min", "n_divergences", "r_ceiling_frac",
    "r_prior_hi", "r_hi_sigmas", "n_warmup", "n_samples", "target_accept",
    "likelihood", "student_t_nu", "timeout_minutes", "refit_reason", "error",
)


def _as_float(value: Any) -> float:
    """
    Convert one table cell to float, returning NaN when missing or unparseable.

    Parameters
    ----------
    value : Any
        Cell value read from a CSV row.

    Returns
    -------
    float
        The value as a float, or NaN.

    Example
    -------
    >>> _as_float("0.25"), _as_float(None)
    (0.25, nan)
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _as_text(value: Any) -> str:
    """
    Convert one table cell to text, mapping None and NaN to the empty string.

    Parameters
    ----------
    value : Any
        Cell value read from a CSV row.

    Returns
    -------
    str
        Text form of the value, or "" when missing.

    Example
    -------
    >>> _as_text(float("nan")), _as_text("FAILED")
    ('', 'FAILED')
    """
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return str(value)


def _parse_bool(value: Any) -> Optional[bool]:
    """
    Read a boolean that may have round-tripped through CSV as text.

    Parameters
    ----------
    value : Any
        True/False, numpy bool, 'True'/'False', 1/0, or missing.

    Returns
    -------
    bool or None
        Parsed value, or None when it cannot be read.

    Example
    -------
    >>> _parse_bool("False"), _parse_bool(True), _parse_bool(float("nan"))
    (False, True, None)
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1", "1.0", "yes"):
        return True
    if text in ("false", "0", "0.0", "no"):
        return False
    return None


def _read_posterior_row(posterior_csv: str) -> Optional[Dict[str, Any]]:
    """
    Read the single summary row of one target's posterior CSV.

    Parameters
    ----------
    posterior_csv : str
        Path to '<stem>_jaxoplanet_posterior.csv'.

    Returns
    -------
    dict or None
        The row as a dict. None if the file does not exist. An empty dict if
        it exists but cannot be read or has no rows.

    Example
    -------
    >>> _read_posterior_row("/nonexistent.csv") is None
    True
    """
    if not os.path.isfile(posterior_csv):
        return None
    import pandas as pd
    try:
        df = pd.read_csv(posterior_csv)
    except Exception:
        return {}
    if len(df) == 0:
        return {}
    return df.iloc[0].to_dict()


def classify_posterior(
    rec: Optional[Dict[str, Any]],
    *,
    ceiling_max: float = 0.01,
    expect_likelihood: Optional[str] = None,
) -> str:
    """
    Assign one quality status to a posterior summary row (the R1 resume test).

    Checks run in this order and the first failure wins: file missing, file
    unreadable, wrong likelihood, convergence gate failed, Rp/Rs posterior
    pressed against the prior ceiling.

    Parameters
    ----------
    rec : dict or None
        Output of _read_posterior_row, or summary.iloc[0].to_dict().
    ceiling_max : float
        Largest acceptable 'MCMC r ceiling frac' (fraction of draws within
        2 percent of r_hi). 0.01 matches the refit module's warning.
    expect_likelihood : str or None
        When given, a row recording a different 'MCMC likelihood' fails.

    Returns
    -------
    str
        One of 'NO_POSTERIOR', 'UNREADABLE', 'WRONG_LIKELIHOOD',
        'UNCONVERGED', 'CEILING', 'OK'.

    Example
    -------
    >>> classify_posterior({"MCMC converged": True,
    ...                     "MCMC r ceiling frac": 0.035})
    'CEILING'
    """
    if rec is None:
        return "NO_POSTERIOR"
    if not rec:
        return "UNREADABLE"
    if expect_likelihood is not None and "MCMC likelihood" in rec:
        got = _as_text(rec.get("MCMC likelihood")).strip().lower()
        if got != str(expect_likelihood).strip().lower():
            return "WRONG_LIKELIHOOD"
    if _parse_bool(rec.get("MCMC converged")) is not True:
        return "UNCONVERGED"
    frac = _as_float(rec.get("MCMC r ceiling frac"))
    if frac == frac and frac > float(ceiling_max):
        return "CEILING"
    return "OK"


def _diag_fields(rec: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """
    Pull the diagnostic columns of a posterior row into status-table names.

    Parameters
    ----------
    rec : dict or None
        Posterior summary row, or None.

    Returns
    -------
    dict
        n_cadences_fit, r_hat_max, ess_bulk_min, ess_tail_min,
        n_divergences, r_ceiling_frac, r_prior_hi. NaN where absent.

    Example
    -------
    >>> _diag_fields(None)["r_hat_max"]
    nan
    """
    r = rec or {}
    return {
        "n_cadences_fit": _as_float(r.get("N cadences fit")),
        "r_hat_max": _as_float(r.get("MCMC rhat max")),
        "ess_bulk_min": _as_float(r.get("MCMC ess bulk min")),
        "ess_tail_min": _as_float(r.get("MCMC ess tail min")),
        "n_divergences": _as_float(r.get("MCMC n divergences")),
        "r_ceiling_frac": _as_float(r.get("MCMC r ceiling frac")),
        "r_prior_hi": _as_float(r.get("MCMC r prior hi")),
    }


def append_status_row(status_csv: str, row: Dict[str, Any]) -> None:
    """
    Append one attempt to a worker's status CSV, writing the header once.

    Each worker owns exactly one status file, so there is no write contention
    between disBatch tasks. A failed write prints a warning and never stops
    the worker.

    Parameters
    ----------
    status_csv : str
        Path, normally '<batch_csv stem>_status.csv'.
    row : dict
        Values keyed by STATUS_COLUMNS. Missing keys are written empty.

    Example
    -------
    >>> append_status_row("/tmp/demo_status.csv",        # doctest: +SKIP
    ...                   {"TIC_ID": 1, "Sector": 5, "status": "OK"})
    """
    import pandas as pd
    rec = {c: row.get(c, "") for c in STATUS_COLUMNS}
    try:
        write_header = not os.path.isfile(status_csv)
        pd.DataFrame([rec], columns=list(STATUS_COLUMNS)).to_csv(
            status_csv, mode="a", header=write_header, index=False)
    except Exception as exc:
        print(f"[status] WARNING: could not write {status_csv}: "
              f"{type(exc).__name__}: {exc}", flush=True)


def summarize_jaxoplanet_status(
    batch_dir: str,
    nbatches: int,
    *,
    ceiling_max: float = 0.01,
    batch_pattern: str = "jaxoplanet_batch_{:03d}.csv",
    out_csv: Optional[str] = None,
    verbose: bool = True,
) -> "pd.DataFrame":
    """
    Build one status table across every batch, with no log parsing.

    For each target listed in each batch CSV, the status comes from the
    target's posterior CSV via classify_posterior, the same test the worker
    uses to decide whether to refit. The worker status CSV, when present,
    adds the attempt count, the last attempt time, over-budget flags, and
    error text for FAILED targets. Runs made before status files existed are
    still covered: a missing posterior reports as NO_POSTERIOR.

    Batch CSV paths are built from batch_dir, nbatches and batch_pattern.
    No directory is listed or searched.

    Parameters
    ----------
    batch_dir : str
        Directory holding the batch CSVs.
    nbatches : int
        Number of batches, numbered from 1.
    ceiling_max : float
        Ceiling threshold passed to classify_posterior.
    batch_pattern : str
        Batch file name with one integer format field.
    out_csv : str or None
        Output path. Defaults to '<batch_dir>/jaxoplanet_status_all.csv'.
    verbose : bool
        Print status counts and every target that will be refit.

    Returns
    -------
    pd.DataFrame
        One row per target. 'will_refit' is True wherever status != 'OK'.

    Example
    -------
    >>> df = summarize_jaxoplanet_status(                  # doctest: +SKIP
    ...     "/mnt/home/dfeliz/ceph/NEMESIS_II/jaxoplanet/batches", 16)
    """
    import numpy as np
    import pandas as pd

    rows = []
    n_read = 0
    for k in range(1, int(nbatches) + 1):
        batch_csv = os.path.join(batch_dir, batch_pattern.format(k))
        if not os.path.isfile(batch_csv):
            print(f"[summary] WARNING: {batch_csv} not found, skipped",
                  flush=True)
            continue
        batch = pd.read_csv(batch_csv)
        n_read += 1
        tag = os.path.basename(batch_csv)

        last: Dict[Any, Dict[str, Any]] = {}
        n_att: Dict[Any, int] = {}
        status_csv = os.path.splitext(batch_csv)[0] + "_status.csv"
        if os.path.isfile(status_csv):
            try:
                st = pd.read_csv(status_csv)
            except Exception as exc:
                print(f"[summary] WARNING: unreadable {status_csv}: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                st = None
            if st is not None:
                for _, srow in st.iterrows():
                    key = (int(float(srow["TIC_ID"])), int(float(srow["Sector"])))
                    last[key] = srow.to_dict()
                    n_att[key] = n_att.get(key, 0) + 1

        for _, row in batch.iterrows():
            tic = int(row["TIC_ID"])
            sec = int(row["Sector"])
            save_name = str(row.get("save_name", f"TIC_{tic}_Sector_{sec:04d}"))
            post = os.path.join(str(row["outdir"]),
                                save_name) + "_jaxoplanet_posterior.csv"
            rec = _read_posterior_row(post)
            status = classify_posterior(rec, ceiling_max=ceiling_max)
            lr = last.get((tic, sec), {})
            if status == "NO_POSTERIOR" and _as_text(lr.get("status")) == "FAILED":
                status = "FAILED"
            wall_s = _as_float((rec or {}).get("wall_seconds"))
            wall_min = (wall_s / 60.0 if np.isfinite(wall_s)
                        else _as_float(lr.get("wall_minutes")))
            out = {
                "TIC_ID": tic,
                "Sector": sec,
                "save_name": save_name,
                "batch_csv": tag,
                "status": status,
                "will_refit": status != "OK",
                "n_attempts_logged": int(n_att.get((tic, sec), 0)),
                "last_attempt_utc": _as_text(lr.get("attempt_utc")),
                "over_budget": _as_text(lr.get("over_budget")),
                "wall_minutes": wall_min,
                "error": (_as_text(lr.get("error"))
                          if status == "FAILED" else ""),
                "posterior_csv": post,
            }
            out.update(_diag_fields(rec))
            rows.append(out)

    df = pd.DataFrame(rows)
    if out_csv is None:
        out_csv = os.path.join(batch_dir, "jaxoplanet_status_all.csv")
    df.to_csv(out_csv, index=False)

    if verbose:
        print(f"[summary] {len(df)} targets from {n_read} batch files "
              f"-> {out_csv}", flush=True)
        if len(df):
            print(df["status"].value_counts().to_string(), flush=True)
            bad = df[df["will_refit"]]
            known = bad["wall_minutes"].dropna()
            print(f"[summary] {len(bad)} targets will be refit on the next "
                  f"submission. Previously logged fit time for the "
                  f"{len(known)} with a recorded wall time: "
                  f"{known.sum() / 60.0:.1f} h. NO_POSTERIOR targets from "
                  f"runs before status files have no recorded time.",
                  flush=True)
            cols = ["batch_csv", "TIC_ID", "Sector", "status",
                    "n_cadences_fit", "wall_minutes", "r_ceiling_frac",
                    "ess_tail_min", "r_hat_max"]
            print(bad.sort_values(["status", "batch_csv", "Sector"])[cols]
                  .to_string(index=False), flush=True)
    return df

def run_jaxoplanet_batch(
    batch_csv: str,
    *,
    n_chains: int = 4,
    n_warmup: int = 2000,
    n_samples: int = 3000,
    target_accept: float = 0.95,
    ld_mode: str = "gaussian",
    prior_style: str = "paper1",
    r_hi_sigmas: float = 10.0,
    r_hi_abs: Optional[float] = None,
    rho_star_prior: bool = True,
    rho_frac_err: float = 0.15,
    likelihood: str = "student_t",
    student_t_nu: float = 4.0,
    point_estimate: str = "median",
    n_pred_draws: int = 300,
    make_corner: bool = True,
    make_trace: bool = True,
    make_timeseries: bool = True,
    save_idata: bool = True,
    idata_thin: int = 5,
    timeout_minutes: float = 360.0,
    ceiling_max: float = 0.01,
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """
    Drain one batch CSV, fitting every star-sector in it sequentially.

    One worker owns one batch and works through it in order, matching the
    disBatch architecture used by run_injection_batch.py: the unit of
    parallelism is the batch, not the target.

    Resume (R1). A target is skipped only when its posterior CSV exists AND
    'MCMC converged' is True AND 'MCMC r ceiling frac' <= ceiling_max AND the
    recorded likelihood matches the request. Missing, unconverged,
    ceiling-limited, unreadable, or wrong-likelihood posteriors are refit and
    overwritten. overwrite=True refits everything.

    Radius prior edge. By default the Uniform radius prior ends at
    RP_RS + r_hi_sigmas * sigma_RP_RS, computed per target. When r_hi_abs is
    given, the edge is that fixed number for every target instead, and
    r_hi_sigmas is ignored and recorded as NaN in the status CSV. r_hi_abs is
    forwarded to the refit only when set, so the worker still runs against a
    refit module that predates the option.

    Wall-time budget. The previous SIGALRM timeout could not interrupt NUTS,
    because sampling runs inside one compiled XLA call and Python only acts
    on the signal after it returns. It therefore discarded completed fits
    without saving any time. It is replaced by a flag: every finished fit is
    saved, and any fit longer than timeout_minutes is marked
    over_budget=True in the status CSV. A truly runaway fit runs until the
    SLURM walltime, as it effectively did before.

    Status table. Every attempt appends one row to
    '<batch_csv stem>_status.csv': status, wall time, diagnostics, settings,
    and error text. See summarize_jaxoplanet_status for the combined view.

    The observation likelihood is forwarded explicitly. refine_transit_parameters
    defaults to 'normal'; this worker defaults to 'student_t' with nu = 4.0,
    the adopted survey setting, so a batch cannot silently fall back to the
    Gaussian model. Pre-flight checks run before any fit: an unknown
    likelihood name, a deployed refit module that does not accept the
    likelihood keyword, and a requested r_hi_abs that the deployed refit does
    not accept or that is not a usable number all stop the worker
    immediately rather than failing every target one by one. After each fit
    the recorded 'MCMC likelihood' is compared with the request.

    Parameters
    ----------
    batch_csv : str
        Batch CSV from generate_jaxoplanet_manifest.py.
    n_chains, n_warmup, n_samples, target_accept : int, int, int, float
        NUTS settings passed through to refine_transit_parameters.
    ld_mode, prior_style, rho_star_prior, rho_frac_err : ...
        Model configuration passed through unchanged.
    r_hi_sigmas : float
        Uniform Rp/Rs prior ceiling in units of sigma_RP_RS above the search
        value. Default 10.0. Ignored when r_hi_abs is given.
    r_hi_abs : float or None
        Fixed Uniform Rp/Rs prior ceiling for every target, e.g. 1.0. None
        keeps the per-target r_hi_sigmas rule.
    likelihood : str
        'student_t' (survey default) or 'normal'.
    student_t_nu : float
        Student-t degrees of freedom. Must exceed 2. Ignored for 'normal'.
    point_estimate : str
        'median' or 'maxl'. Affects figures only, never the table.
    n_pred_draws : int
        Credible-band draws in the phase-folded figure.
    make_corner, make_trace, make_timeseries, save_idata : bool
        Output switches.
    idata_thin : int
        Posterior thinning factor before writing.
    timeout_minutes : float
        Per-target wall-time budget for the over_budget flag only. 0 disables
        the flag. Default 360.
    ceiling_max : float
        Largest acceptable 'MCMC r ceiling frac' for a fit to count as done.
    overwrite : bool
        Refit every target regardless of existing posteriors.
    verbose : bool
        Per-target progress.

    Raises
    ------
    SystemExit
        If the batch CSV is unreadable or missing a required column, if the
        likelihood settings are invalid, if the deployed refit module does
        not accept the likelihood keyword, or if r_hi_abs is requested but
        unsupported or unusable.

    Example
    -------
    >>> run_jaxoplanet_batch(                                  # doctest: +SKIP
    ...     "/mnt/home/dfeliz/ceph/NEMESIS_II/jaxoplanet/pilot_flat/"
    ...     "jaxoplanet_batch_001.csv",
    ...     likelihood="student_t", student_t_nu=4.0, r_hi_abs=1.0)
    """
    t_start = time.time()

    import inspect

    import numpy as np
    import pandas as pd

    likelihood = str(likelihood).strip().lower()
    if likelihood not in ("normal", "student_t"):
        raise SystemExit(
            f"likelihood must be 'normal' or 'student_t', got {likelihood!r}")
    if likelihood == "student_t" and not float(student_t_nu) > 2.0:
        raise SystemExit(
            f"student_t_nu must exceed 2 for a finite variance, "
            f"got {student_t_nu}")
    if r_hi_abs is not None:
        try:
            _edge = float(r_hi_abs)
        except (TypeError, ValueError):
            raise SystemExit(f"r_hi_abs must be a number, got {r_hi_abs!r}")
        if not (np.isfinite(_edge) and _edge > 0.01):
            raise SystemExit(
                f"r_hi_abs must be a finite number above 0.01, got {r_hi_abs!r}")

    try:
        batch = pd.read_csv(batch_csv)
    except Exception as exc:
        raise SystemExit(f"cannot read batch CSV {batch_csv!r}: "
                         f"{type(exc).__name__}: {exc}")

    required = ("TIC_ID", "Sector", "lc_path", "bestfit_path", "outdir")
    missing = [c for c in required if c not in batch.columns]
    if missing:
        raise SystemExit(
            f"batch CSV {batch_csv!r} is missing {missing}. Regenerate it with "
            f"generate_jaxoplanet_manifest.py."
        )

    configure_runtime(n_chains, verbose=verbose)
    from nemesis_jaxoplanet_refit import refine_transit_parameters  # type: ignore

    refit_params = inspect.signature(refine_transit_parameters).parameters
    if "likelihood" not in refit_params or "student_t_nu" not in refit_params:
        import nemesis_jaxoplanet_refit as _refit_mod  # type: ignore
        raise SystemExit(
            f"the deployed nemesis_jaxoplanet_refit.py at "
            f"{_refit_mod.__file__} does not accept likelihood/student_t_nu. "
            f"It is an older copy; deploy the current module before running."
        )
    if r_hi_abs is not None and "r_hi_abs" not in refit_params:
        import nemesis_jaxoplanet_refit as _refit_mod  # type: ignore
        raise SystemExit(
            f"r_hi_abs={r_hi_abs} was requested but the deployed "
            f"nemesis_jaxoplanet_refit.py at {_refit_mod.__file__} does not "
            f"accept r_hi_abs. Apply the fixed-edge edits before running."
        )
    edge_kwargs: Dict[str, Any] = (
        {"r_hi_abs": float(r_hi_abs)} if r_hi_abs is not None else {}
    )

    budget_min = (float(timeout_minutes)
                  if timeout_minutes and timeout_minutes > 0 else 0.0)
    status_csv = os.path.splitext(batch_csv)[0] + "_status.csv"
    settings = {
        "r_hi_sigmas": (float(r_hi_sigmas) if r_hi_abs is None else np.nan),
        "n_warmup": int(n_warmup),
        "n_samples": int(n_samples),
        "target_accept": float(target_accept),
        "likelihood": likelihood,
        "student_t_nu": (float(student_t_nu)
                         if likelihood == "student_t" else np.nan),
        "timeout_minutes": budget_min,
    }

    n_done = n_skip = n_fail = n_unconverged = n_ceiling = n_over = 0
    tag = os.path.basename(batch_csv)
    nu_txt = f", nu={float(student_t_nu):g}" if likelihood == "student_t" else ""
    edge_txt = (f"r_hi_abs={float(r_hi_abs):g}" if r_hi_abs is not None
                else f"r_hi_sigmas={float(r_hi_sigmas):g}")
    print(f"[worker] {tag}: {len(batch)} targets, n_chains={n_chains}, "
          f"warmup={n_warmup}, samples={n_samples}, "
          f"likelihood={likelihood}{nu_txt}, "
          f"rho_star_prior={rho_star_prior}, {edge_txt}, "
          f"ceiling_max={float(ceiling_max):g}, "
          f"budget={budget_min:g} min (flag only), status={status_csv}",
          flush=True)

    for i, row in batch.iterrows():
        tic = int(row["TIC_ID"])
        sec = int(row["Sector"])
        outdir = str(row["outdir"])
        save_name = str(row.get("save_name", f"TIC_{tic}_Sector_{sec:04d}"))
        os.makedirs(outdir, exist_ok=True)
        stem = os.path.join(outdir, save_name)
        out_csv = f"{stem}_jaxoplanet_posterior.csv"
        label = f"[{i+1}/{len(batch)}] TIC {tic} S{sec:04d}"

        prior_status = classify_posterior(
            _read_posterior_row(out_csv),
            ceiling_max=ceiling_max,
            expect_likelihood=likelihood,
        )
        if (not overwrite) and prior_status == "OK":
            n_skip += 1
            if verbose:
                print(f"{label}: SKIP (posterior exists and passes)",
                      flush=True)
            continue

        reason = "overwrite" if overwrite else prior_status
        t0 = time.time()
        if reason == "NO_POSTERIOR":
            print(f"{label}: start", flush=True)
        else:
            print(f"{label}: start (refit, previous status {reason})",
                  flush=True)

        base: Dict[str, Any] = {
            "TIC_ID": tic,
            "Sector": sec,
            "save_name": save_name,
            "batch_csv": tag,
            "attempt_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "refit_reason": reason,
        }
        base.update(settings)

        try:
            summary, idata = refine_transit_parameters(
                lightcurve=str(row["lc_path"]),
                bestfit=str(row["bestfit_path"]),
                outdir=outdir,
                save_name=save_name,
                u_fixed=None,
                ld_mode=ld_mode,
                prior_style=prior_style,
                r_hi_sigmas=r_hi_sigmas,
                rho_star_prior=rho_star_prior,
                rho_frac_err=rho_frac_err,
                likelihood=likelihood,
                student_t_nu=float(student_t_nu),
                point_estimate=point_estimate,
                n_warmup=n_warmup,
                n_samples=n_samples,
                n_chains=n_chains,
                target_accept=target_accept,
                make_plots=True,
                make_timeseries=make_timeseries,
                make_trace=make_trace,
                make_corner=make_corner,
                n_pred_draws=n_pred_draws,
                show_plots=False,
                save_idata=False,   # the worker owns idata saving, see below
                verbose=verbose,
                **edge_kwargs,
            )

            got = str(summary["MCMC likelihood"].iloc[0]).strip().lower() \
                if "MCMC likelihood" in summary.columns else "absent"
            if got != likelihood:
                raise RuntimeError(
                    f"requested likelihood={likelihood!r} but the refit "
                    f"recorded {got!r}; posterior not written")

            idata_path = None
            if save_idata:
                idata_path = save_inference_data(idata, stem, thin=idata_thin,
                                                 verbose=verbose)

            # Record provenance in the same row the catalog will consume.
            summary = summary.copy()
            summary["batch_csv"] = tag
            summary["idata_path"] = idata_path if idata_path else ""
            summary["wall_seconds"] = float(time.time() - t0)
            summary.to_csv(out_csv, index=False)

            # Release the Figure handles the refit function attached.
            summary.attrs.pop("figures", None)
            import matplotlib.pyplot as plt
            plt.close("all")

            wall_min = (time.time() - t0) / 60.0
            rec = summary.iloc[0].to_dict()
            status = classify_posterior(rec, ceiling_max=ceiling_max)
            over = bool(budget_min > 0 and wall_min > budget_min)
            conv = bool(summary["MCMC converged"].iloc[0])

            n_done += 1
            if status == "UNCONVERGED":
                n_unconverged += 1
            elif status == "CEILING":
                n_ceiling += 1
            if over:
                n_over += 1

            over_txt = f", OVER BUDGET ({budget_min:g} min)" if over else ""
            print(f"{label}: OK in {wall_min:.1f} min, converged={conv}, "
                  f"status={status}{over_txt}", flush=True)

            out = dict(base)
            out.update(_diag_fields(rec))
            out.update({"status": status, "over_budget": over,
                        "wall_minutes": round(wall_min, 2), "error": ""})
            append_status_row(status_csv, out)

        except Exception as exc:
            n_fail += 1
            wall_min = (time.time() - t0) / 60.0
            msg = f"{type(exc).__name__}: {exc}"
            print(f"{label}: FAILED ({msg})", flush=True)
            traceback.print_exc()
            out = dict(base)
            out.update({
                "status": "FAILED",
                "over_budget": bool(budget_min > 0 and wall_min > budget_min),
                "wall_minutes": round(wall_min, 2),
                "error": msg[:500],
            })
            append_status_row(status_csv, out)

    elapsed_h = (time.time() - t_start) / 3600.0
    print(f"[worker] {tag} COMPLETE: Done={n_done} Skip={n_skip} "
          f"Fail={n_fail} Unconverged={n_unconverged} Ceiling={n_ceiling} "
          f"OverBudget={n_over} Elapsed={elapsed_h:.2f}h", flush=True)

def run_jaxoplanet_batch_OLD(
    batch_csv: str,
    *,
    n_chains: int = 4,
    n_warmup: int = 2000,
    n_samples: int = 3000,
    target_accept: float = 0.95,
    ld_mode: str = "gaussian",
    prior_style: str = "paper1",
    r_hi_sigmas: float = 10.0,
    rho_star_prior: bool = True,
    rho_frac_err: float = 0.15,
    likelihood: str = "student_t",
    student_t_nu: float = 4.0,
    point_estimate: str = "median",
    n_pred_draws: int = 300,
    make_corner: bool = True,
    make_trace: bool = True,
    make_timeseries: bool = True,
    save_idata: bool = True,
    idata_thin: int = 5,
    timeout_minutes: float = 360.0,
    ceiling_max: float = 0.01,
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """
    Drain one batch CSV, fitting every star-sector in it sequentially.

    One worker owns one batch and works through it in order, matching the
    disBatch architecture used by run_injection_batch.py: the unit of
    parallelism is the batch, not the target.

    Resume (R1). A target is skipped only when its posterior CSV exists AND
    'MCMC converged' is True AND 'MCMC r ceiling frac' <= ceiling_max AND the
    recorded likelihood matches the request. Missing, unconverged,
    ceiling-limited, unreadable, or wrong-likelihood posteriors are refit and
    overwritten. overwrite=True refits everything.

    Wall-time budget. The previous SIGALRM timeout could not interrupt NUTS,
    because sampling runs inside one compiled XLA call and Python only acts
    on the signal after it returns. It therefore discarded completed fits
    without saving any time. It is replaced by a flag: every finished fit is
    saved, and any fit longer than timeout_minutes is marked
    over_budget=True in the status CSV. A truly runaway fit runs until the
    SLURM walltime, as it effectively did before.

    Status table. Every attempt appends one row to
    '<batch_csv stem>_status.csv': status, wall time, diagnostics, settings,
    and error text. See summarize_jaxoplanet_status for the combined view.

    The observation likelihood is forwarded explicitly. refine_transit_parameters
    defaults to 'normal'; this worker defaults to 'student_t' with nu = 4.0,
    the adopted survey setting, so a batch cannot silently fall back to the
    Gaussian model. Two pre-flight checks run before any fit: an unknown
    likelihood name, and a deployed refit module that does not accept the
    likelihood keyword, both stop the worker immediately rather than failing
    every target one by one. After each fit the recorded 'MCMC likelihood'
    is compared with the request.

    Parameters
    ----------
    batch_csv : str
        Batch CSV from generate_jaxoplanet_manifest.py.
    n_chains, n_warmup, n_samples, target_accept : int, int, int, float
        NUTS settings passed through to refine_transit_parameters.
    ld_mode, prior_style, rho_star_prior, rho_frac_err : ...
        Model configuration passed through unchanged.
    r_hi_sigmas : float
        Uniform Rp/Rs prior ceiling in units of sigma_RP_RS above the search
        value. Default 10.0 (was 5.0).
    likelihood : str
        'student_t' (survey default) or 'normal'.
    student_t_nu : float
        Student-t degrees of freedom. Must exceed 2. Ignored for 'normal'.
    point_estimate : str
        'median' or 'maxl'. Affects figures only, never the table.
    n_pred_draws : int
        Credible-band draws in the phase-folded figure.
    make_corner, make_trace, make_timeseries, save_idata : bool
        Output switches.
    idata_thin : int
        Posterior thinning factor before writing.
    timeout_minutes : float
        Per-target wall-time budget for the over_budget flag only. 0 disables
        the flag. Default 360.
    ceiling_max : float
        Largest acceptable 'MCMC r ceiling frac' for a fit to count as done.
    overwrite : bool
        Refit every target regardless of existing posteriors.
    verbose : bool
        Per-target progress.

    Raises
    ------
    SystemExit
        If the batch CSV is unreadable or missing a required column, if the
        likelihood settings are invalid, or if the deployed refit module does
        not accept the likelihood keyword.

    Example
    -------
    >>> run_jaxoplanet_batch(                                  # doctest: +SKIP
    ...     "/mnt/home/dfeliz/ceph/NEMESIS_II/jaxoplanet/batches/"
    ...     "jaxoplanet_batch_001.csv",
    ...     likelihood="student_t", student_t_nu=4.0)
    """
    t_start = time.time()

    import inspect

    import numpy as np
    import pandas as pd

    likelihood = str(likelihood).strip().lower()
    if likelihood not in ("normal", "student_t"):
        raise SystemExit(
            f"likelihood must be 'normal' or 'student_t', got {likelihood!r}")
    if likelihood == "student_t" and not float(student_t_nu) > 2.0:
        raise SystemExit(
            f"student_t_nu must exceed 2 for a finite variance, "
            f"got {student_t_nu}")

    try:
        batch = pd.read_csv(batch_csv)
    except Exception as exc:
        raise SystemExit(f"cannot read batch CSV {batch_csv!r}: "
                         f"{type(exc).__name__}: {exc}")

    required = ("TIC_ID", "Sector", "lc_path", "bestfit_path", "outdir")
    missing = [c for c in required if c not in batch.columns]
    if missing:
        raise SystemExit(
            f"batch CSV {batch_csv!r} is missing {missing}. Regenerate it with "
            f"generate_jaxoplanet_manifest.py."
        )

    configure_runtime(n_chains, verbose=verbose)
    from nemesis_jaxoplanet_refit import refine_transit_parameters  # type: ignore

    refit_params = inspect.signature(refine_transit_parameters).parameters
    if "likelihood" not in refit_params or "student_t_nu" not in refit_params:
        import nemesis_jaxoplanet_refit as _refit_mod  # type: ignore
        raise SystemExit(
            f"the deployed nemesis_jaxoplanet_refit.py at "
            f"{_refit_mod.__file__} does not accept likelihood/student_t_nu. "
            f"It is an older copy; deploy the current module before running."
        )

    budget_min = (float(timeout_minutes)
                  if timeout_minutes and timeout_minutes > 0 else 0.0)
    status_csv = os.path.splitext(batch_csv)[0] + "_status.csv"
    settings = {
        "r_hi_sigmas": float(r_hi_sigmas),
        "n_warmup": int(n_warmup),
        "n_samples": int(n_samples),
        "target_accept": float(target_accept),
        "likelihood": likelihood,
        "student_t_nu": (float(student_t_nu)
                         if likelihood == "student_t" else np.nan),
        "timeout_minutes": budget_min,
    }

    n_done = n_skip = n_fail = n_unconverged = n_ceiling = n_over = 0
    tag = os.path.basename(batch_csv)
    nu_txt = f", nu={float(student_t_nu):g}" if likelihood == "student_t" else ""
    print(f"[worker] {tag}: {len(batch)} targets, n_chains={n_chains}, "
          f"warmup={n_warmup}, samples={n_samples}, "
          f"likelihood={likelihood}{nu_txt}, "
          f"rho_star_prior={rho_star_prior}, "
          f"r_hi_sigmas={float(r_hi_sigmas):g}, "
          f"ceiling_max={float(ceiling_max):g}, "
          f"budget={budget_min:g} min (flag only), status={status_csv}",
          flush=True)

    for i, row in batch.iterrows():
        tic = int(row["TIC_ID"])
        sec = int(row["Sector"])
        outdir = str(row["outdir"])
        save_name = str(row.get("save_name", f"TIC_{tic}_Sector_{sec:04d}"))
        os.makedirs(outdir, exist_ok=True)
        stem = os.path.join(outdir, save_name)
        out_csv = f"{stem}_jaxoplanet_posterior.csv"
        label = f"[{i+1}/{len(batch)}] TIC {tic} S{sec:04d}"

        prior_status = classify_posterior(
            _read_posterior_row(out_csv),
            ceiling_max=ceiling_max,
            expect_likelihood=likelihood,
        )
        if (not overwrite) and prior_status == "OK":
            n_skip += 1
            if verbose:
                print(f"{label}: SKIP (posterior exists and passes)",
                      flush=True)
            continue

        reason = "overwrite" if overwrite else prior_status
        t0 = time.time()
        if reason == "NO_POSTERIOR":
            print(f"{label}: start", flush=True)
        else:
            print(f"{label}: start (refit, previous status {reason})",
                  flush=True)

        base: Dict[str, Any] = {
            "TIC_ID": tic,
            "Sector": sec,
            "save_name": save_name,
            "batch_csv": tag,
            "attempt_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "refit_reason": reason,
        }
        base.update(settings)

        try:
            summary, idata = refine_transit_parameters(
                lightcurve=str(row["lc_path"]),
                bestfit=str(row["bestfit_path"]),
                outdir=outdir,
                save_name=save_name,
                u_fixed=None,
                ld_mode=ld_mode,
                prior_style=prior_style,
                r_hi_sigmas=r_hi_sigmas,
                rho_star_prior=rho_star_prior,
                rho_frac_err=rho_frac_err,
                likelihood=likelihood,
                student_t_nu=float(student_t_nu),
                point_estimate=point_estimate,
                n_warmup=n_warmup,
                n_samples=n_samples,
                n_chains=n_chains,
                target_accept=target_accept,
                make_plots=True,
                make_timeseries=make_timeseries,
                make_trace=make_trace,
                make_corner=make_corner,
                n_pred_draws=n_pred_draws,
                show_plots=False,
                save_idata=False,   # the worker owns idata saving, see below
                verbose=verbose,
            )

            got = str(summary["MCMC likelihood"].iloc[0]).strip().lower() \
                if "MCMC likelihood" in summary.columns else "absent"
            if got != likelihood:
                raise RuntimeError(
                    f"requested likelihood={likelihood!r} but the refit "
                    f"recorded {got!r}; posterior not written")

            idata_path = None
            if save_idata:
                idata_path = save_inference_data(idata, stem, thin=idata_thin,
                                                 verbose=verbose)

            # Record provenance in the same row the catalog will consume.
            summary = summary.copy()
            summary["batch_csv"] = tag
            summary["idata_path"] = idata_path if idata_path else ""
            summary["wall_seconds"] = float(time.time() - t0)
            summary.to_csv(out_csv, index=False)

            # Release the Figure handles the refit function attached.
            summary.attrs.pop("figures", None)
            import matplotlib.pyplot as plt
            plt.close("all")

            wall_min = (time.time() - t0) / 60.0
            rec = summary.iloc[0].to_dict()
            status = classify_posterior(rec, ceiling_max=ceiling_max)
            over = bool(budget_min > 0 and wall_min > budget_min)
            conv = bool(summary["MCMC converged"].iloc[0])

            n_done += 1
            if status == "UNCONVERGED":
                n_unconverged += 1
            elif status == "CEILING":
                n_ceiling += 1
            if over:
                n_over += 1

            over_txt = f", OVER BUDGET ({budget_min:g} min)" if over else ""
            print(f"{label}: OK in {wall_min:.1f} min, converged={conv}, "
                  f"status={status}{over_txt}", flush=True)

            out = dict(base)
            out.update(_diag_fields(rec))
            out.update({"status": status, "over_budget": over,
                        "wall_minutes": round(wall_min, 2), "error": ""})
            append_status_row(status_csv, out)

        except Exception as exc:
            n_fail += 1
            wall_min = (time.time() - t0) / 60.0
            msg = f"{type(exc).__name__}: {exc}"
            print(f"{label}: FAILED ({msg})", flush=True)
            traceback.print_exc()
            out = dict(base)
            out.update({
                "status": "FAILED",
                "over_budget": bool(budget_min > 0 and wall_min > budget_min),
                "wall_minutes": round(wall_min, 2),
                "error": msg[:500],
            })
            append_status_row(status_csv, out)

    elapsed_h = (time.time() - t_start) / 3600.0
    print(f"[worker] {tag} COMPLETE: Done={n_done} Skip={n_skip} "
          f"Fail={n_fail} Unconverged={n_unconverged} Ceiling={n_ceiling} "
          f"OverBudget={n_over} Elapsed={elapsed_h:.2f}h", flush=True)

def run_jaxoplanet_batch_OLDER(
    batch_csv: str,
    *,
    n_chains: int = 4,
    n_warmup: int = 2000,
    n_samples: int = 3000,
    target_accept: float = 0.95,
    ld_mode: str = "gaussian",
    prior_style: str = "paper1",
    r_hi_sigmas: float = 5.0,
    rho_star_prior: bool = True,
    rho_frac_err: float = 0.15,
    likelihood: str = "student_t",
    student_t_nu: float = 4.0,
    point_estimate: str = "median",
    n_pred_draws: int = 300,
    make_corner: bool = True,
    make_trace: bool = True,
    make_timeseries: bool = True,
    save_idata: bool = True,
    idata_thin: int = 5,
    timeout_minutes: float = 120.0,
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """
    Drain one batch CSV, fitting every star-sector in it sequentially.

    One worker owns one batch and works through it in order, matching the
    disBatch architecture used by run_injection_batch.py: the unit of
    parallelism is the batch, not the target.

    Each target gets a SIGALRM hard timeout. A pathological posterior can make
    NUTS run effectively forever, and without a per-target ceiling one target
    can consume the whole 48-hour allocation.

    The observation likelihood is forwarded explicitly. refine_transit_parameters
    defaults to 'normal'; this worker defaults to 'student_t' with nu = 4.0,
    the adopted survey setting, so a batch cannot silently fall back to the
    Gaussian model. Two pre-flight checks run before any fit: an unknown
    likelihood name, and a deployed refit module that does not accept the
    likelihood keyword, both stop the worker immediately rather than failing
    every target one by one. After each fit the recorded 'MCMC likelihood'
    is compared with the request.

    Parameters
    ----------
    batch_csv : str
        Batch CSV from generate_jaxoplanet_manifest.py.
    n_chains, n_warmup, n_samples, target_accept : int, int, int, float
        NUTS settings passed through to refine_transit_parameters.
    ld_mode, prior_style, r_hi_sigmas, rho_star_prior, rho_frac_err : ...
        Model configuration passed through unchanged.
    likelihood : str
        'student_t' (survey default) or 'normal'.
    student_t_nu : float
        Student-t degrees of freedom. Must exceed 2. Ignored for 'normal'.
    point_estimate : str
        'median' or 'maxl'. Affects figures only, never the table.
    n_pred_draws : int
        Credible-band draws in the phase-folded figure.
    make_corner, make_trace, make_timeseries, save_idata : bool
        Output switches.
    idata_thin : int
        Posterior thinning factor before writing.
    timeout_minutes : float
        Per-target hard ceiling. 0 disables.
    overwrite : bool
        Refit targets whose posterior CSV already exists.
    verbose : bool
        Per-target progress.

    Raises
    ------
    SystemExit
        If the batch CSV is unreadable or missing a required column, if the
        likelihood settings are invalid, or if the deployed refit module does
        not accept the likelihood keyword.

    Example
    -------
    >>> run_jaxoplanet_batch(                                  # doctest: +SKIP
    ...     "/mnt/home/dfeliz/ceph/NEMESIS_II/jaxoplanet/batches/"
    ...     "jaxoplanet_batch_001.csv",
    ...     likelihood="student_t", student_t_nu=4.0)
    """
    t_start = time.time()

    import inspect

    import numpy as np
    import pandas as pd

    likelihood = str(likelihood).strip().lower()
    if likelihood not in ("normal", "student_t"):
        raise SystemExit(
            f"likelihood must be 'normal' or 'student_t', got {likelihood!r}")
    if likelihood == "student_t" and not float(student_t_nu) > 2.0:
        raise SystemExit(
            f"student_t_nu must exceed 2 for a finite variance, "
            f"got {student_t_nu}")

    try:
        batch = pd.read_csv(batch_csv)
    except Exception as exc:
        raise SystemExit(f"cannot read batch CSV {batch_csv!r}: "
                         f"{type(exc).__name__}: {exc}")

    required = ("TIC_ID", "Sector", "lc_path", "bestfit_path", "outdir")
    missing = [c for c in required if c not in batch.columns]
    if missing:
        raise SystemExit(
            f"batch CSV {batch_csv!r} is missing {missing}. Regenerate it with "
            f"generate_jaxoplanet_manifest.py."
        )

    configure_runtime(n_chains, verbose=verbose)
    from nemesis_jaxoplanet_refit import refine_transit_parameters  # type: ignore

    refit_params = inspect.signature(refine_transit_parameters).parameters
    if "likelihood" not in refit_params or "student_t_nu" not in refit_params:
        import nemesis_jaxoplanet_refit as _refit_mod  # type: ignore
        raise SystemExit(
            f"the deployed nemesis_jaxoplanet_refit.py at "
            f"{_refit_mod.__file__} does not accept likelihood/student_t_nu. "
            f"It is an older copy; deploy the current module before running."
        )

    if timeout_minutes and timeout_minutes > 0:
        signal.signal(signal.SIGALRM, _alarm_handler)
    timeout_s = int(max(0, timeout_minutes * 60))

    n_done = n_skip = n_fail = n_unconverged = 0
    tag = os.path.basename(batch_csv)
    nu_txt = f", nu={float(student_t_nu):g}" if likelihood == "student_t" else ""
    print(f"[worker] {tag}: {len(batch)} targets, n_chains={n_chains}, "
          f"warmup={n_warmup}, samples={n_samples}, "
          f"likelihood={likelihood}{nu_txt}, "
          f"rho_star_prior={rho_star_prior}, timeout={timeout_minutes} min",
          flush=True)

    for i, row in batch.iterrows():
        tic = int(row["TIC_ID"])
        sec = int(row["Sector"])
        outdir = str(row["outdir"])
        save_name = str(row.get("save_name", f"TIC_{tic}_Sector_{sec:04d}"))
        os.makedirs(outdir, exist_ok=True)
        stem = os.path.join(outdir, save_name)
        out_csv = f"{stem}_jaxoplanet_posterior.csv"

        if (not overwrite) and os.path.isfile(out_csv):
            n_skip += 1
            if verbose:
                print(f"[{i+1}/{len(batch)}] TIC {tic} S{sec:04d}: SKIP "
                      f"(posterior exists)", flush=True)
            continue

        t0 = time.time()
        print(f"[{i+1}/{len(batch)}] TIC {tic} S{sec:04d}: start", flush=True)

        if timeout_s:
            signal.alarm(timeout_s)
        try:
            summary, idata = refine_transit_parameters(
                lightcurve=str(row["lc_path"]),
                bestfit=str(row["bestfit_path"]),
                outdir=outdir,
                save_name=save_name,
                u_fixed=None,
                ld_mode=ld_mode,
                prior_style=prior_style,
                r_hi_sigmas=r_hi_sigmas,
                rho_star_prior=rho_star_prior,
                rho_frac_err=rho_frac_err,
                likelihood=likelihood,
                student_t_nu=float(student_t_nu),
                point_estimate=point_estimate,
                n_warmup=n_warmup,
                n_samples=n_samples,
                n_chains=n_chains,
                target_accept=target_accept,
                make_plots=True,
                make_timeseries=make_timeseries,
                make_trace=make_trace,
                make_corner=make_corner,
                n_pred_draws=n_pred_draws,
                show_plots=False,
                save_idata=False,   # the worker owns idata saving, see below
                verbose=verbose,
            )

            got = str(summary["MCMC likelihood"].iloc[0]).strip().lower() \
                if "MCMC likelihood" in summary.columns else "absent"
            if got != likelihood:
                raise RuntimeError(
                    f"requested likelihood={likelihood!r} but the refit "
                    f"recorded {got!r}; posterior not written")

            idata_path = None
            if save_idata:
                idata_path = save_inference_data(idata, stem, thin=idata_thin,
                                                 verbose=verbose)

            # Record provenance in the same row the catalog will consume.
            summary = summary.copy()
            summary["batch_csv"] = tag
            summary["idata_path"] = idata_path if idata_path else ""
            summary["wall_seconds"] = float(time.time() - t0)
            summary.to_csv(out_csv, index=False)

            # Release the Figure handles the refit function attached.
            summary.attrs.pop("figures", None)
            import matplotlib.pyplot as plt
            plt.close("all")

            conv = bool(summary["MCMC converged"].iloc[0])
            if not conv:
                n_unconverged += 1
            n_done += 1
            print(f"[{i+1}/{len(batch)}] TIC {tic} S{sec:04d}: OK in "
                  f"{(time.time()-t0)/60:.1f} min, converged={conv}",
                  flush=True)

        except TargetTimeout:
            n_fail += 1
            print(f"[{i+1}/{len(batch)}] TIC {tic} S{sec:04d}: TIMEOUT after "
                  f"{timeout_minutes} min. Moving on.", flush=True)
        except Exception as exc:
            n_fail += 1
            print(f"[{i+1}/{len(batch)}] TIC {tic} S{sec:04d}: FAILED "
                  f"({type(exc).__name__}: {exc})", flush=True)
            traceback.print_exc()
        finally:
            if timeout_s:
                signal.alarm(0)

    elapsed_h = (time.time() - t_start) / 3600.0
    print(f"[worker] {tag} COMPLETE: Done={n_done} Skip={n_skip} "
          f"Fail={n_fail} Unconverged={n_unconverged} "
          f"Elapsed={elapsed_h:.2f}h", flush=True)

def main() -> None:
    """
    CLI driver. Fits one batch, or with --summarize builds the combined
    status table without importing JAX.

    Example
    -------
    python run_jaxoplanet_batch.py jaxoplanet_batch_001.csv \\
        --n_chains 4 --likelihood student_t --student_t_nu 4.0 --r_hi_abs 1.0

    python run_jaxoplanet_batch.py --summarize \\
        --batch_dir /mnt/home/dfeliz/ceph/NEMESIS_II/jaxoplanet/batches \\
        --nbatches 16
    """
    ap = argparse.ArgumentParser(description="Run one jaxoplanet refit batch.")
    ap.add_argument("batch_csv", nargs="?", default=None)
    ap.add_argument("--n_chains", type=int, default=4)
    ap.add_argument("--n_warmup", type=int, default=2000)
    ap.add_argument("--n_samples", type=int, default=3000)
    ap.add_argument("--target_accept", type=float, default=0.95)
    ap.add_argument("--ld_mode", default="gaussian",
                    choices=["fixed", "gaussian", "free"])
    ap.add_argument("--prior_style", default="paper1",
                    choices=["paper1", "lognormal"])
    ap.add_argument("--r_hi_sigmas", type=float, default=10.0)
    ap.add_argument("--r_hi_abs", type=float, default=None,
                    help="fixed Rp/Rs prior ceiling for every target, e.g. 1.0; "
                         "overrides --r_hi_sigmas")
    ap.add_argument("--no_rho_prior", action="store_true", default=False)
    ap.add_argument("--rho_frac_err", type=float, default=0.15)
    ap.add_argument("--likelihood", default="student_t",
                    choices=["normal", "student_t"])
    ap.add_argument("--student_t_nu", type=float, default=4.0)
    ap.add_argument("--point_estimate", default="median",
                    choices=["median", "maxl"])
    ap.add_argument("--n_pred_draws", type=int, default=300)
    ap.add_argument("--no_corner", action="store_true", default=False)
    ap.add_argument("--no_trace", action="store_true", default=False)
    ap.add_argument("--no_timeseries", action="store_true", default=False)
    ap.add_argument("--no_idata", action="store_true", default=False)
    ap.add_argument("--idata_thin", type=int, default=5)
    ap.add_argument("--timeout_minutes", type=float, default=360.0,
                    help="wall-time budget for the over_budget flag only")
    ap.add_argument("--ceiling_max", type=float, default=0.01,
                    help="max 'MCMC r ceiling frac' for a fit to count as done")
    ap.add_argument("--overwrite", action="store_true", default=False)
    ap.add_argument("--quiet", action="store_true", default=False)
    ap.add_argument("--summarize", action="store_true", default=False,
                    help="build the combined status table and exit")
    ap.add_argument("--batch_dir", default=None,
                    help="batch CSV directory, required with --summarize")
    ap.add_argument("--nbatches", type=int, default=16)
    ap.add_argument("--summary_csv", default=None,
                    help="output path for --summarize")
    args = ap.parse_args()

    if args.summarize:
        if not args.batch_dir:
            ap.error("--summarize requires --batch_dir")
        summarize_jaxoplanet_status(
            args.batch_dir,
            args.nbatches,
            ceiling_max=args.ceiling_max,
            out_csv=args.summary_csv,
            verbose=not args.quiet,
        )
        return

    if args.batch_csv is None:
        ap.error("batch_csv is required unless --summarize is given")

    run_jaxoplanet_batch(
        args.batch_csv,
        n_chains=args.n_chains,
        n_warmup=args.n_warmup,
        n_samples=args.n_samples,
        target_accept=args.target_accept,
        ld_mode=args.ld_mode,
        prior_style=args.prior_style,
        r_hi_sigmas=args.r_hi_sigmas,
        r_hi_abs=args.r_hi_abs,
        rho_star_prior=not args.no_rho_prior,
        rho_frac_err=args.rho_frac_err,
        likelihood=args.likelihood,
        student_t_nu=args.student_t_nu,
        point_estimate=args.point_estimate,
        n_pred_draws=args.n_pred_draws,
        make_corner=not args.no_corner,
        make_trace=not args.no_trace,
        make_timeseries=not args.no_timeseries,
        save_idata=not args.no_idata,
        idata_thin=args.idata_thin,
        timeout_minutes=args.timeout_minutes,
        ceiling_max=args.ceiling_max,
        overwrite=args.overwrite,
        verbose=not args.quiet,
    )

def main_OLD() -> None:
    """
    CLI driver. Fits one batch, or with --summarize builds the combined
    status table without importing JAX.

    Example
    -------
    python run_jaxoplanet_batch.py jaxoplanet_batch_001.csv \\
        --n_chains 4 --likelihood student_t --student_t_nu 4.0

    python run_jaxoplanet_batch.py --summarize \\
        --batch_dir /mnt/home/dfeliz/ceph/NEMESIS_II/jaxoplanet/batches \\
        --nbatches 16
    """
    ap = argparse.ArgumentParser(description="Run one jaxoplanet refit batch.")
    ap.add_argument("batch_csv", nargs="?", default=None)
    ap.add_argument("--n_chains", type=int, default=4)
    ap.add_argument("--n_warmup", type=int, default=2000)
    ap.add_argument("--n_samples", type=int, default=3000)
    ap.add_argument("--target_accept", type=float, default=0.95)
    ap.add_argument("--ld_mode", default="gaussian",
                    choices=["fixed", "gaussian", "free"])
    ap.add_argument("--prior_style", default="paper1",
                    choices=["paper1", "lognormal"])
    ap.add_argument("--r_hi_sigmas", type=float, default=10.0)
    ap.add_argument("--no_rho_prior", action="store_true", default=False)
    ap.add_argument("--rho_frac_err", type=float, default=0.15)
    ap.add_argument("--likelihood", default="student_t",
                    choices=["normal", "student_t"])
    ap.add_argument("--student_t_nu", type=float, default=4.0)
    ap.add_argument("--point_estimate", default="median",
                    choices=["median", "maxl"])
    ap.add_argument("--n_pred_draws", type=int, default=300)
    ap.add_argument("--no_corner", action="store_true", default=False)
    ap.add_argument("--no_trace", action="store_true", default=False)
    ap.add_argument("--no_timeseries", action="store_true", default=False)
    ap.add_argument("--no_idata", action="store_true", default=False)
    ap.add_argument("--idata_thin", type=int, default=5)
    ap.add_argument("--timeout_minutes", type=float, default=360.0,
                    help="wall-time budget for the over_budget flag only")
    ap.add_argument("--ceiling_max", type=float, default=0.01,
                    help="max 'MCMC r ceiling frac' for a fit to count as done")
    ap.add_argument("--overwrite", action="store_true", default=False)
    ap.add_argument("--quiet", action="store_true", default=False)
    ap.add_argument("--summarize", action="store_true", default=False,
                    help="build the combined status table and exit")
    ap.add_argument("--batch_dir", default=None,
                    help="batch CSV directory, required with --summarize")
    ap.add_argument("--nbatches", type=int, default=16)
    ap.add_argument("--summary_csv", default=None,
                    help="output path for --summarize")
    args = ap.parse_args()

    if args.summarize:
        if not args.batch_dir:
            ap.error("--summarize requires --batch_dir")
        summarize_jaxoplanet_status(
            args.batch_dir,
            args.nbatches,
            ceiling_max=args.ceiling_max,
            out_csv=args.summary_csv,
            verbose=not args.quiet,
        )
        return

    if args.batch_csv is None:
        ap.error("batch_csv is required unless --summarize is given")

    run_jaxoplanet_batch(
        args.batch_csv,
        n_chains=args.n_chains,
        n_warmup=args.n_warmup,
        n_samples=args.n_samples,
        target_accept=args.target_accept,
        ld_mode=args.ld_mode,
        prior_style=args.prior_style,
        r_hi_sigmas=args.r_hi_sigmas,
        rho_star_prior=not args.no_rho_prior,
        rho_frac_err=args.rho_frac_err,
        likelihood=args.likelihood,
        student_t_nu=args.student_t_nu,
        point_estimate=args.point_estimate,
        n_pred_draws=args.n_pred_draws,
        make_corner=not args.no_corner,
        make_trace=not args.no_trace,
        make_timeseries=not args.no_timeseries,
        save_idata=not args.no_idata,
        idata_thin=args.idata_thin,
        timeout_minutes=args.timeout_minutes,
        ceiling_max=args.ceiling_max,
        overwrite=args.overwrite,
        verbose=not args.quiet,
    )

def main_OLDER() -> None:
    """
    CLI driver. One positional argument, the rest flags.

    Example
    -------
    python run_jaxoplanet_batch.py jaxoplanet_batch_001.csv \\
        --n_chains 4 --likelihood student_t --student_t_nu 4.0
    """
    ap = argparse.ArgumentParser(description="Run one jaxoplanet refit batch.")
    ap.add_argument("batch_csv")
    ap.add_argument("--n_chains", type=int, default=4)
    ap.add_argument("--n_warmup", type=int, default=2000)
    ap.add_argument("--n_samples", type=int, default=3000)
    ap.add_argument("--target_accept", type=float, default=0.95)
    ap.add_argument("--ld_mode", default="gaussian",
                    choices=["fixed", "gaussian", "free"])
    ap.add_argument("--prior_style", default="paper1",
                    choices=["paper1", "lognormal"])
    ap.add_argument("--r_hi_sigmas", type=float, default=5.0)
    ap.add_argument("--no_rho_prior", action="store_true", default=False)
    ap.add_argument("--rho_frac_err", type=float, default=0.15)
    ap.add_argument("--likelihood", default="student_t",
                    choices=["normal", "student_t"])
    ap.add_argument("--student_t_nu", type=float, default=4.0)
    ap.add_argument("--point_estimate", default="median",
                    choices=["median", "maxl"])
    ap.add_argument("--n_pred_draws", type=int, default=300)
    ap.add_argument("--no_corner", action="store_true", default=False)
    ap.add_argument("--no_trace", action="store_true", default=False)
    ap.add_argument("--no_timeseries", action="store_true", default=False)
    ap.add_argument("--no_idata", action="store_true", default=False)
    ap.add_argument("--idata_thin", type=int, default=5)
    ap.add_argument("--timeout_minutes", type=float, default=120.0)
    ap.add_argument("--overwrite", action="store_true", default=False)
    ap.add_argument("--quiet", action="store_true", default=False)
    args = ap.parse_args()

    run_jaxoplanet_batch(
        args.batch_csv,
        n_chains=args.n_chains,
        n_warmup=args.n_warmup,
        n_samples=args.n_samples,
        target_accept=args.target_accept,
        ld_mode=args.ld_mode,
        prior_style=args.prior_style,
        r_hi_sigmas=args.r_hi_sigmas,
        rho_star_prior=not args.no_rho_prior,
        rho_frac_err=args.rho_frac_err,
        likelihood=args.likelihood,
        student_t_nu=args.student_t_nu,
        point_estimate=args.point_estimate,
        n_pred_draws=args.n_pred_draws,
        make_corner=not args.no_corner,
        make_trace=not args.no_trace,
        make_timeseries=not args.no_timeseries,
        save_idata=not args.no_idata,
        idata_thin=args.idata_thin,
        timeout_minutes=args.timeout_minutes,
        overwrite=args.overwrite,
        verbose=not args.quiet,
    )

def run_jaxoplanet_batch_OLDEST(
    batch_csv: str,
    *,
    n_chains: int = 4,
    n_warmup: int = 2000,
    n_samples: int = 3000,
    target_accept: float = 0.95,
    ld_mode: str = "gaussian",
    prior_style: str = "paper1",
    r_hi_sigmas: float = 5.0,
    rho_star_prior: bool = True,
    rho_frac_err: float = 0.15,
    point_estimate: str = "median",
    n_pred_draws: int = 300,
    make_corner: bool = True,
    make_trace: bool = True,
    make_timeseries: bool = True,
    save_idata: bool = True,
    idata_thin: int = 5,
    timeout_minutes: float = 120.0,
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """
    Drain one batch CSV, fitting every star-sector in it sequentially.

    One worker owns one batch and works through it in order, matching the
    disBatch architecture used by run_injection_batch.py: the unit of
    parallelism is the batch, not the target.

    Each target gets a SIGALRM hard timeout. A pathological posterior can make
    NUTS run effectively forever, and without a per-target ceiling one target
    can consume the whole 48-hour allocation.

    Parameters
    ----------
    batch_csv : str
        Batch CSV from generate_jaxoplanet_manifest.py.
    n_chains, n_warmup, n_samples, target_accept : int, int, int, float
        NUTS settings passed through to refine_transit_parameters.
    ld_mode, prior_style, r_hi_sigmas, rho_star_prior, rho_frac_err : ...
        Model configuration passed through unchanged.
    point_estimate : str
        'median' or 'maxl'. Affects figures only, never the table.
    n_pred_draws : int
        Credible-band draws in the phase-folded figure. Each draw is a separate
        eager model evaluation, so this is real wall time; 300 is a reasonable
        batch compromise against the 500 used interactively.
    make_corner, make_trace, make_timeseries, save_idata : bool
        Output switches.
    idata_thin : int
        Posterior thinning factor before writing.
    timeout_minutes : float
        Per-target hard ceiling. 0 disables.
    overwrite : bool
        Refit targets whose posterior CSV already exists.
    verbose : bool
        Per-target progress.

    Raises
    ------
    SystemExit
        If the batch CSV is unreadable or missing a required column.
    """
    t_start = time.time()

    import numpy as np
    import pandas as pd

    try:
        batch = pd.read_csv(batch_csv)
    except Exception as exc:
        raise SystemExit(f"cannot read batch CSV {batch_csv!r}: "
                         f"{type(exc).__name__}: {exc}")

    required = ("TIC_ID", "Sector", "lc_path", "bestfit_path", "outdir")
    missing = [c for c in required if c not in batch.columns]
    if missing:
        raise SystemExit(
            f"batch CSV {batch_csv!r} is missing {missing}. Regenerate it with "
            f"generate_jaxoplanet_manifest.py."
        )

    configure_runtime(n_chains, verbose=verbose)
    from nemesis_jaxoplanet_refit import refine_transit_parameters  # type: ignore

    if timeout_minutes and timeout_minutes > 0:
        signal.signal(signal.SIGALRM, _alarm_handler)
    timeout_s = int(max(0, timeout_minutes * 60))

    n_done = n_skip = n_fail = n_unconverged = 0
    tag = os.path.basename(batch_csv)
    print(f"[worker] {tag}: {len(batch)} targets, n_chains={n_chains}, "
          f"warmup={n_warmup}, samples={n_samples}, "
          f"rho_star_prior={rho_star_prior}, timeout={timeout_minutes} min",
          flush=True)

    for i, row in batch.iterrows():
        tic = int(row["TIC_ID"])
        sec = int(row["Sector"])
        outdir = str(row["outdir"])
        save_name = str(row.get("save_name", f"TIC_{tic}_Sector_{sec:04d}"))
        os.makedirs(outdir, exist_ok=True)
        stem = os.path.join(outdir, save_name)
        out_csv = f"{stem}_jaxoplanet_posterior.csv"

        if (not overwrite) and os.path.isfile(out_csv):
            n_skip += 1
            if verbose:
                print(f"[{i+1}/{len(batch)}] TIC {tic} S{sec:04d}: SKIP "
                      f"(posterior exists)", flush=True)
            continue

        t0 = time.time()
        print(f"[{i+1}/{len(batch)}] TIC {tic} S{sec:04d}: start", flush=True)

        if timeout_s:
            signal.alarm(timeout_s)
        try:
            summary, idata = refine_transit_parameters(
                lightcurve=str(row["lc_path"]),
                bestfit=str(row["bestfit_path"]),
                outdir=outdir,
                save_name=save_name,
                u_fixed=None,
                ld_mode=ld_mode,
                prior_style=prior_style,
                r_hi_sigmas=r_hi_sigmas,
                rho_star_prior=rho_star_prior,
                rho_frac_err=rho_frac_err,
                point_estimate=point_estimate,
                n_warmup=n_warmup,
                n_samples=n_samples,
                n_chains=n_chains,
                target_accept=target_accept,
                make_plots=True,
                make_timeseries=make_timeseries,
                make_trace=make_trace,
                make_corner=make_corner,
                n_pred_draws=n_pred_draws,
                show_plots=False,
                save_idata=False,   # the worker owns idata saving, see below
                verbose=verbose,
            )

            idata_path = None
            if save_idata:
                idata_path = save_inference_data(idata, stem, thin=idata_thin,
                                                 verbose=verbose)

            # Record provenance in the same row the catalog will consume.
            summary = summary.copy()
            summary["batch_csv"] = tag
            summary["idata_path"] = idata_path if idata_path else ""
            summary["wall_seconds"] = float(time.time() - t0)
            summary.to_csv(out_csv, index=False)

            # Release the Figure handles the refit function attached.
            summary.attrs.pop("figures", None)
            import matplotlib.pyplot as plt
            plt.close("all")

            conv = bool(summary["MCMC converged"].iloc[0])
            if not conv:
                n_unconverged += 1
            n_done += 1
            print(f"[{i+1}/{len(batch)}] TIC {tic} S{sec:04d}: OK in "
                  f"{(time.time()-t0)/60:.1f} min, converged={conv}",
                  flush=True)

        except TargetTimeout:
            n_fail += 1
            print(f"[{i+1}/{len(batch)}] TIC {tic} S{sec:04d}: TIMEOUT after "
                  f"{timeout_minutes} min. Moving on.", flush=True)
        except Exception as exc:
            n_fail += 1
            print(f"[{i+1}/{len(batch)}] TIC {tic} S{sec:04d}: FAILED "
                  f"({type(exc).__name__}: {exc})", flush=True)
            traceback.print_exc()
        finally:
            if timeout_s:
                signal.alarm(0)

    elapsed_h = (time.time() - t_start) / 3600.0
    print(f"[worker] {tag} COMPLETE: Done={n_done} Skip={n_skip} "
          f"Fail={n_fail} Unconverged={n_unconverged} "
          f"Elapsed={elapsed_h:.2f}h", flush=True)


def main_OLDEST() -> None:
    """CLI driver. One positional argument, the rest flags."""
    ap = argparse.ArgumentParser(description="Run one jaxoplanet refit batch.")
    ap.add_argument("batch_csv")
    ap.add_argument("--n_chains", type=int, default=4)
    ap.add_argument("--n_warmup", type=int, default=2000)
    ap.add_argument("--n_samples", type=int, default=3000)
    ap.add_argument("--target_accept", type=float, default=0.95)
    ap.add_argument("--ld_mode", default="gaussian",
                    choices=["fixed", "gaussian", "free"])
    ap.add_argument("--prior_style", default="paper1",
                    choices=["paper1", "lognormal"])
    ap.add_argument("--r_hi_sigmas", type=float, default=5.0)
    ap.add_argument("--no_rho_prior", action="store_true", default=False)
    ap.add_argument("--rho_frac_err", type=float, default=0.15)
    ap.add_argument("--point_estimate", default="median",
                    choices=["median", "maxl"])
    ap.add_argument("--n_pred_draws", type=int, default=300)
    ap.add_argument("--no_corner", action="store_true", default=False)
    ap.add_argument("--no_trace", action="store_true", default=False)
    ap.add_argument("--no_timeseries", action="store_true", default=False)
    ap.add_argument("--no_idata", action="store_true", default=False)
    ap.add_argument("--idata_thin", type=int, default=5)
    ap.add_argument("--timeout_minutes", type=float, default=120.0)
    ap.add_argument("--overwrite", action="store_true", default=False)
    ap.add_argument("--quiet", action="store_true", default=False)
    args = ap.parse_args()

    run_jaxoplanet_batch(
        args.batch_csv,
        n_chains=args.n_chains,
        n_warmup=args.n_warmup,
        n_samples=args.n_samples,
        target_accept=args.target_accept,
        ld_mode=args.ld_mode,
        prior_style=args.prior_style,
        r_hi_sigmas=args.r_hi_sigmas,
        rho_star_prior=not args.no_rho_prior,
        rho_frac_err=args.rho_frac_err,
        point_estimate=args.point_estimate,
        n_pred_draws=args.n_pred_draws,
        make_corner=not args.no_corner,
        make_trace=not args.no_trace,
        make_timeseries=not args.no_timeseries,
        save_idata=not args.no_idata,
        idata_thin=args.idata_thin,
        timeout_minutes=args.timeout_minutes,
        overwrite=args.overwrite,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
