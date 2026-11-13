"""
nemesis_jaxoplanet_refit.py

Transit parameter refinement for NEMESIS II planet candidates.

One public function, ``refine_transit_parameters``, takes a NEMESIS light curve
and the matching TLS (or BLS) ``*bestfit.csv`` and returns posterior-refined
transit parameters from a NumPyro NUTS fit of a jaxoplanet limb-darkened
transit model.

Dependencies: numpy, pandas, matplotlib, astropy, jax, numpyro, arviz,
jaxoplanet. ``corner`` is optional and only needed for ``make_corner=True``.
``numpyro_ext`` is deliberately NOT required.

Tested against the versions pinned in the May 2025 notebook:
jaxoplanet 0.0.4, numpyro 0.18.0, jax 0.4.31, arviz 0.21.0, numpy 1.26.4.
"""

from __future__ import annotations

import os
import re
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


def refine_transit_parameters(
    lightcurve: Union[str, pd.DataFrame],
    bestfit: Union[str, pd.DataFrame],
    *,
    outdir: str = ".",
    save_name: Optional[str] = None,
    method: str = "TLS",
    time_col: str = "Time",
    flux_col: str = "Detrended Flux",
    err_col: str = "Detrended Error",
    quality_col: Optional[str] = "QUALITY_FLAG",
    quality_accept: Sequence[int] = (0,),
    normalize: bool = True,
    ld_mode: str = "fixed",
    u_fixed: Optional[Sequence[float]] = None,
    ld_sigma: float = 0.10,
    prior_style: str = "paper1",
    t0_sigma_durations: float = 1.0,
    duration_log_sigma: float = 0.20,
    r_log_sigma: float = 0.50,
    r_hi_sigmas: float = 1.0,
    r_hi_abs: Optional[float] = None,    
    r_star_err: Optional[float] = None,
    m_star_err: Optional[float] = None,
    rho_star_prior: Optional[Union[bool, Tuple[float, float]]] = False,
    rho_frac_err: float = 0.15,
    rho_prefer: str = "auto",
    rho_unit_tol: float = 0.02,
    window_durations: Optional[float] = 6.0,
    exposure_days: Optional[float] = None,
    supersample: int = 7,
    fit_offset: bool = True,
    fit_jitter: bool = True,
    likelihood: str = "normal",
    student_t_nu: float = 4.0,    
    n_map_steps: int = 3000,
    n_warmup: int = 1500,
    n_samples: int = 2000,
    n_chains: int = 2,
    target_accept: float = 0.95,
    dense_mass: bool = True,
    seed: int = 0,
    make_plots: bool = True,
    make_corner: bool = False,
    make_trace: bool = False,
    make_timeseries: bool = True,
    trace_vars: Optional[Sequence[str]] = None,
    corner_vars: Optional[Sequence[str]] = None,
    phase_bin_minutes: float = 0.0,
    point_estimate: str = "median",
    show_plots: bool = False,
    plot_dpi: int = 200,
    n_pred_draws: int = 200,
    save_idata: bool = False,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Any]:
    """
    Refine transit parameters of a NEMESIS planet candidate with jaxoplanet.

    Replaces the multi-cell notebook workflow (``quad_limb_dark`` +
    ``_numpyro_model`` + ``light_curve_model`` + ``fit_transit`` +
    ``plot_posterior_transit``) with a single call. Reads priors directly from
    the pipeline's ``*bestfit.csv`` so no manual prior dictionary is needed.

    Model
    -----
    Flux is ``f0 * (1 + LC(t))`` where ``LC`` is the analytic quadratically
    limb-darkened light curve of Agol et al. (2020) evaluated through
    ``jaxoplanet.orbits.TransitOrbit``. Orbits are circular. The model is
    integrated over the exposure window by midpoint averaging of
    ``supersample`` subsamples (Kipping 2010).

    Free parameters
    ---------------
    ``t0``, ``log_period``, ``log_duration``, ``r`` (or ``log_r``),
    ``b_frac``, optionally ``f0``, ``log_jitter``, and ``u_q`` (a 2-vector)
    when limb darkening is not held fixed.

    ``TransitOrbit`` is parameterized by transit duration rather than by
    stellar density, so T14 is necessarily a free parameter. The minimum
    parameter count for this model is therefore five, not four.

    Parameters
    ----------
    lightcurve : str or pd.DataFrame
        Path to a NEMESIS ``*_final_LC.csv`` or an in-memory DataFrame.
    bestfit : str or pd.DataFrame
        Path to ``TIC_{ID}_{sector_tag}_TLS_bestfit.csv`` or a single-row
        DataFrame with the same schema.
    outdir : str
        Directory for output CSV, plots, and optional NetCDF. Created if absent.
    save_name : str or None
        Output basename. Parsed from the bestfit filename if None.
    method : str
        ``'TLS'`` or ``'BLS'``. Selects which bestfit columns supply the priors.
    time_col, flux_col, err_col : str
        Light curve column names.
    quality_col : str or None
        Internal quality bitmask column. Set to None to skip quality filtering.
    quality_accept : sequence of int
        QUALITY_FLAG values retained. Default ``(0,)``, matching the survey
        transit search.
    normalize : bool
        Divide flux and error by the median flux when the median differs from
        unity by more than 5 percent. Lets the function accept SAP Flux.
    ld_mode : str
        ``'fixed'``   : u held at ``u_fixed``, no LD parameters sampled.
        ``'gaussian'``: q1, q2 given TruncatedNormal priors centered on the
                        Kipping transform of ``u_fixed``, width ``ld_sigma``.
        ``'free'``    : q1, q2 uniform on (0, 1), the Kipping (2013)
                        uninformative prior used in Paper I.
    u_fixed : sequence of 2 floats or None
        Quadratic limb-darkening coefficients (u1, u2). Required for
        ``'fixed'`` and ``'gaussian'``. Falls back to ``qld_a`` / ``qld_b``
        columns in the bestfit row when present.
    ld_sigma : float
        Prior width in q-space for ``ld_mode='gaussian'``.
    prior_style : str
        ``'paper1'``   : r ~ Uniform(0.01, RP_RS + sigma_RP_RS),
                         b ~ Uniform(0.01, 1 + r), matching Feliz et al. 2021
                         Table 2.
        ``'lognormal'``: log r ~ Normal(ln RP_RS, ``r_log_sigma``),
                         b ~ Uniform(0, 1 + r).
    t0_sigma_durations : float
        T0 prior width in units of the TLS transit duration. Paper I used 1.0.
    duration_log_sigma : float
        Prior width on ln(T14). TLS reports no duration uncertainty.
    r_log_sigma : float
        Prior width on ln(Rp/Rs) when ``prior_style='lognormal'``.
    r_hi_sigmas : float
        Upper edge of the Uniform radius prior, in units of sigma_RP_RS above
        the search value: r_hi = RP_RS + r_hi_sigmas * sigma_RP_RS. 1.0
        reproduces Feliz et al. 2021 Table 2 exactly. Use 5.0 for publication
        so the ceiling never truncates the posterior; check
        'MCMC r ceiling frac' in the output to confirm.       
    r_hi_abs : float or None
        Fixed upper edge of the Uniform radius prior, the same for every
        target. When given it replaces RP_RS + r_hi_sigmas * sigma_RP_RS and
        r_hi_sigmas is ignored. 1.0 caps the companion at the size of the
        star: every planet lies far below it, so the edge binds only when the
        dip is deeper than the target can produce. Used only with
        prior_style='paper1'.
    r_star_err : float or None
        1-sigma stellar radius uncertainty in solar radii. When supplied it is
        propagated into Rp, a, insolation, and T_eq by Monte Carlo. When None,
        those quantities carry only the Rp/Rs posterior width.
    m_star_err : float or None
        1-sigma stellar mass uncertainty in solar masses, propagated into
        a, insolation and T_eq. When None those carry the point mass value.
    rho_star_prior : bool or (float, float) or False
        Constrain the stellar mean density, which breaks the b and a/Rs
        degeneracy and converts b from unmeasured into T14-determined. False
        leaves b prior-dominated at 30 minute cadence. True reads
        'rho [g/ccm]' from the bestfit row, falling back to
        1.41 * M_star / R_star**3, with a fractional width of rho_frac_err.
        A (mu, sigma) tuple in g/cm^3 overrides both.
    rho_frac_err : float
        Fractional 1-sigma width used when rho_star_prior is True. Ignored
        when an explicit (mu, sigma) tuple is given.
    rho_prefer : str
        Which source supplies the stellar mean density.
        'auto' (default) walks, in order: M_star/R_star**3, the
        'rho [rho_sun]' column, the 'rho [g/ccm]' column with its unit
        sniffed, then logg with R_star. M_star/R_star**3 leads because the
        reported a [AU] comes from the same M_star through Kepler's third law,
        so taking the density from elsewhere can leave the prior and the
        reported semi-major axis describing different stars.
        'column' puts 'rho [rho_sun]' first, then 'rho [g/ccm]'.
        'mr' uses M_star/R_star**3 only. 'logg' uses logg and R_star only.
        Cross-checks run under every setting, so a disagreement between
        sources is reported even when it does not change the value used.
    rho_unit_tol : float
        Fractional tolerance for deciding whether 'rho [g/ccm]' holds solar
        units or genuine cgs. 0.02 is comfortably tighter than the 1.4098
        factor separating the two cases.        
    window_durations : float or None
        Keep only cadences within this many transit durations of a predicted
        mid-transit. None keeps the full light curve.
    exposure_days : float or None
        Exposure time in days. Median cadence spacing when None.
    supersample : int
        Subsamples per exposure for the integration. 1 disables it.
    fit_offset : bool
        Fit a multiplicative baseline ``f0``.
    fit_jitter : bool
        Fit a white-noise term added in quadrature to the reported errors.
    likelihood : str
        'normal' (default) reproduces the current model. 'student_t' swaps a
        Student-t observation likelihood with ``student_t_nu`` degrees of
        freedom, which down-weights individual outlying cadences instead of
        inflating every error bar to accommodate them. Use it when the fitted
        jitter is driven by a few cadences rather than by uniformly elevated
        scatter: compare the MAD and the standard deviation of the windowed
        out-of-transit residuals to tell which.
    student_t_nu : float
        Degrees of freedom, fixed rather than sampled for reproducibility.
        4.0 is heavy-tailed enough to absorb a few percent of outliers while
        staying close to Gaussian in the core. Note that for Student-t the
        scale is not the standard deviation: the variance is
        scale^2 * nu/(nu-2), so 2*scale^2 at nu = 4.        
    n_map_steps : int
        Adam steps for the AutoDelta MAP initialization.
    n_warmup, n_samples, n_chains : int
        NUTS settings.
    target_accept : float
        NUTS target acceptance probability.
    dense_mass : bool
        Use a dense mass matrix. Recommended given the b / r / T14 degeneracy.
    seed : int
        PRNG seed.
    make_plots : bool
        Write a phase-folded fit plus residual figure.
    make_trace : bool
        Write an arviz trace plot. Forced on whenever the convergence gate
        fails, regardless of this setting.
    make_timeseries : bool
        Write the un-folded time-series overlay. Gated by ``make_plots``.
    trace_vars : sequence of str or None
        Variables to trace. Defaults to every fitted parameter. Names absent
        from the posterior, or constant within it, are dropped silently, so
        the same list works for ld_mode 'fixed', 'gaussian' and 'free'.
    corner_vars : sequence of str or None
        Subset of corner panel labels to keep. None keeps all of
        'P [d]', 'T0 [BTJD]', 'T14 [hr]', 'Rp/Rs', 'b', 'a/Rs', 'u1', 'u2'.
    phase_bin_minutes : float
        Bin width for the binned points on the phase-folded panel. 0 disables.
    point_estimate : str
        Which parameter vector draws the model curve in the figures.
        'median' (default) uses the per-parameter posterior median, which is
        what the table reports but is not itself a posterior sample.
        'maxl' uses the single highest-likelihood draw, which is a real
        solution. The reported table values are unaffected either way.        
    show_plots : bool
        Call plt.show() on each figure. Set True in a notebook, leave False
        for batch runs over the candidate list.
    plot_dpi : int
        Raster resolution for saved figures.
    n_pred_draws : int
        Posterior draws used for the credible band in the fit figure.
    save_idata : bool
        Write the InferenceData to NetCDF. Roughly 10 to 50 MB per target.
    verbose : bool
        Print progress and diagnostics.

    Returns
    -------
    summary : pd.DataFrame
        Single row, wide format, suitable for concatenation into the candidate
        catalog. Every fitted and derived quantity appears as
        ``MCMC {name}``, ``MCMC {name} Err Lo``, ``MCMC {name} Err Hi``
        (median, median minus 16th, 84th minus median). Also carries
        ``MCMC rhat max``, ``MCMC ess bulk min``, ``MCMC n divergences``,
        and ``MCMC converged``.
    idata : arviz.InferenceData
        Full posterior.
    Figure handles are attached as ``summary.attrs['figures']``, keyed by
        'fit', 'timeseries', 'trace', 'corner'. Saved paths also appear as
        ``Fig {tag}`` columns.    

    Raises
    ------
    KeyError
        A required bestfit or light curve column is absent.
    ValueError
        Bad ``ld_mode`` / ``prior_style`` / ``method``, missing ``u_fixed``
        in a mode that needs it, or fewer than 20 usable cadences.

    Examples
    --------
    >>> summ, idata = refine_transit_parameters(   # doctest: +SKIP
    ...     lightcurve='/Users/daxfeliz/Desktop/TESS/NEMESIS_II/'
    ...                'TIC_259377017_Sector_5_final_LC.csv',
    ...     bestfit='/Users/daxfeliz/Desktop/TESS/NEMESIS_II/'
    ...             'TIC_259377017_Sector_5_TLS_bestfit.csv',
    ...     outdir='./jaxoplanet_fits/',
    ...     u_fixed=(0.1604, 0.4325),
    ...     n_warmup=1500, n_samples=2000, n_chains=2)
    >>> float(summ['MCMC Period [d]'].iloc[0])     # doctest: +SKIP
    5.6587...
    """
    # ------------------------------------------------------------------ setup
    import jax

    jax.config.update("jax_enable_x64", True)

    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    import arviz as az
    import matplotlib.pyplot as plt
    from astropy import constants as apc
    from astropy import units as apu
    from jaxoplanet.orbits import TransitOrbit
    from jaxoplanet.light_curves import limb_dark_light_curve

    method = str(method).strip().upper()
    if method not in ("TLS", "BLS"):
        raise ValueError(f"method must be 'TLS' or 'BLS', got {method!r}")

    ld_mode = str(ld_mode).strip().lower()
    if ld_mode not in ("fixed", "gaussian", "free"):
        raise ValueError(
            f"ld_mode must be 'fixed', 'gaussian' or 'free', got {ld_mode!r}"
        )

    prior_style = str(prior_style).strip().lower()
    if prior_style not in ("paper1", "lognormal"):
        raise ValueError(
            f"prior_style must be 'paper1' or 'lognormal', got {prior_style!r}"
        )
    point_estimate = str(point_estimate).strip().lower()
    if point_estimate not in ("median", "maxl"):
        raise ValueError(
            f"point_estimate must be 'median' or 'maxl', got {point_estimate!r}"
        )

    rho_prefer = str(rho_prefer).strip().lower()
    if rho_prefer not in ("auto", "column", "mr", "logg"):
        raise ValueError(
            f"rho_prefer must be 'auto', 'column', 'mr' or 'logg', "
            f"got {rho_prefer!r}"
        )       
        
    likelihood = str(likelihood).strip().lower()
    if likelihood not in ("normal", "student_t"):
        raise ValueError(
            f"likelihood must be 'normal' or 'student_t', got {likelihood!r}"
        )
    if likelihood == "student_t" and float(student_t_nu) <= 2.0:
        raise ValueError(
            f"student_t_nu must exceed 2 for a finite variance, "
            f"got {student_t_nu}"
        )        

    supersample = int(max(1, supersample))
    os.makedirs(outdir, exist_ok=True)

    # ------------------------------------------------------ unit constants
    # Every physical conversion in this function resolves here, once, from
    # astropy, and is then used as a plain Python float.
    #
    # Quantity objects must NOT reach _model. That function is traced by JAX,
    # which has no unit support, so a Quantity there either raises or silently
    # degrades to numpy and destroys the gradient. Resolve to float here; use
    # floats downstream.
    SEC_PER_DAY = float((1.0 * apu.day).to(apu.s).value)
    MIN_PER_DAY = float((1.0 * apu.day).to(apu.min).value)
    HR_PER_DAY = float((1.0 * apu.day).to(apu.hr).value)
    MIN_PER_HR = float((1.0 * apu.hr).to(apu.min).value)
    G_CGS = float(apc.G.cgs.value)
    GM_SUN_CGS = float(apc.GM_sun.cgs.value)
    CM_PER_AU = float((1.0 * apu.au).to(apu.cm).value)
    RS_OVER_RE = float((1.0 * apu.R_sun).to(apu.R_earth).value)
    RS_OVER_AU = float((1.0 * apu.R_sun).to(apu.au).value)
    # Solar mean density, 1.4098 g/cm^3. This is the "1.41" that used to be
    # hardcoded: rho_star[g/cm^3] = rho_sun * (M/M_sun) / (R/R_sun)^3.
    RHO_SUN_CGS = float(
        (apc.M_sun / ((4.0 / 3.0) * np.pi * apc.R_sun ** 3)).cgs.value
    )
    # Solar surface gravity in cgs, for the logg consistency check below.
    LOGG_SUN = float(np.log10(GM_SUN_CGS / (apc.R_sun.cgs.value ** 2)))
    # Solar effective temperature from L_sun, R_sun and sigma_sb rather than a
    # hardcoded 5772 K. Reproduces the IAU 2015 nominal value by construction.
    T_SUN_K = float(
        ((apc.L_sun / (4.0 * np.pi * apc.R_sun ** 2 * apc.sigma_sb)) ** 0.25)
        .to(apu.K).value
    )
    # Dimensionless flux scale factors. Not unit conversions: astropy has no
    # ppt or ppm, and these are definitional scalings on a ratio.
    PPT = 1.0e3
    PPM = 1.0e6

    # ------------------------------------------------- load the bestfit table
    if isinstance(bestfit, pd.DataFrame):
        bf = bestfit.copy()
        bf_path = None
    else:
        bf_path = str(bestfit)
        bf = pd.read_csv(bf_path)
    if len(bf) == 0:
        raise ValueError("bestfit table is empty")
    row = bf.iloc[0]

    def _get(names: Sequence[str], default: float = np.nan) -> float:
        """First finite value among candidate column names, else default."""
        for nm in names:
            if nm in row.index:
                try:
                    val = float(row[nm])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(val):
                    return val
        return default

    P_tls = _get([f"{method} Period [d]"])
    if not np.isfinite(P_tls):
        raise KeyError(f"'{method} Period [d]' missing or non-finite in bestfit")
    Perr_tls = _get([f"{method} Period Error"])
    T0_tls = _get([f"{method} TC [BTJD]"])
    dur_hr = _get([f"{method} Dur [hrs]"])
    if not np.isfinite(T0_tls) or not np.isfinite(dur_hr) or dur_hr <= 0:
        raise KeyError(
            f"'{method} TC [BTJD]' or '{method} Dur [hrs]' missing or non-finite"
        )
    dur_tls = dur_hr / 24.0

    ror_tls = _get(["RP_RS", f"{method} RP_RS"])
    Rp_RE = _get(["Planet Radius [RE]"])
    Rp_RE_err = _get(["Planet Radius Error"])
    R_star = _get(["Stellar Radius [RS]"])
    M_star = _get(["Stellar Mass [MS]"])
    Teff = _get(["Teff [K]"])

    if not np.isfinite(ror_tls) or ror_tls <= 0:
        if np.isfinite(Rp_RE) and np.isfinite(R_star) and R_star > 0:
            ror_tls = float(
                (Rp_RE * apc.R_earth.to(apu.R_sun).value) / R_star
            )
        else:
            raise KeyError(
                "cannot determine Rp/Rs: 'RP_RS' absent and "
                "'Planet Radius [RE]' / 'Stellar Radius [RS]' unusable"
            )

    # sigma on Rp/Rs, NEMESIS I Eq. 8. The stellar term is dropped when
    # r_star_err is None because the bestfit CSV carries no radius error.
    frac_terms = []
    if np.isfinite(Rp_RE_err) and np.isfinite(Rp_RE) and Rp_RE > 0:
        frac_terms.append((Rp_RE_err / Rp_RE) ** 2)
    if r_star_err is not None and np.isfinite(R_star) and R_star > 0:
        frac_terms.append((float(r_star_err) / R_star) ** 2)
    ror_err = ror_tls * float(np.sqrt(np.sum(frac_terms))) if frac_terms else np.nan
    if not np.isfinite(ror_err) or ror_err <= 0:
        ror_err = 0.5 * ror_tls

        # ------------------------------------------------------ limb darkening in
    # Resolution order: explicit u_fixed, then qld_a/qld_b (the Claret 2017
    # values get_qld hands to both the injection arm and LEO-Vetter), then
    # LEOVetter_transit_u1/u2 (LEO-Vetter takes limb darkening as input rather
    # than fitting it, so these echo the same numbers). Every lookup goes
    # through _get, so a missing column falls through to the next source
    # instead of raising, and a NaN pair never reaches the model.
    ld_source = "u_fixed argument"
    if u_fixed is None:
        qa, qb = _get(["qld_a", "u1"]), _get(["qld_b", "u2"])
        if np.isfinite(qa) and np.isfinite(qb):
            u_fixed, ld_source = (qa, qb), "qld_a/qld_b"
        else:
            qa = _get(["LEOVetter_transit_u1"])
            qb = _get(["LEOVetter_transit_u2"])
            if np.isfinite(qa) and np.isfinite(qb):
                u_fixed, ld_source = (qa, qb), "LEOVetter_transit_u1/u2"
    if ld_mode == "free":
        ld_source = "not used (ld_mode='free')"
    if ld_mode in ("fixed", "gaussian") and u_fixed is None:
        raise ValueError(
            f"ld_mode={ld_mode!r} requires u_fixed=(u1, u2). None of "
            "'qld_a'/'qld_b', 'u1'/'u2' or 'LEOVetter_transit_u1'/"
            "'LEOVetter_transit_u2' is present and finite in this bestfit "
            "row. Obtain them from NEMESIS_pipeline.get_qld(star, verbose) "
            "and pass them in, or set ld_mode='free' for the Kipping (2013) "
            "uninformative prior."
        )
    u_const = (
        np.asarray(np.ravel(u_fixed), dtype=np.float64)
        if u_fixed is not None
        else np.array([0.30, 0.30], dtype=np.float64)
    )
    if u_const.size != 2:
        raise ValueError(f"u_fixed must have exactly 2 elements, got {u_const}")

    def _u_to_q(u_vec: np.ndarray) -> np.ndarray:
        """Kipping (2013) forward transform, u -> q, clipped into the unit square."""
        u1, u2 = float(u_vec[0]), float(u_vec[1])
        denom = u1 + u2
        if abs(denom) < 1e-12:
            return np.array([0.25, 0.50])
        q1 = denom ** 2
        q2 = 0.5 * u1 / denom
        if q1 > 1.0:
            warnings.warn(
                f"(u1+u2)^2 = {q1:.3f} exceeds 1, outside the Kipping "
                "physical triangle. Clipping q1 to 1.",
                RuntimeWarning,
            )
        return np.clip(np.array([q1, q2]), 1e-4, 1.0 - 1e-4)

    q_mu = _u_to_q(u_const)
    if not np.all(np.isfinite(u_const)):
        raise ValueError(
            f"limb-darkening coefficients are not finite: {u_const} "
            f"(source: {ld_source})"
        )
    if verbose:
        print(f"  [prior] u = ({u_const[0]:.4f}, {u_const[1]:.4f}) from "
              f"{ld_source}; Kipping q = ({q_mu[0]:.4f}, {q_mu[1]:.4f})")

    # ------------------------------------------------------ load light curve
    if isinstance(lightcurve, pd.DataFrame):
        lc = lightcurve.copy()
        lc_path = None
    else:
        lc_path = str(lightcurve)
        lc = pd.read_csv(lc_path)

    for col in (time_col, flux_col, err_col):
        if col not in lc.columns:
            raise KeyError(
                f"light curve column {col!r} not found. Available: "
                f"{list(lc.columns)}"
            )

    t = lc[time_col].to_numpy(dtype=np.float64)
    f = lc[flux_col].to_numpy(dtype=np.float64)
    e = lc[err_col].to_numpy(dtype=np.float64)

    n_start = t.size
    if quality_col is not None and quality_col in lc.columns:
        qf = lc[quality_col].to_numpy()
        keep_q = np.isin(qf, np.asarray(list(quality_accept)))
    else:
        keep_q = np.ones(n_start, dtype=bool)

    keep = keep_q & np.isfinite(t) & np.isfinite(f) & np.isfinite(e) & (e > 0)
    t, f, e = t[keep], f[keep], e[keep]
    order = np.argsort(t)
    t, f, e = t[order], f[order], e[order]

    if exposure_days is None:
        dt = np.diff(t)
        dt = dt[(dt > 0) & np.isfinite(dt)]
        exposure_days = (float(np.nanmedian(dt)) if dt.size else float((30.0 * apu.min).to(apu.day).value) )
    exposure_days = float(exposure_days)

    f_med = float(np.nanmedian(f))
    if normalize and abs(f_med - 1.0) > 0.05:
        if verbose:
            print(f"  [prep] median flux {f_med:.5g}, normalizing to unity")
        f = f / f_med
        e = e / f_med

    n_full = t.size
    # Retained at full length for the time-series figure. The fit itself uses
    # only the windowed arrays below.
    t_all, f_all, e_all = t.copy(), f.copy(), e.copy()
    in_window = np.ones(n_full, dtype=bool)
    if window_durations is not None and window_durations > 0:
        phase_d = (t - T0_tls + 0.5 * P_tls) % P_tls - 0.5 * P_tls
        win = np.abs(phase_d) <= float(window_durations) * dur_tls
        if int(win.sum()) >= 50:
            t, f, e = t[win], f[win], e[win]
            in_window = win
        elif verbose:
            print(
                f"  [prep] window kept only {int(win.sum())} cadences, "
                "falling back to the full light curve"
            )

    if t.size < 20:
        raise ValueError(
            f"only {t.size} usable cadences after masking. Check quality_accept "
            f"and the flux column. Started with {n_start}."
        )

    err_med = float(np.nanmedian(e))
    sigma_mean = err_med / np.sqrt(t.size)
    f0_sd = float(max(10.0 * sigma_mean, 1e-5))

    if verbose:
        print(
            f"  [prep] {n_start} -> {n_full} finite/quality -> {t.size} in window; "
            f"texp={exposure_days*1440:.2f} min, supersample={supersample}"
        )

    # ------------------------------------------------------------ prior scales
    if np.isfinite(Perr_tls) and Perr_tls > 0:
        sig_lnP = float(np.clip(Perr_tls / P_tls, 1e-4, 0.05))
    else:
        sig_lnP = 1e-3
        if verbose:
            print(
                "  [prior] TLS Period Error absent or non-positive, "
                "using sigma_lnP = 1e-3"
            )
    t0_sd = float(max(t0_sigma_durations * dur_tls, 1e-4))
    b_lo = 0.01
    r_lo = 0.01
    #r_hi = float(max(ror_tls + float(r_hi_sigmas) * ror_err, 0.02))
    if r_hi_abs is not None:
        r_hi = float(r_hi_abs)
        if not (np.isfinite(r_hi) and r_hi > r_lo):
            raise ValueError(
                f"r_hi_abs must be a finite number above {r_lo}, got {r_hi_abs!r}"
            )
    else:
        r_hi = float(max(ror_tls + float(r_hi_sigmas) * ror_err, 0.02))

    # ------------------------------------------- stellar density resolution
    def _resolve_rho_solar() -> Tuple[float, str, bool]:
        """
        Resolve the stellar mean density in SOLAR units, with provenance.

        Solar units rather than cgs so the function has exactly one conversion
        point, at the call site. Every source below is naturally solar, so none
        needs its own conversion and none can be double-converted.

        Sources, in ``rho_prefer='auto'`` order:

        1. ``M_star / R_star**3``. First because the reported ``a [AU]`` comes
           from the same ``M_star`` through Kepler's third law. Taking the
           density from a separate column can leave the prior and the reported
           semi-major axis describing two different stars.
        2. ``rho [rho_sun]``. Present once repair_bestfit_rho_units has run on
           this file; holds the verbatim TIC value.
        3. ``rho [g/ccm]``, unit sniffed rather than assumed. On an unrepaired
           file that column stores a solar-unit value under a cgs label; on a
           repaired one it is genuine cgs. The ratio to rho_sun*M/R**3
           distinguishes the two: near rho_sun means solar, near 1 means cgs.
        4. ``logg`` with ``R_star``, via rho/rho_sun = 10**(logg - logg_sun) /
           (R/R_sun), which follows from rho/rho_sun = (g/g_sun)/(R/R_sun).
           Independent of ``M_star``, so it covers a missing mass and also
           serves as a cross-check.

        Cross-checks run under every ``rho_prefer`` setting, so a disagreement
        between sources is reported even when it does not change the value
        used.

        Returns
        -------
        (rho_solar, source, repaired) : (float, str, bool)
            ``rho_solar`` is NaN when no source is usable. ``repaired`` is True
            when a finite ``rho [rho_sun]`` column was found.
        """
        col_solar = _get(["rho [rho_sun]"])
        col_cgs = _get(["rho [g/ccm]"])
        logg_col = _get(["logg"])
        repaired = bool(np.isfinite(col_solar) and col_solar > 0)

        r_ok = bool(np.isfinite(R_star) and R_star > 0)
        mr_solar = (
            float(M_star / (R_star ** 3))
            if np.isfinite(M_star) and r_ok else np.nan
        )
        logg_solar = (
            float((10.0 ** (logg_col - LOGG_SUN)) / R_star)
            if np.isfinite(logg_col) and r_ok else np.nan
        )

        # --- sniff the unit of rho [g/ccm] --------------------------------
        cgs_as_solar = np.nan
        cgs_label = "rho [g/ccm] column"
        if np.isfinite(col_cgs) and col_cgs > 0:
            if repaired:
                cgs_as_solar = col_cgs / RHO_SUN_CGS
                cgs_label = "rho [g/ccm] column (cgs, file repaired)"
            elif np.isfinite(mr_solar) and mr_solar > 0:
                ratio = (RHO_SUN_CGS * mr_solar) / col_cgs
                if abs(ratio - RHO_SUN_CGS) / RHO_SUN_CGS < rho_unit_tol:
                    cgs_as_solar = col_cgs
                    cgs_label = "rho [g/ccm] column (solar units, unrepaired)"
                elif abs(ratio - 1.0) < rho_unit_tol:
                    cgs_as_solar = col_cgs / RHO_SUN_CGS
                    cgs_label = "rho [g/ccm] column (cgs, no marker column)"
            else:
                # No rho [rho_sun] and no usable M, R, so the unit cannot be
                # sniffed. Taken as solar, because that is what
                # NEMESIS_pipeline.py writes under this label. Face value
                # would be wrong by rho_sun.
                cgs_as_solar = col_cgs
                cgs_label = "rho [g/ccm] column (assumed solar, unsniffable)"
                warnings.warn(
                    f"'rho [g/ccm]' = {col_cgs:.4f} cannot have its unit "
                    f"verified: no 'rho [rho_sun]' column and no usable "
                    f"M_star, R_star to compare against. Assuming SOLAR "
                    f"units, which is what NEMESIS_pipeline.py writes under "
                    f"this label. Pass rho_star_prior=(mu, sigma) in g/cm^3 "
                    f"if that is wrong for this row.",
                    RuntimeWarning,
                )

        # --- pick a source ------------------------------------------------
        chains = {
            "mr": [("M_star / R_star**3", mr_solar)],
            "logg": [("logg with R_star", logg_solar)],
            "column": [("rho [rho_sun] column", col_solar),
                       (cgs_label, cgs_as_solar)],
            "auto": [("M_star / R_star**3", mr_solar),
                     ("rho [rho_sun] column", col_solar),
                     (cgs_label, cgs_as_solar),
                     ("logg with R_star", logg_solar)],
        }
        rho_solar, source = np.nan, "none"
        for name, val in chains[rho_prefer]:
            if np.isfinite(val) and val > 0:
                rho_solar, source = float(val), name
                break

        if not np.isfinite(rho_solar):
            tried = ", ".join(n for n, _ in chains[rho_prefer])
            warnings.warn(
                f"no usable stellar density from rho_prefer={rho_prefer!r} "
                f"(tried: {tried}). Columns seen: rho [rho_sun]={col_solar}, "
                f"rho [g/ccm]={col_cgs}, logg={logg_col}, "
                f"M_star={M_star}, R_star={R_star}.",
                RuntimeWarning,
            )
            return (np.nan, source, repaired)

        # --- cross-checks, independent of which source won ----------------
        if repaired and np.isfinite(col_cgs) and col_cgs > 0:
            expect = RHO_SUN_CGS * col_solar
            if abs(col_cgs - expect) / expect > rho_unit_tol:
                warnings.warn(
                    f"this file has a 'rho [rho_sun]' column ({col_solar:.4f}) "
                    f"but 'rho [g/ccm]' = {col_cgs:.4f} is not rho_sun times "
                    f"it ({expect:.4f}). The repair is internally "
                    f"inconsistent for this row.",
                    RuntimeWarning,
                )

        if (not repaired) and np.isfinite(col_cgs) and col_cgs > 0 \
                and np.isfinite(mr_solar) and mr_solar > 0:
            ratio = (RHO_SUN_CGS * mr_solar) / col_cgs
            if abs(ratio - RHO_SUN_CGS) / RHO_SUN_CGS < rho_unit_tol:
                warnings.warn(
                    f"'rho [g/ccm]' = {col_cgs:.4f} equals "
                    f"M_star/R_star**3 = {mr_solar:.4f}, the density in SOLAR "
                    f"units, not g/cm^3: the ratio to rho_sun*M/R**3 is "
                    f"{ratio:.4f}, i.e. rho_sun itself. The COLUMN LABEL is "
                    f"wrong, not the stellar parameters. Run "
                    f"repair_bestfit_rho_units on this file, and fix the unit "
                    f"in NEMESIS_pipeline.py before the column is released.",
                    RuntimeWarning,
                )
            elif abs(ratio - 1.0) >= rho_unit_tol and \
                    abs(col_cgs - RHO_SUN_CGS * mr_solar) \
                    / (RHO_SUN_CGS * mr_solar) > 0.10:
                warnings.warn(
                    f"'rho [g/ccm]' = {col_cgs:.4f} is neither solar units "
                    f"nor consistent with rho_sun*M/R**3 = "
                    f"{RHO_SUN_CGS*mr_solar:.4f} g/cm^3 (ratio {ratio:.4f} is "
                    f"neither rho_sun nor 1). The stellar parameters may "
                    f"genuinely be inconsistent. Using {source}.",
                    RuntimeWarning,
                )

        if np.isfinite(logg_solar) and logg_solar > 0 and \
                abs(logg_solar - rho_solar) / rho_solar > 0.05:
            warnings.warn(
                f"logg implies rho/rho_sun = {logg_solar:.4f}, which "
                f"disagrees with the {source} value {rho_solar:.4f} by "
                f"{100*abs(logg_solar-rho_solar)/rho_solar:.1f} percent. "
                f"logg, M_star and R_star are not mutually consistent for "
                f"this row.",
                RuntimeWarning,
            )

        if verbose:
            print(f"  [prior] rho/rho_sun = {rho_solar:.4f} from {source}; "
                  f"file repaired = {repaired}")
        return (float(rho_solar), source, repaired)

    # ----------------------------------------- optional stellar density prior
    rho_mu: Optional[float] = None
    rho_log_sd: float = 0.0
    rho_src: str = "none"
    rho_solar_src: str = "none"
    rho_repaired: bool = False
    if rho_star_prior is not False and rho_star_prior is not None:
        if isinstance(rho_star_prior, (tuple, list)) and len(rho_star_prior) == 2:
            rho_mu, rho_sig = float(rho_star_prior[0]), float(rho_star_prior[1])
            rho_src = "explicit (mu, sigma)"
        else:
            # One conversion point. _resolve_rho_solar returns solar units and
            # handles source selection, unit sniffing and cross-checking, so
            # nothing here needs to know which column supplied the value.
            rho_solar, rho_solar_src, rho_repaired = _resolve_rho_solar()
            if np.isfinite(rho_solar) and rho_solar > 0:
                rho_mu = RHO_SUN_CGS * rho_solar
                rho_src = f"rho_sun * ({rho_solar_src})"
            else:
                rho_mu = np.nan
                rho_src = "unresolved"
            rho_sig = (
                float(rho_frac_err) * rho_mu
                if np.isfinite(rho_mu) and rho_mu > 0 else np.nan
            )
            rho_sig = (
                float(rho_frac_err) * rho_mu
                if np.isfinite(rho_mu) and rho_mu > 0 else np.nan
            )
        if not np.isfinite(rho_mu) or rho_mu <= 0 or not np.isfinite(rho_sig) \
                or rho_sig <= 0:
            warnings.warn(
                "rho_star_prior requested but no usable stellar density. "
                "Continuing without it; b will be prior-dominated.",
                RuntimeWarning,
            )
            rho_mu = None
        else:
            rho_log_sd = float(np.log1p(rho_sig / rho_mu))
            if verbose:
                _aRs_implied = ( G_CGS * rho_mu * (P_tls * SEC_PER_DAY) ** 2 / (3.0 * np.pi) ) ** (1.0 / 3.0)
                print(f"  [prior] rho_star = {rho_mu:.3f} +/- {rho_sig:.3f} "
                      f"g/cm^3 from {rho_src}; sigma_ln = {rho_log_sd:.4f}; "
                      f"implied a/Rs = {_aRs_implied:.2f}")

    # ------------------------------------------------------ model definition

    # ------------------------------------------------------ model definition
    # Offsets for midpoint integration across the exposure (Kipping 2010).
    if supersample > 1:
        offsets = jnp.asarray(
            (np.arange(supersample) - 0.5 * (supersample - 1))
            * exposure_days
            / supersample
        )
    else:
        offsets = None

    u_const_j = jnp.asarray(u_const)
    q_mu_j = jnp.asarray(q_mu)

    def _q_to_u(q):
        """Kipping (2013) inverse transform, q -> u."""
        sqrt_q1 = jnp.sqrt(q[0])
        return jnp.stack([2.0 * sqrt_q1 * q[1], sqrt_q1 * (1.0 - 2.0 * q[1])])

    def _transit(t_arr, t0, period, duration, r, b, u):
        """Exposure-integrated fractional flux deficit, shape (len(t_arr),)."""
        orbit = TransitOrbit(
            period=period,
            duration=duration,
            time_transit=t0,
            impact_param=b,
            radius_ratio=r,
        )
        lc_fn = limb_dark_light_curve(orbit, u)
        n = t_arr.shape[0]
        if offsets is None:
            return jnp.reshape(lc_fn(t_arr), (n,))
        tt = jnp.reshape(t_arr[:, None] + offsets[None, :], (-1,))
        return jnp.mean(jnp.reshape(lc_fn(tt), (n, supersample)), axis=1)

    def _model(t_obs, yerr_obs, y_obs=None):
        t0 = numpyro.sample("t0", dist.Normal(T0_tls, t0_sd))

        log_period = numpyro.sample(
            "log_period", dist.Normal(float(np.log(P_tls)), sig_lnP)
        )
        period = numpyro.deterministic("period", jnp.exp(log_period))

        log_duration = numpyro.sample(
            "log_duration",
            dist.Normal(float(np.log(dur_tls)), float(duration_log_sigma)),
        )
        duration = numpyro.deterministic("duration", jnp.exp(log_duration))

        if prior_style == "paper1":
            r = numpyro.sample("r", dist.Uniform(r_lo, r_hi))
        else:
            log_r = numpyro.sample(
                "log_r",
                dist.Normal(float(np.log(ror_tls)), float(r_log_sigma)),
            )
            r = numpyro.deterministic("r", jnp.exp(log_r))

        # b sampled through a unit proxy so its ceiling can depend on r.
        b_frac = numpyro.sample("b_frac", dist.Uniform(0.0, 1.0))
        b = numpyro.deterministic("b", b_lo + b_frac * ((1.0 + r) - b_lo))

        # Exactly one u_q sample site, or none at all.
        if ld_mode == "fixed":
            u = u_const_j
        elif ld_mode == "free":
            q = numpyro.sample(
                "u_q",
                dist.Uniform(1e-6, 1.0 - 1e-6).expand((2,)).to_event(1),
            )
            u = numpyro.deterministic("u", _q_to_u(q))
        else:
            q = numpyro.sample(
                "u_q",
                dist.TruncatedNormal(
                    q_mu_j,
                    float(ld_sigma),
                    low=1e-6,
                    high=1.0 - 1e-6,
                ).to_event(1),
            )
            u = numpyro.deterministic("u", _q_to_u(q))

        if rho_mu is not None:
            # Circular-orbit inversion: T14, P, b and r imply a/Rs, which
            # implies rho_star. Constraining rho_star therefore pins a/Rs and
            # converts b from unmeasured into T14-determined.
            sin_arg = jnp.sin(jnp.pi * duration / period)
            aRs_m = jnp.sqrt(
                b ** 2
                + jnp.clip((1.0 + r) ** 2 - b ** 2, 0.0, None)
                / jnp.clip(sin_arg ** 2, 1e-12, None)
            )
            rho_m = numpyro.deterministic(
                "rho_star_model",
                3.0 * jnp.pi / (G_CGS * (period * SEC_PER_DAY) ** 2) * aRs_m ** 3,
            )
            numpyro.factor(
                "rho_star_prior_factor",
                dist.Normal(float(np.log(rho_mu)), rho_log_sd).log_prob(
                    jnp.log(jnp.clip(rho_m, 1e-8, None))
                ),
            )

        deficit = _transit(t_obs, t0, period, duration, r, b, u)

        if fit_offset:
            f0 = numpyro.sample("f0", dist.Normal(1.0, f0_sd))
        else:
            f0 = 1.0
        y_model = f0 * (1.0 + deficit)

        if fit_jitter:
            log_jitter = numpyro.sample(
                "log_jitter", dist.Normal(float(np.log(err_med)), 2.0)
            )
            sig = jnp.sqrt(yerr_obs ** 2 + jnp.exp(2.0 * log_jitter))
        else:
            sig = yerr_obs

        if likelihood == "student_t":
            # Heavy tails absorb a few outlying cadences locally instead of
            # inflating sig for all of them, which is what a Normal plus a
            # jitter term does.
            numpyro.sample(
                "obs",
                dist.StudentT(float(student_t_nu), y_model, sig),
                obs=y_obs,
            )
        else:
            numpyro.sample("obs", dist.Normal(y_model, sig), obs=y_obs)

    # Init values. Keys MUST match the sample site names above. Any key that
    # does not match is silently ignored by init_to_value, which is exactly
    # how the notebook version lost its radius initialization.
    init_vals: Dict[str, Any] = {
        "t0": float(T0_tls),
        "log_period": float(np.log(P_tls)),
        "log_duration": float(np.log(dur_tls)),
        "b_frac": 0.30,
    }
    if prior_style == "paper1":
        init_vals["r"] = float(np.clip(ror_tls, r_lo + 1e-6, r_hi - 1e-6))
    else:
        init_vals["log_r"] = float(np.log(ror_tls))
    if ld_mode in ("free", "gaussian"):
        init_vals["u_q"] = jnp.asarray(q_mu)
    if fit_offset:
        init_vals["f0"] = 1.0
    if fit_jitter:
        init_vals["log_jitter"] = float(np.log(err_med))

    # ------------------------------------------------------------------- MAP
    t_j = jnp.asarray(t)
    f_j = jnp.asarray(f)
    e_j = jnp.asarray(e)

    key = jax.random.PRNGKey(int(seed))
    key_map, key_mcmc = jax.random.split(key)

    guide = numpyro.infer.autoguide.AutoDelta(
        _model, init_loc_fn=numpyro.infer.init_to_value(values=init_vals)
    )
    svi = numpyro.infer.SVI(
        _model, guide, numpyro.optim.Adam(step_size=1e-2),
        loss=numpyro.infer.Trace_ELBO(),
    )
    if verbose:
        print(f"  [map ] AutoDelta + Adam, {n_map_steps} steps")
    svi_result = svi.run(
        key_map, int(n_map_steps), t_j, e_j, y_obs=f_j, progress_bar=False
    )
    map_params = guide.median(svi_result.params)

    map_ok = all(bool(np.all(np.isfinite(np.asarray(v)))) for v in map_params.values())
    final_loss = float(np.asarray(svi_result.losses)[-1])
    if (not map_ok) or (not np.isfinite(final_loss)):
        warnings.warn(
            "MAP optimization returned non-finite values. Falling back to the "
            "prior-based init for NUTS.",
            RuntimeWarning,
        )
        map_params = {k: jnp.asarray(v) for k, v in init_vals.items()}
    elif verbose:
        print(f"  [map ] final Evidence Lower Bound (ELBO) loss {final_loss:.4f}")

    # ------------------------------------------------------------------ NUTS
    n_dev = jax.local_device_count()
    chain_method = "parallel" if n_dev >= n_chains else "sequential"
    if verbose and chain_method == "sequential" and n_chains > 1:
        print(
            f"  [nuts] {n_dev} JAX device(s) for {n_chains} chains, running "
            "sequentially. Call numpyro.set_host_device_count(n_chains) before "
            "any JAX work to parallelize."
        )

    kernel = numpyro.infer.NUTS(
        _model,
        dense_mass=bool(dense_mass),
        target_accept_prob=float(target_accept),
        init_strategy=numpyro.infer.init_to_value(values=map_params),
    )
    mcmc = numpyro.infer.MCMC(
        kernel,
        num_warmup=int(n_warmup),
        num_samples=int(n_samples),
        num_chains=int(n_chains),
        chain_method=chain_method,
        progress_bar=bool(verbose),
    )
    mcmc.run(key_mcmc, t_j, e_j, y_obs=f_j)
    idata = az.from_numpyro(mcmc)
    post_samples = mcmc.get_samples()

    # ---------------------------------------------------------- diagnostics
    diag_vars = [
        v for v in ("t0", "period", "duration", "r", "b", "f0", "log_jitter", "u")
        if v in idata.posterior
    ]
    az_summ = az.summary(idata, var_names=diag_vars)
    rhat_max = float(np.nanmax(az_summ["r_hat"].to_numpy()))
    ess_min = float(np.nanmin(az_summ["ess_bulk"].to_numpy()))
    ess_tail_min = (
        float(np.nanmin(az_summ["ess_tail"].to_numpy()))
        if "ess_tail" in az_summ.columns else np.nan
    )
    try:
        n_div = int(np.asarray(idata.sample_stats["diverging"].values).sum())
    except (KeyError, AttributeError):
        n_div = -1
    # ess_tail governs the 16th and 84th percentiles, which are what gets
    # published. ess_bulk governs the median. Both are gated.
    converged = bool(
        rhat_max <= 1.01
        and ess_min >= 400
        and (not np.isfinite(ess_tail_min) or ess_tail_min >= 400)
        and n_div == 0
    )

    if verbose:
        print(
            f"  [diag] r_hat max {rhat_max:.4f}, ess_bulk min {ess_min:.0f}, "
            f"ess_tail min {ess_tail_min:.0f}, divergences {n_div}, "
            f"converged {converged}"
        )
    if not converged:
        warnings.warn(
            f"Convergence gate not met for {save_name or 'target'}: "
            f"r_hat_max={rhat_max:.4f} (<=1.01), ess_bulk_min={ess_min:.0f} "
            f"(>=400), divergences={n_div} (==0). Do not publish these "
            "intervals without re-running with more warmup or a higher "
            "target_accept.",
            RuntimeWarning,
        )
    #
    # --------------------------------------------------- derived quantities
    def _flat(name: str) -> np.ndarray:
        return np.asarray(idata.posterior[name].values).reshape(-1)

    t0_s = _flat("t0")
    P_s = _flat("period")
    D_s = _flat("duration")
    r_s = _flat("r")
    b_s = _flat("b")
    n_draw = t0_s.size

    if "u" in idata.posterior:
        u_s = np.asarray(idata.posterior["u"].values).reshape(n_draw, 2)
    else:
        u_s = np.tile(u_const, (n_draw, 1))

    f0_s = _flat("f0") if "f0" in idata.posterior else np.ones(n_draw)
    # ------------------------------------------- maximum-likelihood draw
    # The median is what gets reported (it is invariant under
    # reparameterization and it is the field convention), but the single
    # best-fitting sampled point is worth carrying alongside it: it is a real
    # point in parameter space rather than a stack of marginal summaries, and
    # a large gap between the two flags a skewed or degenerate posterior.
    i_best = 0
    loglike_best = np.nan
    try:
        if not np.allclose(
            np.asarray(post_samples["t0"]).reshape(-1), t0_s, rtol=0, atol=0
        ):
            raise RuntimeError(
                "mcmc.get_samples() ordering does not match arviz; refusing "
                "to index the max-likelihood draw."
            )
        _ll = numpyro.infer.util.log_likelihood(
            _model, post_samples, t_j, e_j, y_obs=f_j
        )["obs"]
        loglike_s = np.asarray(_ll).reshape(n_draw, -1).sum(axis=1)
        i_best = int(np.nanargmax(loglike_s))
        loglike_best = float(loglike_s[i_best])
    except Exception as exc:
        warnings.warn(
            f"max-likelihood draw not computed: {type(exc).__name__}: {exc}. "
            "MaxL columns will report the first draw.",
            RuntimeWarning,
        )

    # --------------------------------------------- radius ceiling check
    # Must run after r_s exists. The Uniform radius prior has a hard upper
    # edge; when the posterior is pressed against it, the upper error bar on
    # Rp/Rs is set by where the prior stops rather than by the data, and the
    # high-b branch is truncated along with it.
    if prior_style == "paper1":
        _ceil_frac = float(np.mean(r_s > 0.98 * r_hi))
        if verbose:
            print(f"  [prior] Rp/Rs ceiling r_hi={r_hi:.5f}; "
                  f"{100*_ceil_frac:.2f} percent of draws within 2 percent "
                  f"of it")
        if _ceil_frac > 0.01:
            warnings.warn(
                f"{100*_ceil_frac:.1f} percent of the Rp/Rs posterior sits "
                f"within 2 percent of the prior ceiling r_hi={r_hi:.5f}. The "
                "upper error bar on Rp/Rs is prior-limited, and the high-b "
                "branch is truncated. "
                + ("With a fixed edge this usually means the dip is deeper "
                   "than this star can produce; check the flux contamination "
                   "correction." if r_hi_abs is not None
                   else "Re-run with a larger r_hi_sigmas."),
                RuntimeWarning,
            )

    # Circular-orbit inversion of the T14 expression for a/Rs:
    #   sin(pi T14 / P) = sqrt((1+r)^2 - b^2) / sqrt((a/Rs)^2 - b^2)
    with np.errstate(invalid="ignore", divide="ignore"):
        sin_arg = np.sin(np.pi * D_s / P_s)
        num = np.clip((1.0 + r_s) ** 2 - b_s ** 2, 0.0, None)
        aRs_s = np.sqrt(b_s ** 2 + num / np.clip(sin_arg ** 2, 1e-12, None))
        inc_s = np.degrees(np.arccos(np.clip(b_s / np.clip(aRs_s, 1e-12, None), -1, 1)))

    P_sec = P_s * SEC_PER_DAY
    rho_s = 3.0 * np.pi / (G_CGS * P_sec ** 2) * aRs_s ** 3  # g cm^-3

    rng = np.random.default_rng(int(seed) + 1)
    if np.isfinite(R_star) and R_star > 0:
        if r_star_err is not None and float(r_star_err) > 0:
            Rstar_s = rng.normal(R_star, float(r_star_err), size=n_draw)
            Rstar_s = np.clip(Rstar_s, 1e-3, None)
        else:
            Rstar_s = np.full(n_draw, R_star)
    else:
        Rstar_s = np.full(n_draw, np.nan)

    Rp_s = r_s * Rstar_s * RS_OVER_RE

    # Semi-major axis from Kepler's third law, NOT from the fitted a/Rs.
    # The transit-shape a/Rs inherits the impact-parameter prior whenever b is
    # unconstrained, which at 30 minute cadence is the normal case for this
    # survey. a from P and M_star is roughly an order of magnitude tighter and
    # carries no dependence on b.
    if np.isfinite(M_star) and M_star > 0:
        if m_star_err is not None and float(m_star_err) > 0:
            Mstar_s = np.clip(
                rng.normal(M_star, float(m_star_err), size=n_draw), 1e-3, None
            )
        else:
            Mstar_s = np.full(n_draw, M_star)
        # Kepler's third law in cgs from astropy's GM_sun, rather than the
        # a^3 = M P^2 shortcut in solar-and-Julian-year units, which is off by
        # roughly 0.05 percent in a because 1 AU^3 / yr_Julian^2 is not exactly
        # GM_sun / 4 pi^2. The planet mass is neglected: for a few-Earth-mass
        # planet on an M dwarf that is a fractional error in a below 1e-5.
        a_cm_s = ( GM_SUN_CGS * Mstar_s * (P_s * SEC_PER_DAY) ** 2 / (4.0 * np.pi ** 2) ) ** (1.0 / 3.0)
        a_AU_s = a_cm_s / CM_PER_AU
        aRs_dyn_s = a_AU_s / np.clip(Rstar_s * RS_OVER_AU, 1e-12, None)
    else:
        a_AU_s = aRs_s * Rstar_s * RS_OVER_AU
        aRs_dyn_s = aRs_s.copy()
        warnings.warn(
            "'Stellar Mass [MS]' unusable, falling back to the transit-shape "
            "a/Rs for a, insolation and T_eq. Those three then inherit the "
            "impact-parameter prior and their intervals are not measurements.",
            RuntimeWarning,
        )

    if np.isfinite(Teff) and Teff > 0:
        # S/S_earth = (L_star/L_sun) / (a/AU)^2, with
        # L_star/L_sun = (R/R_sun)^2 (Teff/T_sun)^4. T_sun comes from astropy's
        # L_sun, R_sun and sigma_sb rather than a hardcoded 5772 K.
        S_s = (Rstar_s ** 2) * ((Teff / T_SUN_K) ** 4) \
            / np.clip(a_AU_s ** 2, 1e-12, None)
        # Zero albedo and full day-night redistribution. Both are assumptions,
        # not measurements; say so wherever Teq is reported.
        Teq_s = Teff * np.sqrt(1.0 / np.clip(2.0 * aRs_dyn_s, 1e-12, None))
    else:
        S_s = np.full(n_draw, np.nan)
        Teq_s = np.full(n_draw, np.nan)

    depth_ppt_s = (r_s ** 2) * 1e3

    # -------------------------------------------------------- summary record
    def _pct(x: np.ndarray) -> Tuple[float, float, float]:
        good = np.asarray(x, dtype=float)
        good = good[np.isfinite(good)]
        if good.size == 0:
            return (np.nan, np.nan, np.nan)
        lo, med, hi = np.percentile(good, [15.865, 50.0, 84.135])
        return (float(med), float(med - lo), float(hi - med))

    fields: List[Tuple[str, np.ndarray]] = [
        ("Period [d]", P_s),
        ("T0 [BTJD]", t0_s),
        ("Dur [hrs]", D_s * 24.0),
        ("RP_RS", r_s),
        ("b", b_s),
        ("a_over_R_star (shape)", aRs_s),
        ("a_over_R_star (dynamical)", aRs_dyn_s),
        ("Inclination [deg]", inc_s),
        ("rho_star [g/ccm]", rho_s),
        ("Planet Radius [RE]", Rp_s),
        ("a [AU]", a_AU_s),
        ("Insolation [SE]", S_s),
        ("Teq [K]", Teq_s),
        ("Depth [ppt]", depth_ppt_s),
        ("u1", u_s[:, 0]),
        ("u2", u_s[:, 1]),
        ("f0", f0_s),
    ]
    if "log_jitter" in idata.posterior:
        fields.append(("Jitter [ppm]", np.exp(_flat("log_jitter")) * 1e6))

    # ------------------------------------------------- prior shrinkage
    # s = 1 - (sd_post / sd_prior)^2. s near 0 means the posterior is the
    # prior and the reported interval is not a measurement; s near 1 means
    # the data dominate. Draws come from the same model with the likelihood
    # switched off, so every prior transformation is handled automatically.
    shrink: Dict[str, float] = {}
    try:
        # log_jitter and f0 are included because both can be
        # prior-dominated. A jitter term whose posterior reproduces its prior
        # is reporting the reported-error scale back, not measuring excess
        # scatter, and it still inflates every credible interval.
        shrink_pairs = [
            ("t0", t0_s), ("period", P_s), ("duration", D_s),
            ("r", r_s), ("b", b_s),
        ]
        if fit_offset and "f0" in idata.posterior:
            shrink_pairs.append(("f0", f0_s))
        if fit_jitter and "log_jitter" in idata.posterior:
            # Compared in log space, where the prior lives.
            shrink_pairs.append(("log_jitter", _flat("log_jitter")))
        prior_draws = numpyro.infer.Predictive(
            _model, num_samples=2000,
            return_sites=tuple(nm for nm, _ in shrink_pairs),
        )(jax.random.PRNGKey(int(seed) + 7), t_j, e_j, y_obs=None)
        for nm, post_arr in shrink_pairs:
            sd_pr = float(np.nanstd(np.asarray(prior_draws[nm]).reshape(-1)))
            sd_po = float(np.nanstd(post_arr))
            shrink[nm] = (
                float(np.clip(1.0 - (sd_po / sd_pr) ** 2, 0.0, 1.0))
                if sd_pr > 0 else np.nan
            )
    except Exception as exc:
        warnings.warn(
            f"prior shrinkage diagnostic failed: {type(exc).__name__}: {exc}",
            RuntimeWarning,
        )

    if verbose and shrink:
        if rho_mu is None:
            note = "   (<0.3 prior-dominated, >0.7 data-dominated)"
        else:
            note = ("   NOTE: rho_star_prior is ON, so b shrinkage measures "
                    "the density prior, not the light curve")
        print("  [shrink] " + "  ".join(
            f"{k}={v:.2f}" for k, v in shrink.items()) + note)

    rec: Dict[str, Any] = {}
    for name, arr in fields:
        med, lo, hi = _pct(arr)
        rec[f"MCMC {name}"] = med
        rec[f"MCMC {name} Err Lo"] = lo
        rec[f"MCMC {name} Err Hi"] = hi
        _bv = np.asarray(arr, dtype=float)[i_best]
        rec[f"MCMC {name} MaxL"] = float(_bv) if np.isfinite(_bv) else np.nan

    # -------------------------------------------------------- naming / ids
    # Bestfit CSVs written by NEMESIS_pipeline.py carry no TIC identifier at
    # all; the collector adds 'ID' later. Try both conventions, then fall back
    # to the regex on save_name below.
    tic_id = next(
        (row[c] for c in ("TIC_ID", "ID", "TIC") if c in row.index), np.nan)
    sector_tag = next(
        (row[c] for c in ("Sector", "sector") if c in row.index), np.nan)
    #
    if save_name is None:
        stem = os.path.basename(bf_path) if bf_path else "target"
        m = re.match(r"(TIC_\d+_.+?)_(?:BLS|TLS)_bestfit\.csv$", stem)
        save_name = m.group(1) if m else os.path.splitext(stem)[0]
    if not (isinstance(tic_id, (int, float)) and np.isfinite(tic_id)):
        m_tic = re.search(r"TIC[_ ]?(\d+)", str(save_name))
        tic_id = int(m_tic.group(1)) if m_tic else np.nan

    rec.update({
        "TIC_ID": tic_id,
        "Sector": sector_tag,
        "save_name": save_name,
        "method": method,
        "prior_style": prior_style,
        "ld_mode": ld_mode,
        "ld_source": ld_source,
        "N cadences fit": int(t.size),
        "Exposure [min]": exposure_days * 1440.0,
        "Supersample": supersample,
        "MCMC rhat max": rhat_max,
        "MCMC ess bulk min": ess_min,
        "MCMC ess tail min": ess_tail_min,
        "MCMC n divergences": n_div,
        "MCMC n draws": int(n_draw),
        "MCMC max loglike": loglike_best,
        "MCMC point estimate": point_estimate,
        "MCMC likelihood": likelihood,
        "MCMC student_t_nu": ( float(student_t_nu) if likelihood == "student_t" else np.nan ),        
        "MCMC converged": converged,
        "MCMC shrinkage t0": shrink.get("t0", np.nan),
        "MCMC shrinkage period": shrink.get("period", np.nan),
        "MCMC shrinkage duration": shrink.get("duration", np.nan),
        "MCMC shrinkage RP_RS": shrink.get("r", np.nan),
        "MCMC shrinkage b": shrink.get("b", np.nan),
        "MCMC shrinkage f0": shrink.get("f0", np.nan),
        "MCMC shrinkage log_jitter": shrink.get("log_jitter", np.nan),
        "MCMC jitter over reported err": (
            float(np.exp(np.median(_flat("log_jitter"))) / err_med)
            if fit_jitter and "log_jitter" in idata.posterior else np.nan
        ),
        "MCMC b constrained": bool(
            np.isfinite(shrink.get("b", np.nan)) and shrink.get("b", 0.0) >= 0.5
        ),
        "MCMC r prior hi": (
            float(r_hi) if prior_style == "paper1" else np.nan
        ),
        "MCMC r ceiling frac": (
            float(np.mean(r_s > 0.98 * r_hi)) if prior_style == "paper1"
            else np.nan
        ),
        "MCMC rho prior applied": bool(rho_mu is not None),
        "MCMC rho prior mu [g/ccm]": (
            float(rho_mu) if rho_mu is not None else np.nan
        ),
        "MCMC rho prior source": rho_src,
        "MCMC rho solar source": rho_solar_src,
        "MCMC rho prefer": rho_prefer,
        "MCMC rho file repaired": rho_repaired,
        "MCMC rho column [g/ccm]": _get(["rho [g/ccm]"]),
        "MCMC rho column [rho_sun]": _get(["rho [rho_sun]"]),
        "MCMC rho from logg [rho_sun]": (
            float((10.0 ** (_get(["logg"]) - LOGG_SUN)) / R_star)
            if np.isfinite(_get(["logg"])) and np.isfinite(R_star)
            and R_star > 0 else np.nan
        ),
        f"{method} Period [d]": P_tls,
        f"{method} TC [BTJD]": T0_tls,
        f"{method} Dur [hrs]": dur_hr,
        f"{method} RP_RS": ror_tls,
    })

    summary = pd.DataFrame([rec])
    out_csv = os.path.join(outdir, f"{save_name}_jaxoplanet_posterior.csv")
    summary.to_csv(out_csv, index=False)
    if verbose:
        print(f"  [save] {out_csv}")

    if save_idata:
        out_nc = os.path.join(outdir, f"{save_name}_jaxoplanet_idata.nc")
        idata.to_netcdf(out_nc)
        if verbose:
            print(f"  [save] {out_nc}")

        # -------------------------------------------------------------- figures
    def _make_figures() -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Build every requested diagnostic figure.

        Nested so it closes over the posterior arrays, the exposure-integrated
        model evaluator, and the prepared data. Nothing here alters the fit or
        the summary numbers.

        Returns
        -------
        figs : dict
            Matplotlib Figure handles keyed by tag.
        paths : dict
            Saved file paths keyed by the same tags.
        """
        figs: Dict[str, Any] = {}
        paths: Dict[str, str] = {}

        if point_estimate == "maxl":
            med = {
                "t0": float(t0_s[i_best]), "period": float(P_s[i_best]),
                "duration": float(D_s[i_best]), "r": float(r_s[i_best]),
                "b": float(b_s[i_best]), "f0": float(f0_s[i_best]),
                "u": u_s[i_best],
            }
        else:
            med = {
                "t0": float(np.median(t0_s)),
                "period": float(np.median(P_s)),
                "duration": float(np.median(D_s)),
                "r": float(np.median(r_s)),
                "b": float(np.median(b_s)),
                "f0": float(np.median(f0_s)),
                "u": np.median(u_s, axis=0),
            }

        def _eval(t_arr: np.ndarray, pars: Dict[str, Any]) -> np.ndarray:
            """Exposure-integrated model flux at arbitrary times."""
            return float(pars["f0"]) * (
                1.0
                + np.asarray(
                    _transit(
                        jnp.asarray(np.asarray(t_arr, dtype=np.float64)),
                        pars["t0"], pars["period"], pars["duration"],
                        pars["r"], pars["b"], jnp.asarray(pars["u"]),
                    )
                )
            )

        def _varying(names: Sequence[str]) -> List[str]:
            """
            Keep only posterior variables that exist and actually move.

            This is what makes az.plot_trace safe across ld_mode settings:
            'u' is absent entirely when limb darkening is held fixed, and
            near-constant when ld_sigma is small.
            """
            out: List[str] = []
            for nm in names:
                if nm not in idata.posterior:
                    continue
                arr = np.asarray(idata.posterior[nm].values, dtype=float)
                if arr.size and np.isfinite(np.nanstd(arr)) and np.nanstd(arr) > 0.0:
                    out.append(nm)
            return out

        def _bin_phase(x_hr: np.ndarray, y: np.ndarray, width_hr: float):
            """Uniform phase bins. Returns (centers, means, errors on the mean)."""
            if width_hr <= 0 or x_hr.size == 0:
                return None
            edges = np.arange(x_hr.min(), x_hr.max() + width_hr, width_hr)
            if edges.size < 2:
                return None
            which = np.digitize(x_hr, edges) - 1
            cx, cy, ce = [], [], []
            for k in range(edges.size - 1):
                sel = which == k
                n_k = int(sel.sum())
                if n_k < 2:
                    continue
                cx.append(0.5 * (edges[k] + edges[k + 1]))
                cy.append(float(np.nanmean(y[sel])))
                ce.append(float(np.nanstd(y[sel]) / np.sqrt(n_k)))
            if not cx:
                return None
            return np.array(cx), np.array(cy), np.array(ce)

        def _finish(fig, tag: str) -> None:
            """Save, register, then show or close."""
            path = os.path.join(outdir, f"{save_name}_jaxoplanet_{tag}.png")
            fig.savefig(path, dpi=int(plot_dpi), bbox_inches="tight")
            figs[tag] = fig
            paths[tag] = path
            if show_plots:
                plt.show()
            else:
                plt.close(fig)
            if verbose:
                print(f"  [save] {path}")

        # ---------------------------------------------- 1. phase-folded fit
        if make_plots:
            # Match the fit window so no fitted cadence is invisible in the
            # figure. A cropped panel hides exactly the out-of-window outliers
            # that drive the jitter term.
            _fw = float(window_durations) if window_durations else 5.0
            half_win = float(min(0.5 * med["period"],
                                 max(_fw, 5.0) * med["duration"]))
            ph_grid = np.linspace(-half_win, half_win, 1200)
            model_med = _eval(med["t0"] + ph_grid, med)

            n_band = int(min(n_pred_draws, n_draw))
            idx = rng.choice(n_draw, size=n_band, replace=False)
            band = np.empty((n_band, ph_grid.size), dtype=np.float64)
            for k, j in enumerate(idx):
                band[k] = _eval(
                    med["t0"] + ph_grid,
                    {"t0": float(t0_s[j]), "period": float(P_s[j]),
                     "duration": float(D_s[j]), "r": float(r_s[j]),
                     "b": float(b_s[j]), "f0": float(f0_s[j]), "u": u_s[j]},
                )
            band_lo = np.percentile(band, 15.865, axis=0)
            band_hi = np.percentile(band, 84.135, axis=0)

            ph_data = (t - med["t0"] + 0.5 * med["period"]) % med["period"] \
                - 0.5 * med["period"]
            resid = f - _eval(t, med)
            sig_resid = float(np.nanstd(resid))
            if not np.isfinite(sig_resid) or sig_resid <= 0:
                sig_resid = 1.0

            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(7.0, 6.0), sharex=True,
                gridspec_kw={"height_ratios": [3, 1]},
            )
            ax1.errorbar(24 * ph_data, f, e, fmt=".", ms=3, color="0.70",
                         ecolor="0.88", zorder=1, label="NEMESIS detrended")
            binned = _bin_phase(24 * ph_data, f, float(phase_bin_minutes) / 60.0)
            if binned is not None:
                bx, by, be = binned
                ax1.errorbar(bx, by, be, fmt="o", ms=4, color="k", ecolor="k",
                             lw=1.0, zorder=4,
                             label=f"{phase_bin_minutes:.0f} min bins")
            ax1.fill_between(24 * ph_grid, band_lo, band_hi, color="crimson",
                             alpha=0.30, lw=0, zorder=2, label="68% credible")
            ax1.plot(24 * ph_grid, model_med, color="crimson", lw=1.6, zorder=3,
                     label="posterior median")
            ax1.set_ylabel("Normalized flux", fontsize=12)
            ax1.legend(fontsize=8, loc="lower right")
            ax1.set_title(
                f"{save_name}   P = {med['period']:.6f} d   "
                f"Rp/Rs = {med['r']:.5f}   b = {med['b']:.3f}   "
                f"[{point_estimate}]",
                fontsize=11,
            )
            ax2.axhline(0.0, color="crimson", lw=1.0, zorder=3)
            ax2.errorbar(24 * ph_data, resid, e, fmt=".", ms=3, color="0.70",
                         ecolor="0.88", zorder=1)
            ax2.set_ylim(-5 * sig_resid, 5 * sig_resid)
            ax2.set_yticks(np.array([-3, 0, 3]) * sig_resid)
            ax2.set_yticklabels(["-3\u03c3", "0", "+3\u03c3"])
            ax2.set_ylabel("Residual", fontsize=12)
            ax2.set_xlabel(r"Hours from $T_0$", fontsize=12)
            ax2.set_xlim(-24 * half_win, 24 * half_win)
            fig.tight_layout(pad=1.0)
            _finish(fig, "fit")

        # ------------------------------------------- 2. time-series overlay
        if make_plots and make_timeseries:
            model_all = _eval(t_all, med)
            resid_all = f_all - model_all
            s_all = float(np.nanstd(resid_all))
            if not np.isfinite(s_all) or s_all <= 0:
                s_all = 1.0

            fig, (axA, axB) = plt.subplots(
                2, 1, figsize=(11.0, 5.5), sharex=True,
                gridspec_kw={"height_ratios": [3, 1]},
            )
            axA.errorbar(t_all, f_all, e_all, fmt=".", ms=2.5, color="0.72",
                         ecolor="0.90", zorder=1, label="NEMESIS detrended")
            if not np.all(in_window):
                axA.plot(t_all[in_window], f_all[in_window], ".", ms=2.5,
                         color="0.25", zorder=2, label="cadences fit")
            # Break the model line across data gaps. Without this, matplotlib
            # draws a chord from the last cadence before a downlink (which can
            # be mid-transit) to the first cadence after it.
            t_plot = t_all.astype(np.float64)
            m_plot = np.asarray(model_all, dtype=np.float64)
            brk = np.where(np.diff(t_plot) > 10.0 * exposure_days)[0]
            if brk.size:
                t_plot = np.insert(t_plot, brk + 1, np.nan)
                m_plot = np.insert(m_plot, brk + 1, np.nan)
            axA.plot(t_plot, m_plot, color="crimson", lw=1.1, zorder=3,
                     label=f"{point_estimate} model")

            k_lo = int(np.floor((float(t_all.min()) - med["t0"]) / med["period"]))
            k_hi = int(np.ceil((float(t_all.max()) - med["t0"]) / med["period"]))
            tts = med["t0"] + np.arange(k_lo, k_hi + 1) * med["period"]
            tts = tts[(tts >= t_all.min()) & (tts <= t_all.max())]
            if 0 < tts.size <= 60:
                for k_tt, tt_ in enumerate(tts):
                    axA.axvline(
                        tt_, color="tab:blue", ls=":", lw=0.9, alpha=0.75,
                        zorder=0,
                        label="predicted mid-transit" if k_tt == 0 else None,
                    )
            axA.set_ylabel("Normalized flux", fontsize=12)
            axA.legend(fontsize=8, loc="lower right", ncol=2)
            axA.set_title(
                f"{save_name}   {int(t.size)} of {int(t_all.size)} cadences fit"
                f"   {tts.size} transits in baseline",
                fontsize=11,
            )
            axB.axhline(0.0, color="crimson", lw=1.0, zorder=3)
            axB.errorbar(t_all, resid_all, e_all, fmt=".", ms=2.5, color="0.72",
                         ecolor="0.90", zorder=1)
            axB.set_ylim(-5 * s_all, 5 * s_all)
            axB.set_yticks(np.array([-3, 0, 3]) * s_all)
            axB.set_yticklabels(["-3\u03c3", "0", "+3\u03c3"])
            axB.set_ylabel("Residual", fontsize=12)
            axB.set_xlabel("Time [BTJD]", fontsize=12)
            fig.tight_layout(pad=1.0)
            _finish(fig, "timeseries")

        # -------------------------------------------------- 3. trace plot
        # Forced on when the convergence gate fails: that is exactly the case
        # where the chains need looking at.
        if bool(make_trace) or (not converged):
            default_tv = ["t0", "period", "duration", "r", "b",
                          "u", "f0", "log_jitter"]
            tv = _varying(list(trace_vars) if trace_vars is not None
                          else default_tv)
            if not tv:
                warnings.warn(
                    "trace plot skipped: no requested variable both exists in "
                    "the posterior and varies. With ld_mode='fixed' there is "
                    "no 'u' site to trace.",
                    RuntimeWarning,
                )
            else:
                try:
                    axes = az.plot_trace(
                        idata, var_names=tv, compact=True,
                        divergences="bottom", figsize=(9.0, 1.9 * len(tv)),
                    )
                except (KeyError, ValueError, TypeError):
                    axes = az.plot_trace(
                        idata, var_names=tv, compact=True,
                        figsize=(9.0, 1.9 * len(tv)),
                    )
                fig_tr = np.ravel(np.asarray(axes, dtype=object))[0].get_figure()
                fig_tr.suptitle(
                    f"{save_name}   r_hat max {rhat_max:.4f}   "
                    f"ess_bulk min {ess_min:.0f}   divergences {n_div}",
                    fontsize=11,
                )
                fig_tr.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
                _finish(fig_tr, "trace")

        # ------------------------------------------------- 4. corner plot
        if make_corner:
            try:
                import corner
            except ImportError:
                warnings.warn(
                    "make_corner=True but the corner package is not installed. "
                    "Skipping the corner plot.",
                    RuntimeWarning,
                )
            else:
                t0_ref = float(np.floor(np.median(t0_s)))
                t0_label = f"T0 - {t0_ref:.0f} [d]"
                cols: Dict[str, np.ndarray] = {
                    "P [d]": P_s,
                    t0_label: t0_s - t0_ref,
                    "T14 [hr]": D_s * 24.0,
                    "Rp/Rs": r_s,
                    "b": b_s,
                    "a/Rs": aRs_s,
                }
                if "u" in idata.posterior:
                    cols["u1"] = u_s[:, 0]
                    cols["u2"] = u_s[:, 1]
                if corner_vars is not None:
                    cols = {k: v for k, v in cols.items() if k in corner_vars}

                # Search-stage values, drawn as reference lines so the shift
                # away from the prior center is visible.
                truth_map: Dict[str, Optional[float]] = {
                    "P [d]": P_tls,
                    t0_label: T0_tls - t0_ref,
                    "T14 [hr]": dur_hr,
                    "Rp/Rs": ror_tls,
                    "b": None,
                    "a/Rs": _get(["a_over_R_star"]),
                    "u1": float(u_const[0]),
                    "u2": float(u_const[1]),
                }

                labels = list(cols.keys())
                arr = np.column_stack([cols[k] for k in labels])
                keep_col = np.nanstd(arr, axis=0) > 0
                labels = [lab for lab, m in zip(labels, keep_col) if m]
                arr = arr[:, keep_col]

                if arr.shape[1] < 2:
                    warnings.warn(
                        "corner plot skipped: fewer than two varying "
                        "parameters after filtering.",
                        RuntimeWarning,
                    )
                else:
                    truths: List[Optional[float]] = []
                    for lab in labels:
                        val = truth_map.get(lab)
                        if val is None or not np.isfinite(val):
                            truths.append(None)
                        else:
                            truths.append(float(val))
                    fig_c = corner.corner(
                        arr, labels=labels,
                        truths=truths, truth_color="tab:blue",
                        quantiles=[0.15865, 0.5, 0.84135],
                        show_titles=True, title_fmt=".4g",
                        title_kwargs={"fontsize": 9},
                        label_kwargs={"fontsize": 10},
                    )
                    fig_c.suptitle(
                        f"{save_name}   blue lines: {method} search values",
                        fontsize=11,
                    )
                    _finish(fig_c, "corner")

        return figs, paths

    fig_handles, fig_paths = _make_figures()
    summary.attrs["figures"] = fig_handles
    for _tag, _path in fig_paths.items():
        summary[f"Fig {_tag}"] = _path
    if fig_paths:
        # Rewrite so the catalog row carries its own figure paths.
        summary.to_csv(out_csv, index=False)

    return summary, idata
