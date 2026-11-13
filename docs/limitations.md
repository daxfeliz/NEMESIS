# Limitations, security, and scientific interpretation

**Status.** This is a proposed documentation/tutorial addition, not a
new supported API surface. The scientific `pipeline.py` and package
metadata are unchanged. The 3 external NEMESIS II wrappers live in
`examples/reference_wrappers/`, not in `nemesis_tess` itself.

**jaxoplanet modeling.** `nemesis_jaxoplanet_refit.py` is now included as
supplied. The tutorial uses its genuine limb-darkened NUTS model, but a
short demonstration does not certify its runtime or posterior quality. The
wrapper requires valid limb-darkening inputs (`u_fixed` or model columns),
and its stellar-density prior prints unit cross-checks. Inspect R-hat, ESS,
divergences, and radius-prior truncation on real science fits.

**TRICERATOPS security.** The supplied wrapper temporarily disables TLS
certificate verification for `stev.oapd.inaf.it` during some TRILEGAL
requests. This weakens server authentication. Prefer a trustworthy
pre-staged TRILEGAL population or a proper verified certificate chain.
Do not present these network workarounds as a general-purpose standard
without review. The notebook is opt-in and off by default.

**FPP interpretation.** Reported FPP/NFPP depend on aperture geometry,
field-star priors, transit assumptions, and Monte Carlo draws. A 1-trial
smoke run is not a disposition; at 30-minute cadence the wrapper
intentionally does not use TRICERATOPS's most stringent planet-validation
tier without better-cadence evidence.

**Rotation.** Use photometry retaining stellar variability (`PLD Flux` as
in the supplied showcase), not necessarily planet-search detrending.
A ~4.1-day period peak can have harmonics and activity evolution.

**Flares.** The PDF example is an introductory threshold demonstration,
not the original FINDFLARE implementation. The source notebook contained
a plotting typo and an outlier threshold that could clip positive flares;
these are corrected/documented rather than duplicated without explanation.
Global MAD thresholds on variable stars require careful interpretation.

**Documentation build.** Read the Docs renders notebooks without executing
MAST downloads, NUTS sampling, TRICERATOPS population synthesis or plotting.
All scientific runs require real input data and a separate local test.
