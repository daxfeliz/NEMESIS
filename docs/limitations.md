# Scientific scope, limitations, and security

**Package scope.** The installed `nemesis_tess` package provides the main NEMESIS extraction, detrending, BLS/TLS transit-search, and vetting-report pipeline. The jaxoplanet and TRICERATOPS reference modules under `examples/reference_wrappers/` illustrate subsequent NEMESIS II characterization stages but are **not installed public APIs**. The [API inventory](api_reference.rst) distinguishes supported entry points from internal and legacy functions.

**Detection and vetting.** A threshold-crossing event, a NEMESIS planet-candidate detection, a statistically favored planetary scenario, and a validated/confirmed planet are different concepts. The NEMESIS II manuscript uses automatic LEO-Vetter diagnostics to label detections, while injection/recovery and orbit-scrambling provide *population-level* sensitivity and false-alarm information; neither constitutes independent confirmation of a planet. A single-sector search is primarily sensitive to short orbital periods and can miss weaker planets in multi-planet systems.

**jaxoplanet modeling.** `nemesis_jaxoplanet_refit.py` is supplied in the reference scripts. Short tutorial chains are intended to demonstrate the workflow, not to certify a posterior. Inspect R-hat, effective sample sizes, divergences, radius-prior truncation, and appropriate stellar-parameter priors before reporting scientific parameters.

**TRICERATOPS security.** The supplied reference wrapper contains a narrowly scoped workaround that temporarily disables TLS certificate verification for some TRILEGAL requests to `stev.oapd.inaf.it`. This weakens server authentication; prefer a verified certificate chain or trusted pre-staged background populations and review this path before adopting it for production or untrusted networks.

**FPP/NFPP interpretation.** These probabilities depend on aperture geometry, background-star priors, observational cadence, the flux and depth supplied, and Monte Carlo sampling. A one-trial smoke test is **not** a scientific disposition. The NEMESIS II draft does not claim validation from 30-minute-cadence TRICERATOPS probabilities alone.

**Rotation and flares.** For stellar rotation, analyze flux that retains rotational modulation rather than assuming a planet-search detrending preserves it. The simple threshold-based flare notebook is an educational demonstration, **not** a validated FINDFLARE implementation; real flare studies need a noise model, detection efficiency, and attention to false positives.

**Documentation builds.** Read the Docs renders saved notebook outputs without rerunning MAST downloads, NUTS fitting, or TRICERATOPS. Online scientific services may be unavailable independently of NEMESIS's Python environment.
