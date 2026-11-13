# NEMESIS II scientific reference wrappers

These scripts are included as tutorial reference implementations of the
NEMESIS II science workflow. The TRICERATOPS wrapper has been adapted to
call `nemesis_tess.compat.load_triceratops()`; unlike the original HPC script,
it is **not** a byte-for-byte unchanged copy.

They are **not** installed as part of `nemesis_tess` and should not be treated
as stable public APIs.

- `nemesis_jaxoplanet_refit.py`: actual JAX/NumPyro/`jaxoplanet` transit
  model. Primary API `refine_transit_parameters(lightcurve, bestfit, ...)`.
  Source uses NumPyro NUTS, circular `TransitOrbit`, quadratic limb
  darkening, exposure supersampling, and optional stellar-density prior.
- `run_jaxoplanet_batch.py`: production disBatch worker. Imports the
  refit function above. Includes machine-specific code paths, CPU thread
  configuration, output saving, quality-aware resume and status tables.
- `nemesis_triceratops.py`: statistical validation wrapper requiring a
  retained NEMESIS pixel mask and TESScut FITS file. Converts target flux
  units to the aperture units used by TRICERATOPS and can use a verified
  jaxoplanet posterior's ephemeris. Contains a site-specific TLS-verification
  workaround that needs review before being recommended broadly.

To import the refit wrapper in a checkout, prepend this folder to
`sys.path` as demonstrated in notebook 02 (or run from a path configured
to include it). Keep upstream HPC script copies separately if you need
machine-specific changes; do not modify these tutorial snapshots silently.
