# From blind transit search to statistical validation

The NEMESIS II tutorials follow three scientifically distinct stages:

1. **NEMESIS + integrated LEO-Vetter:** Extract/detrend FFI light curves,
   search with BLS/TLS, and generate vetting reports. LEO-Vetter is already
   run as part of the report; do not duplicate it.
2. **jaxoplanet / NumPyro:** After selecting a candidate, call the actual
   `nemesis_jaxoplanet_refit.refine_transit_parameters` model, supplied in
   `examples/reference_wrappers/nemesis_jaxoplanet_refit.py`. The HPC
   `run_jaxoplanet_batch.py` worker calls this same function using disBatch
   batches and a quality-aware restart policy. That separate model is
   reference source, not automatically installed by `nemesis-tess`.
3. **TRICERATOPS:** Run the supplied `run_triceratops_on_nemesis` wrapper
   with the search best-fit, final LC, exact original aperture mask and
   TESScut FITS. If a verified jaxoplanet posterior is provided, use its
   period, epoch and duration. Use the *search-measured depth*, not the
   geometric jaxoplanet `(Rp/Rs)^2` depth; the wrapper also handles
   NEMESIS target-deblended versus TRICERATOPS aperture-unit flux.

## Preserve the necessary files

Before calling `full_pipeline` in notebook 01, set:

```python
settings['keep_imagedata'] = True
settings['keep_extra_files'] = True
settings['keep_FITS'] = True
settings['keep_extra_transit_files'] = True
```

The final flag is required for retaining the report products, distinct
from preserving FITS and the `*_pix_mask.pkl` files.

## Notebooks

- [01: transit search and LEO-Vetter](notebooks/01_nemesis_leovetter.ipynb)
- [02: actual NEMESIS II jaxoplanet/NumPyro refit](notebooks/02_jaxoplanet_refit.ipynb)
- [03: TRICERATOPS validation](notebooks/03_triceratops_validation.ipynb)
- [04: showcase stellar rotation](notebooks/04_rotation_lomb_scargle.ipynb)
- [05: showcase simple flare detection](notebooks/05_flare_threshold_search.ipynb)

Notebooks 01–03 share `~/nemesis_tutorials/toi1235_s20` for a single-star
example. Notebook 02 uses short tutorial chains **only for technical testing**;
production uses the batch script's settings: `n_chains=4`, `n_warmup=2000`,
`n_samples=3000`, Student-t likelihood with `student_t_nu=4.0`,
`target_accept=0.95`, and default `r_hi_sigmas=10.0`.
Before forwarding its posterior to notebook 03, inspect convergence,
ESS, divergences and radius-prior ceiling occupancy. Notebook 03 is
**opt-in** due to long sampling and external TIC/MAST/TRILEGAL queries.
Its default scientific setting is 20 trials with 5,000,000 samples each;
its 1-trial smoke test does not establish a trustworthy FPP.

## Stellar-variability showcase

The reference PDF's rotation case uses TIC 350545089 (2MASS
J12394672+0410471) and finds a ~4.143-day Lomb-Scargle peak. Our notebook
runs only the relevant TESS Sector 23 and uses `PLD Flux`; the period is
a candidate, to be checked for aliases and spot evolution.

The flare case uses the same simple threshold idea as the PDF, with
`N1=2`, `N2=1`, `N3=3`. The PDF's heading mentions TIC 388857263
(Proxima Centauri), but the *executed code and downloaded sectors* use TIC
260506296, Sectors 1–3. We use that executed TIC. The PDF's `Nsigma=[inf,2]`
clips positive PLD outliers with v1.96; the notebook disables the pipeline's
final outlier-removal step to preserve prospective flares, and corrects a
plotting variable typo. This is **not** a published FINDFLARE implementation.
