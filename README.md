# NEMESIS

NEMESIS (Exoplanet TraNsit SurvEy of Nearby M-dwarfs in TESS FFIS) builds
light curves from TESS full-frame image cutouts, removes spacecraft
systematics with pixel level decorrelation, and searches the detrended light
curves for transiting planets with BLS and TLS. It is the pipeline behind
NEMESIS I (Feliz et al. 2021, AJ, 161, 247) and NEMESIS II (Feliz et al.,
in preparation).

## Install

Python 3.12 is required by the current package metadata
(`requires-python = ">=3.12,<3.13"`). Other Python versions have not
been validated for this pinned dependency set.

```bash
pip install "nemesis-tess @ git+https://github.com/daxfeliz/NEMESIS.git"
```

For the optional TRICERATOPS analysis environment, use the validation extra:

```bash
python -m pip install "nemesis-tess[validation] @ git+https://github.com/daxfeliz/NEMESIS.git"
```

This extra constrains TRICERATOPS, PyTransit and setuptools to the tested
legacy versions. The reference validation wrapper in
`examples/reference_wrappers/` calls `nemesis_tess.compat.load_triceratops()`
to address obsolete PyTransit imports, but the reference wrapper **is not
installed** as a NEMESIS public API by `pip install`. Clone the repository
when following that tutorial. Direct third-party TRICERATOPS imports are
not protected by the NEMESIS compatibility helper.

## Quick start

```python
import os
from nemesis_tess import full_pipeline, default_NEMESIS_pipeline_settings

settings = default_NEMESIS_pipeline_settings()
hdu, LC_df, transit_results, run_settings = full_pipeline(
    ID=103633434, path=os.getcwd() + "/", cadence="30 minute", Sector=20, settings=settings)
```

`full_pipeline` downloads a TESScut cutout, builds and detrends the light
curve, runs BLS and TLS, and writes light curves, best-fit tables and
multi-page vetting reports under the folder you pass.

## Reproducing the NEMESIS II environment

`environment/nemesis2_rusty_pip_freeze.txt` lists every package version in
the environment that produced the NEMESIS II results (Linux, Python 3.12.9).
An independent rebuild from this list was verified to match the original
package for package. To rebuild it:

```bash
python3.12 -m venv nemesis2-env
source nemesis2-env/bin/activate
python -m pip install --no-deps -r environment/nemesis2_rusty_pip_freeze.txt
```

## Development install

```bash
git clone https://github.com/daxfeliz/NEMESIS.git
cd NEMESIS
pip install -e ".[dev]"
```

## Tutorials and Read the Docs

The public docs source is in `docs/` and the five Jupyter examples in
`docs/notebooks/`. The separate scientific reference wrappers live in
`examples/reference_wrappers/`, including the actual `nemesis_jaxoplanet_refit.py`
function used by the NEMESIS II batch worker. These examples are *not* new
exports from the installed `nemesis_tess` library.

The tutorials demonstrate (1) NEMESIS + LEO-Vetter BLS/TLS reports,
(2) follow-up jaxoplanet/NumPyro transit refinement,
(3) TRICERATOPS FPP/NFPP, (4) Lomb–Scargle rotation of TIC 350545089,
and (5) the simple flare-threshold demonstration from the showcase PDF
on TIC 260506296 (the code-executed TIC rather than the PDF heading).

To build locally:

```bash
python -m pip install -r docs/requirements.txt
python scripts/validate_notebooks.py
python -m sphinx -b html docs docs/_build/html
```

Read the Docs renders notebooks without executing remote science runs.


## Citing

Please cite Feliz et al. (2021, AJ, 161, 247) and NEMESIS II (Feliz et al.,
in preparation), along with the tools NEMESIS builds on, in particular
Lightkurve (Lightkurve Collaboration 2018), wotan (Hippke et al. 2019),
TLS (Hippke & Heller 2019) and LEO-Vetter (Kunimoto et al. 2025).

## License

MIT. See `LICENSE`.
