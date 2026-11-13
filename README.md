# NEMESIS

NEMESIS (Exoplanet TraNsit SurvEy of Nearby M-dwarfs in TESS FFIS) builds
light curves from TESS full-frame image cutouts, removes spacecraft
systematics with pixel level decorrelation, and searches the detrended light
curves for transiting planets with BLS and TLS. It is the pipeline behind
NEMESIS I (Feliz et al. 2021, AJ, 161, 247) and NEMESIS II (Feliz et al.,
in preparation).

## Install

Python 3.10 or newer is required; NEMESIS II ran on Python 3.12.

```bash
pip install "nemesis-tess @ git+https://github.com/daxfeliz/NEMESIS.git"
```

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

## Citing

Please cite Feliz et al. (2021, AJ, 161, 247) and NEMESIS II (Feliz et al.,
in preparation), along with the tools NEMESIS builds on, in particular
Lightkurve (Lightkurve Collaboration 2018), wotan (Hippke et al. 2019),
TLS (Hippke & Heller 2019) and LEO-Vetter (Kunimoto et al. 2025).

## License

MIT. See `LICENSE`.
