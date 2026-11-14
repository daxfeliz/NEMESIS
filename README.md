# NEMESIS

<p align="center">
  <img src="docs/_static/images/nemesis-logo.png"
       alt="NEMESIS project logo"
       width="350">
</p>

**Developed and maintained by [Dax L. Feliz](https://daxfeliz.github.io/)**

[Personal Website](https://daxfeliz.github.io/) |
[GitHub Profile](https://github.com/daxfeliz) |
[NEMESIS Repository](https://github.com/daxfeliz/NEMESIS) |
[Documentation](https://nemesis-tess.readthedocs.io/en/latest/)

**Exoplanet TraNsit SurvEy of Nearby M-dwarfs in TESS FFIs**

NEMESIS is an **open-source Python framework for extracting, processing, and searching time-series photometry from NASA's Transiting Exoplanet Survey Satellite (TESS) full-frame images (FFIs)**, with an emphasis on nearby M-dwarf stars. Rather than requiring a pre-existing light-curve product or a star selected for short-cadence observations, NEMESIS can retrieve TESSCut pixel data from the Mikulski Archive for Space Telescopes (MAST) and generate custom light curves for targets observed in the FFIs.

The framework combines background estimation, centroid measurements, aperture photometry, nearby-source contamination treatment, spacecraft-quality and pointing-jitter screening, and **pixel-level decorrelation (PLD)** to mitigate instrumental systematics. It then supports blind searches for periodic transit signals using **Box-fitting Least Squares (BLS)** and **Transit Least Squares (TLS)**. Its outputs include processed light curves, fitted search parameters, and multi-page diagnostic/vetting reports incorporating **LEO-Vetter** measurements. The broader NEMESIS II survey uses the same detection and automated-vetting chain in transit-injection and orbit-scrambling control experiments to measure survey sensitivity and false-alarm reliability.

NEMESIS was introduced in [Feliz et al. (2021, *The Astronomical Journal*)](https://iopscience.iop.org/article/10.3847/1538-3881/abedb3). Its follow-up study, **NEMESIS II (Feliz et al., in preparation)**, describes a search of 213,658 single-sector light curves of 113,047 M dwarfs in TESS Primary Mission Sectors 1–26. The manuscript is **not yet posted or peer-reviewed**; posting to arXiv and submission to *The Astronomical Journal* are planned. Reported transit signals and planet candidates are not, by themselves, confirmed planets.

<!-- To add a logo, create docs/_static/images/nemesis-logo.png and uncomment the next line. -->
<!-- <p align="center"><img src="docs/_static/images/nemesis-logo.png" alt="NEMESIS project logo" width="360"></p> -->

## Project links

- **Documentation:** [NEMESIS on Read the Docs](https://nemesis-tess.readthedocs.io/en/latest/)
- **Source repository:** [github.com/daxfeliz/NEMESIS](https://github.com/daxfeliz/NEMESIS)
- **Developer GitHub profile:** [github.com/daxfeliz](https://github.com/daxfeliz)
- **Developer website:** [daxfeliz.github.io](https://daxfeliz.github.io/)
- **NEMESIS I:** [Feliz et al. (2021), *AJ*, 161, 247](https://iopscience.iop.org/article/10.3847/1538-3881/abedb3)
- **NEMESIS II:** Feliz et al. (in preparation); planned arXiv posting and submission to *The Astronomical Journal* (link to follow)

## Installation

**Python 3.12** is required by the current package metadata. Install from GitHub in a clean virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # macOS/Linux
python -m pip install --upgrade pip
python -m pip install 'nemesis-tess @ git+https://github.com/daxfeliz/NEMESIS.git'
python -m pip check
```

For the optional analysis dependencies (including TRICERATOPS and jaxoplanet), use `nemesis-tess[analysis]`; for all defined optional workflows, use `nemesis-tess[all]`. The separate NEMESIS II jaxoplanet and TRICERATOPS reference wrappers are kept in `examples/reference_wrappers/` and **are not part of the installed public Python API**. See the [installation guide](https://nemesis-tess.readthedocs.io/en/latest/installation.html) for prerequisites, extras, and platform notes.

## Quick start

```python
from pathlib import Path
from nemesis_tess import full_pipeline, default_NEMESIS_pipeline_settings

output_dir = Path.cwd() / "nemesis_output"
output_dir.mkdir(exist_ok=True)

settings = default_NEMESIS_pipeline_settings()
hdu, lc_df, transit_results, run_settings = full_pipeline(
    ID=103633434,  # TOI-1235, a published planet host
    path=str(output_dir) + "/",
    cadence="30 minute",
    Sector=20,
    settings=settings,
)

if lc_df is None or lc_df.empty:
    raise RuntimeError("No usable NEMESIS light curve was produced.")
```

This example downloads observational data and can take several minutes. Output products are written beneath `nemesis_output/`.

## Documentation and function index

- [Getting Started / Installation](https://nemesis-tess.readthedocs.io/en/latest/installation.html)
- [Processing and vetting workflow](https://nemesis-tess.readthedocs.io/en/latest/workflow.html)
- [**API reference** — signatures and descriptions](https://nemesis-tess.readthedocs.io/en/latest/api_reference.html)
- [**Alphabetical function and class index**](https://nemesis-tess.readthedocs.io/en/latest/function_index.html)
- [Worked notebooks](https://nemesis-tess.readthedocs.io/en/latest/index.html) — transit searches, jaxoplanet, TRICERATOPS, stellar rotation, and flares
- [Scientific limitations](https://nemesis-tess.readthedocs.io/en/latest/limitations.html)

The API inventory covers the functions and classes **defined in the installed NEMESIS source**, including internal/legacy helpers. **Not all indexed functions are stable, supported top-level imports**: the main supported entry points are `full_pipeline`, `LC_creation`, `Transit_Pipeline`, `default_NEMESIS_pipeline_settings`, and `NoImageDataError`. The reference pages are generated from source without executing the pipeline. Run `python scripts/generate_api_reference.py` after changing source-level function or class definitions.

## Citation

If your work uses NEMESIS, please cite **[Feliz et al. (2021, *AJ*, 161, 247)](https://doi.org/10.3847/1538-3881/abedb3)** and, once available, **Feliz et al. (2026, NEMESIS II; forthcoming)**. Until NEMESIS II is public, describe it as *in preparation* rather than citing a nonexistent preprint or publication. Please also cite relevant underlying methods and software when applicable (such as Lightkurve, wotan, TLS, and LEO-Vetter). See [Citation guidance](https://nemesis-tess.readthedocs.io/en/latest/citing.html) and [`CITATION.cff`](CITATION.cff).

## Bug reports, questions, and contributions

NEMESIS is open-source software distributed under the **[MIT license](LICENSE)**. Please use [GitHub Issues](https://github.com/daxfeliz/NEMESIS/issues) to report reproducible bugs, ask questions, or request features. Contributions and pull requests are welcome at [the NEMESIS repository](https://github.com/daxfeliz/NEMESIS); see [Contributing](https://nemesis-tess.readthedocs.io/en/latest/contributing.html) for suggested report details and a development workflow. Do not post private credentials or unpublished sensitive data in public issues.

## Reproducibility and development

The file `environment/nemesis2_rusty_pip_freeze.txt` records the Linux/Python 3.12.9 environment used for the NEMESIS II survey. It is a **historical scientific environment snapshot**, not a universal cross-platform installation recipe. For a local development checkout:

```bash
git clone https://github.com/daxfeliz/NEMESIS.git
cd NEMESIS
python -m pip install -e '.[dev]'
```

The documentation is built with Sphinx, MyST, and nbsphinx. Notebooks are published with their **saved outputs** but are **not executed** during the hosted Read the Docs build.
