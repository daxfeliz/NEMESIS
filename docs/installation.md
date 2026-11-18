# Installation

NEMESIS is currently distributed directly from its [GitHub repository](https://github.com/daxfeliz/NEMESIS). The distribution is named **`nemesis-tess`** and its Python import is **`nemesis_tess`**. The current package metadata requires **Python 3.12** (`>=3.12,<3.13`); other versions have not been validated with the pinned scientific dependencies.

## Requirements

- Python 3.12 and its standard `venv` module
- `git` (pip retrieves the project and one pinned scientific dependency from Git repositories)
- Internet access to GitHub, PyPI, and—when processing TESS data—MAST and any external catalog services required by the selected workflow
- Sufficient disk space for downloaded cutouts, processing products, and vetting reports

The package has been exercised on macOS and in the Linux scientific environment described by the NEMESIS II environment snapshot. Other systems and architectures may require compatible compiled dependency wheels; their support is not guaranteed.

## Standard installation

Create a clean virtual environment and install the core package:

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # macOS or Linux
python -m pip install --upgrade pip
python -m pip install 'nemesis-tess @ git+https://github.com/daxfeliz/NEMESIS.git'
python -m pip check
```

On Windows PowerShell, activation is `.venv\Scripts\Activate.ps1` instead of `source .venv/bin/activate`. The scientific dependency set has not been validated on Windows; the activation command does not imply full platform support.

Confirm that the principal imports work:

```bash
python - <<'PY'
from importlib.metadata import version
from nemesis_tess import full_pipeline, default_NEMESIS_pipeline_settings

print("NEMESIS:", version("nemesis-tess"))
print("Pipeline import: PASS")
PY
```

This checks installation and imports only. An end-to-end test additionally downloads and analyzes TESS data; see [Notebook 01](notebooks/01_nemesis_leovetter.ipynb).

## Optional workflows

Install one of the optional dependency groups as needed:

```bash
# jaxoplanet / NumPyro analysis libraries:
python -m pip install 'nemesis-tess[modeling] @ git+https://github.com/daxfeliz/NEMESIS.git'

# TRICERATOPS and its compatible legacy dependencies:
python -m pip install 'nemesis-tess[validation] @ git+https://github.com/daxfeliz/NEMESIS.git'

# Modeling and validation together:
python -m pip install 'nemesis-tess[analysis] @ git+https://github.com/daxfeliz/NEMESIS.git'

# All defined extras (larger installation):
python -m pip install 'nemesis-tess[all] @ git+https://github.com/daxfeliz/NEMESIS.git'
```

The separate jaxoplanet/NumPyro and TRICERATOPS **reference wrapper modules are not exported by the installed `nemesis_tess` package**. The extras install their *dependencies*, not the scripts under `examples/reference_wrappers/`. Clone the repository to run those example workflows. The TRICERATOPS wrapper calls `nemesis_tess.compat.load_triceratops()` to accommodate legacy PyTransit imports; importing TRICERATOPS independently does not apply that compatibility helper.

## Clone the source and use the notebooks

```bash
git clone https://github.com/daxfeliz/NEMESIS.git
cd NEMESIS
python -m pip install -e '.[all,dev]'
python -m pip install -r docs/requirements.txt
jupyter lab
```

Read [Workflow](workflow.md) before running the follow-up characterization examples. Some require substantial CPU time, original TESS cutouts, aperture masks, or external catalog queries. The short demonstration fits and single-trial FPP calculations are not substitutes for scientific-quality analyses.

## Build the documentation locally

Documentation rendering uses Sphinx and `nbsphinx`, which also requires the **Pandoc executable**. Pandoc is not itself installed by `docs/requirements.txt`; install it using the [official Pandoc instructions](https://pandoc.org/installing.html) or an appropriate distribution for your operating system, then check `pandoc --version`.

```bash
python -m pip install -r docs/requirements.txt
python scripts/validate_notebooks.py
python scripts/generate_api_reference.py --check
python -m sphinx -E -a -b html docs docs/_build/html
```

Read the Docs obtains Pandoc automatically from `.readthedocs.yaml`. It builds the documentation without running the scientific notebooks (`nbsphinx_execute = "never"`).

## Troubleshooting

**A pip certificate-verification error** generally indicates a local CA/trust-store, proxy, or Python installation issue, not a NEMESIS requirement. Update the system and Python certificate configuration or consult your institution's IT support. Do **not** disable TLS validation or use `--trusted-host` as a routine workaround (meaning Transport Layer Security, not Transit Least Squares, hehehe).

**A TESS or catalog query fails:** MAST, CDS/VizieR, and other archive services can experience outages or rate limits independently of the Python installation. Recheck the service and the original exception. A failed external query does not necessarily imply a pip or NEMESIS installation problem.

**A compiled dependency cannot be installed:** verify Python 3.12, the OS/CPU architecture, and that you are in a fresh environment. The scientific versions are pinned to the NEMESIS II baseline; don't downgrade NumPy, SciPy, or JAX independently without testing the complete workflow.

For reproducibility, record your Python version, `python -m pip freeze`, NEMESIS Git commit, target TIC, sector, cadence, and non-default settings.
