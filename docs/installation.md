# Installation

The initial scientific reference configuration is Python **3.12**, matching
`environment/nemesis2_rusty_pip_freeze.txt`. It has been tested for a complete
single-sector NEMESIS BLS/TLS run on an Apple Silicon Mac.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'nemesis-tess[all] @ git+https://github.com/daxfeliz/NEMESIS.git'
python -m pip check
```

For the notebooks and the reference wrappers, clone the source repository:

```bash
git clone https://github.com/daxfeliz/NEMESIS.git
cd NEMESIS
python -m pip install -e '.[all,dev]'
python -m pip install -r docs/requirements.txt
jupyter lab
```

The JAX and TRICERATOPS extras install their dependencies, **not** the
three NEMESIS II wrapper modules under `examples/reference_wrappers/`.
Those modules are included **as source files in a Git clone** and the
notebooks import them from there. They are not exported as part of the
`nemesis_tess` installed library. See [Workflow](workflow.md).

If pip on macOS Sonoma raises the previously observed PyPI trust-store error,
`PIP_USE_DEPRECATED=legacy-certs` preserves certificate verification while
using pip's legacy trust path. Do not use `--trusted-host` by default.

For reproducibility, record `python --version`, `python -m pip freeze`, the
NEMESIS Git commit, the TIC/sector/cadence, and all nondefault settings.
