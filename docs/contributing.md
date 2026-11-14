# Bug reports, questions, and contributions

NEMESIS is open source under the [MIT license](https://github.com/daxfeliz/NEMESIS/blob/main/LICENSE). Source code and development history are available at [github.com/daxfeliz/NEMESIS](https://github.com/daxfeliz/NEMESIS).

## Ask a question or report a bug

Open a [GitHub Issue](https://github.com/daxfeliz/NEMESIS/issues). For a reproducible bug report, include:

1. The installed NEMESIS version or Git commit, Python version, OS/architecture, and relevant dependency versions.
2. The smallest runnable example, including TIC ID, sector/cadence, and any non-default pipeline settings when relevant.
3. The complete error traceback and the expected versus observed behavior.
4. Whether the issue involves a remote service (such as MAST or VizieR), local files, a scientific result, or documentation.

Avoid attaching credentials, access tokens, sensitive file paths, or data that you do not have permission to share publicly.

## Contribute a change

Pull requests are welcome. For major scientific changes, open an issue first to discuss expected behavior and validation criteria. Keep unrelated scientific, dependency, and documentation changes separate where possible.

```bash
git clone https://github.com/daxfeliz/NEMESIS.git
cd NEMESIS
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python scripts/validate_notebooks.py
python scripts/generate_api_reference.py --check
```

If you add or rename a top-level function or class in the installed package, regenerate the source-derived API pages before committing:

```bash
python scripts/generate_api_reference.py
```

The documentation is hosted on [Read the Docs](https://nemesis-tess.readthedocs.io/en/latest/). For local HTML rendering, install `docs/requirements.txt` and Pandoc, then run `python -m sphinx -E -a -b html docs docs/_build/html`.

Scientific changes should include a minimal test or reproducible comparison, with relevant input data and numerical acceptance criteria. Please don't change survey thresholds or default scientific behavior solely to eliminate a software warning without explaining the scientific impact.
