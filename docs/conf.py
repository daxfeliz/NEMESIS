"""Sphinx configuration for the public NEMESIS documentation."""

project = "NEMESIS"
author = "Dax L. Feliz"
release = "1.96"
version = "1.96"

extensions = ["myst_parser", "nbsphinx"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"
exclude_patterns = ["_build", "**/.ipynb_checkpoints"]
html_theme = "sphinx_rtd_theme"
html_title = "NEMESIS — TESS M-dwarf transit survey"

# Explicitly forbid runtime use of remote services, notebook fits, or local data.
nbsphinx_execute = "never"
nbsphinx_allow_errors = False
nbsphinx_kernel_name = "python3"

# Avoid evaluating pipeline.py at documentation build time. The package is
# intentionally not installed in the RTD docs environment.
