"""
nemesis_tess
============

NEMESIS: photometry, detrending and transit search of M dwarfs in TESS
full-frame images (Feliz et al. 2021, AJ, 161, 247; NEMESIS II, Feliz et al.,
in preparation).

``nemesis_tess.pipeline`` is the NEMESIS pipeline module. Its main entry
points are re-exported here, so both of these work:

    from nemesis_tess import full_pipeline
    from nemesis_tess.pipeline import full_pipeline

The package version is read from the installed metadata, which setuptools
takes from ``__version__`` in pipeline.py at install time.
"""
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

try:
    __version__: str = _distribution_version("nemesis-tess")
except PackageNotFoundError:  # source tree that was never installed
    __version__ = "0+unknown"

from .pipeline import (  # noqa: E402
    LC_creation,
    NoImageDataError,
    Transit_Pipeline,
    __version_message__,
    default_NEMESIS_pipeline_settings,
    full_pipeline,
)

__all__ = [
    "__version__",
    "__version_message__",
    "full_pipeline",
    "LC_creation",
    "Transit_Pipeline",
    "default_NEMESIS_pipeline_settings",
    "NoImageDataError",
]
