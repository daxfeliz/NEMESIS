"""Compatibility helpers for legacy optional dependencies."""

from importlib import import_module
from importlib.metadata import version
from types import ModuleType


def load_triceratops() -> ModuleType:
    """
    Import TRICERATOPS with compatibility fixes for PyTransit 2.2.0.

    Restores two removed numerical-library aliases before PyTransit
    is imported. Does not modify installed dependency files.

    Returns
    -------
    ModuleType
        The triceratops.triceratops modeling module.

    Raises
    ------
    RuntimeError
        If the PyTransit version differs from the tested version.
    """
    import numpy as np
    from scipy import integrate

    pytransit_version = version("pytransit")

    if pytransit_version != "2.2.0":
        raise RuntimeError(
            "NEMESIS TRICERATOPS compatibility has been tested "
            f"with PyTransit 2.2.0, not {pytransit_version}."
        )

    # NumPy >= 1.24: restore the removed built-in integer alias.
    if "int" not in vars(np):
        setattr(np, "int", int)

    # SciPy >= 1.14: restore the old trapezoidal integration name.
    if not hasattr(integrate, "trapz"):
        setattr(integrate, "trapz", integrate.trapezoid)

    return import_module("triceratops.triceratops")
