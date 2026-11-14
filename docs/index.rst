NEMESIS documentation
=====================

**NEMESIS** (Exoplanet TraNsit SurvEy of Nearby M-dwarfs in TESS FFIs)
is an open-source Python framework for extracting custom time-series
photometry from TESS full-frame images, correcting instrumental effects
with pixel-level decorrelation, searching for transits with BLS and TLS,
and producing diagnostic reports with automated LEO-Vetter measurements.

The NEMESIS II survey additionally uses *separate* jaxoplanet and
TRICERATOPS analysis wrappers and injection-recovery / orbit-scrambling
controls. The separate wrappers shown in the tutorials are **not installed
public APIs** of ``nemesis_tess``. See :doc:`workflow` for the distinctions.

Project and developer
---------------------

NEMESIS is developed and maintained by
`Dax L. Feliz <https://daxfeliz.github.io/>`_.

- `Developer website <https://daxfeliz.github.io/>`_
- `GitHub profile <https://github.com/daxfeliz>`_
- `NEMESIS source code <https://github.com/daxfeliz/NEMESIS>`_


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   workflow
   limitations

.. toctree::
   :maxdepth: 1
   :caption: Example notebooks

   notebooks/01_nemesis_leovetter
   notebooks/02_jaxoplanet_refit
   notebooks/03_triceratops_validation
   notebooks/04_rotation_lomb_scargle
   notebooks/05_flare_threshold_search

.. toctree::
   :maxdepth: 1
   :caption: Reference

   api_reference
   function_index

.. toctree::
   :maxdepth: 1
   :caption: Project

   citing
   contributing

.. note::

   Read the Docs displays **saved notebook results** without executing
   network downloads, transit searches, MCMC fits, or TRICERATOPS runs.
