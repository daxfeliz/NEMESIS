NEMESIS documentation
=====================

**NEMESIS** (Exoplanet TraNsit SurvEy of Nearby M-dwarfs in TESS FFIs)
extracts and detrends TESS light curves, runs BLS and TLS, and generates
vetting reports including LEO-Vetter diagnostics.

The NEMESIS II analysis flow additionally uses a separate jaxoplanet
refit module and a separate TRICERATOPS validation wrapper. Those modules
are not automatically part of the ``nemesis_tess`` install.

.. toctree::
   :maxdepth: 2
   :caption: Guides

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

.. note::

   Notebooks are **not executed** by Read the Docs. Download and run them
   locally with the appropriate dependencies and input data.
