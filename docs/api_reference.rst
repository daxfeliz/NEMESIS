API reference
=============

This reference is generated from NEMESIS source syntax and docstrings,
**without importing or running the scientific pipeline**. Definitions,
signatures, and source locations reflect the committed code at generation
time. Details and behavior are described in the linked source docstrings.

The **supported top-level exports** are ``full_pipeline``,
``LC_creation``, ``Transit_Pipeline``,
``default_NEMESIS_pipeline_settings``, and ``NoImageDataError``.
The complete inventory below also includes internal helpers and
legacy-named functions, which are **not promised as a stable public API**.
The NEMESIS II jaxoplanet and TRICERATOPS reference wrappers live in
``examples/reference_wrappers/`` and are not installed package exports.

To update both reference pages after source edits, run
``python scripts/generate_api_reference.py`` at the repository root.

This snapshot contains **229 unique top-level functions/classes**
and **8 class methods**.

.. note::

   An earlier top-level definition is replaced by a later definition
   of the same name at import time: ``nemesis_tess.pipeline._sec_str``.
   The reference lists the final active definition.

Supported top-level exports
---------------------------

.. _api-nemesis-tess-pipeline-default-nemesis-pipeline-settings:

default_NEMESIS_pipeline_settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L26894>`__

**Description:** Return the default settings dictionary for the NEMESIS pipeline.

.. code-block:: python

   def default_NEMESIS_pipeline_settings(return_dict: bool=True): ...

.. _api-nemesis-tess-pipeline-full-pipeline:

full_pipeline
~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L27444>`__

**Description:** End-to-end NEMESIS pipeline: FFI/TPF retrieval → SAP → PLD → detrending → outlier removal → BLS/TLS transit search → vetting reports.

.. code-block:: python

   def full_pipeline(ID, path, cadence, Sector, star=None, settings: dict | None=None, **kwargs): ...

.. _api-nemesis-tess-pipeline-lc-creation:

LC_creation
~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L15409>`__

**Description:** Produce processed TESS light curves for one or more sectors using the NEMESIS pipeline (Feliz et al. 2021, https://arxiv.org/abs/2103.05647).

.. code-block:: python

   def LC_creation(ID, nearby_star_catalog, cadence, Sector, default_settings, **kwargs): ...

.. _api-nemesis-tess-pipeline-noimagedataerror:

NoImageDataError
~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L272>`__

**Description:** Raised when no TESScut/TPF/FFI image data are available for a TIC+Sector+cadence.

.. code-block:: python

   class NoImageDataError

.. _api-nemesis-tess-pipeline-transit-pipeline:

Transit_Pipeline
~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L26625>`__

**Description:** Run BLS and optionally TLS transit searches, produce transit plots, and generate multi-page vetting reports.

.. code-block:: python

   def Transit_Pipeline(ID, nearby_star_catalog, Sector, cadence, hdu, input_LC, default_settings, sap_diag=None): ...

Other module-level functions and classes
----------------------------------------

.. _api-nemesis-tess-pipeline-align-arrays:

align_arrays
~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L21657>`__

**Description:** Align the model flux to the data flux by interpolating the model flux to match the phase values of the data.

.. code-block:: python

   def align_arrays(phase_data, flux_data, phase_model, model_flux): ...

.. _api-nemesis-tess-pipeline-all-transit-times:

all_transit_times
~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L23597>`__

**Description:** Return all predicted mid-transit times within the time span of t.

.. code-block:: python

   def all_transit_times(T0, t, period): ...

.. _api-nemesis-tess-pipeline-append-pdf:

append_PDF
~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L20142>`__

**Description:** See source for parameters and behavior.

.. code-block:: python

   def append_PDF(ID, Sector, cadence, input_files, BLSorTLS, default_settings): ...

.. _api-nemesis-tess-pipeline-apply-nan-to-flagged-rows:

apply_nan_to_flagged_rows
~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L7809>`__

**Description:** NaN non-preserved columns on cadences whose QUALITY\_FLAG indicates flux-corrupting events.

.. code-block:: python

   def apply_nan_to_flagged_rows(df: pd.DataFrame, preserve_columns: Iterable[str], quality_column: str='QUALITY_FLAG') -> pd.DataFrame: ...

.. _api-nemesis-tess-pipeline-applying-mdump-removal:

Applying_Mdump_removal
~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L11805>`__

**Description:** Apply momentum dump and sector jitter masking to a raw SAP light curve.

.. code-block:: python

   def Applying_Mdump_removal(ID, Sector, hdu, SAP_LC, flux, combined_mask, default_settings): ...

.. _api-nemesis-tess-pipeline-asymmetric:

asymmetric
~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L25113>`__

**Description:** FA check: left/right transit asymmetry.

.. code-block:: python

   def asymmetric(metrics, thresholds, pad: int=8, float_round: int=2): ...

.. _api-nemesis-tess-pipeline-bad-fit:

bad_fit
~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L24845>`__

**Description:** FA check: poor fit of the trapezoidal transit model.

.. code-block:: python

   def bad_fit(metrics, thresholds, pad: int=4, float_round: int=2): ...

.. _api-nemesis-tess-pipeline-bad-shape:

bad_shape
~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L24461>`__

**Description:** FA check: flux increases within the putative transit (bad transit shape).

.. code-block:: python

   def bad_shape(metrics, thresholds, pad: int=8, float_round: int=2): ...

.. _api-nemesis-tess-pipeline-bin-ff:

bin_ff
~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L21519>`__

**Description:** No description available in the source docstring.

.. code-block:: python

   def bin_ff(pf, ff, duration, cadence): ...

.. _api-nemesis-tess-pipeline-bin-func:

Bin_func
~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L629>`__

**Description:** This function will produced a light curve that is binned with a weighted bins.

.. code-block:: python

   def Bin_func(time, flux, error, binsize): ...

.. _api-nemesis-tess-pipeline-bin-quaternion-features:

bin_quaternion_features
~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L7244>`__

**Description:** Bin 2-sec quaternion data to LC cadence and compute the 36 QLP feature series following Kunimoto et al. (2022), arXiv:2211.04387.

.. code-block:: python

   def bin_quaternion_features(lc_time: 'np.ndarray', q_time: 'np.ndarray', q1: 'np.ndarray', q2: 'np.ndarray', q3: 'np.ndarray', cadence_days: 'Optional[float]'=None, min_samples: int=10, time_offset: float=0.0) -> 'Tuple[pd.DataFrame, List[str]]': ...

.. _api-nemesis-tess-pipeline-bls-func:

BLS_func
~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L17329>`__

**Description:** Run the Astropy BLS algorithm and assemble all NEMESIS transit search outputs.

.. code-block:: python

   def BLS_func(ID, nearby_star_catalog, Sector, cadence, hdu, time, flux, error, default_settings): ...

.. _api-nemesis-tess-pipeline-calc-flux-contamination:

calc_flux_contamination
~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L5821>`__

**Description:** Compute the flux contamination ratio and its 1-sigma uncertainty for a target star given nearby TIC sources.

.. code-block:: python

   def calc_flux_contamination(nearby_star_catalog, ID: int, N_TESS_pixels: int=2, Zeropoint: float=20.44, default_mag_err: float=0.05) -> tuple: ...

.. _api-nemesis-tess-pipeline-calc-mean-depths:

calc_mean_depths
~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L20796>`__

**Description:** See source for parameters and behavior.

.. code-block:: python

   def calc_mean_depths(time, flux, T0, P, Dur): ...

.. _api-nemesis-tess-pipeline-calc-transit-times:

calc_transit_times
~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L20770>`__

**Description:** Return all mid-transit times within t

.. code-block:: python

   def calc_transit_times(T0, t, period): ...

.. _api-nemesis-tess-pipeline-calculate-semi-major-axis:

calculate_semi_major_axis
~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L11966>`__

**Description:** Calculate the semi-major axis using Kepler's third law.

.. code-block:: python

   def calculate_semi_major_axis(Period, M_star, R_star): ...

.. _api-nemesis-tess-pipeline-cdpp:

CDPP
~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L5601>`__

**Description:** Compute the Combined Differential Photometric Precision (CDPP) of a light curve.

.. code-block:: python

   def CDPP(time, flux, error, method, unit, cad, binsize=1.0 / 24.0): ...

.. _api-nemesis-tess-pipeline-center-cutout:

center_cutout
~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L867>`__

**Description:** For TESS Full Frame Images (FFI), this function will try to center the FFI on the target star, much like how Target Pixel Files (TPF) are centered on the target star.

.. code-block:: python

   def center_cutout(hdu, cutoutsize, cadence): ...

.. _api-nemesis-tess-pipeline-centroid-analysis:

centroid_analysis
~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L22735>`__

**Description:** Compute a centroid motion diagnostic for a NEMESIS TCE and render it on the Page-4 vetting report axes.

.. code-block:: python

   def centroid_analysis(ID, Sector, hdu, LC_df, transit_results_df, ax1, ax2, ax3, ax4, verbose: bool=False): ...

.. _api-nemesis-tess-pipeline-centroid-quadratic:

centroid_quadratic
~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L5447>`__

**Description:** This function computes the quadratic estimate of the centroid in a 2d-array. This method will fit a simple 2D second-order polynomial $P(x, y) = a + bx + cy + dx^2 + exy + fy^2$ to the 3x3 patch of pixels centered on…

.. code-block:: python

   def centroid_quadratic(data, mask, reference_pixel): ...

.. _api-nemesis-tess-pipeline-chases:

chases
~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L24620>`__

**Description:** FA check: events not unique in the local light curve (few transits).

.. code-block:: python

   def chases(metrics, thresholds, pad: int=5, float_round: int=2): ...

.. _api-nemesis-tess-pipeline-check-all-thresholds:

check_ALL_thresholds
~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L25770>`__

**Description:** Run all 17–18 LEO-Vetter FA/FP checks and return an aggregate flag.

.. code-block:: python

   def check_ALL_thresholds(metrics, verbose: bool=False, collect_output_strings: bool=False): ...

.. _api-nemesis-tess-pipeline-check-centroids:

check_centroids
~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L5108>`__

**Description:** This function will compare the location of the centroids produced by the 'SAP' function with the location of the reference pixel.

.. code-block:: python

   def check_centroids(ID, Sector, cadence, reference_pixel, default_settings): ...

.. _api-nemesis-tess-pipeline-check-edge:

check_edge
~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L5996>`__

**Description:** This function is used to check if a polygon contains the edges of a Full Frame Image (FFI) or Target Pixel File (TPF).

.. code-block:: python

   def check_edge(hdu, mypols): ...

.. _api-nemesis-tess-pipeline-check-imports-are-installed:

check_imports_are_installed
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L85>`__

**Description:** Try to import every module (and submodule) that NEMESIS uses in NEMESIS\_pipeline\_new\_temp\_2.py. Prints a compact report and returns (missing, imported) where each is a list of module strings.

.. code-block:: python

   def check_imports_are_installed(verbose=True): ...

.. _api-nemesis-tess-pipeline-check-transit-symmetry:

check_transit_symmetry
~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L21965>`__

**Description:** Assess left/right transit symmetry using uncertainty-normalised (SNR-on-the-asymmetry) tests for both transit depth and fractional transit duration (q\_tran).

.. code-block:: python

   def check_transit_symmetry(metrics, snr=None, n_sigma_depth: float=3.0, n_sigma_qtran: float=5.0) -> tuple: ...

.. _api-nemesis-tess-pipeline-check-wcs:

check_wcs
~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L1012>`__

**Description:** Returns an astropy.wcs.WCS for the input TESS FFI/TPF.

.. code-block:: python

   def check_wcs(hdu): ...

.. _api-nemesis-tess-pipeline-chi:

chi
~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L25210>`__

**Description:** FA check: inconsistent per-transit SNRs (χ² statistic).

.. code-block:: python

   def chi(metrics, thresholds, pad: int=4, float_round: int=2): ...

.. _api-nemesis-tess-pipeline-classify-retrieval-error:

classify_retrieval_error
~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L4175>`__

**Description:** Sort a cutout-retrieval exception into a plain-language cause.

.. code-block:: python

   def classify_retrieval_error(err: BaseException) -> Tuple[str, str]: ...

.. _api-nemesis-tess-pipeline-cleanup:

cleanup
~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L27103>`__

**Description:** Conditionally delete NEMESIS intermediate and ancillary data products at the end of a pipeline run.

.. code-block:: python

   def cleanup(ID, cadence, Sector, default_settings): ...

.. _api-nemesis-tess-pipeline-collect-nearby-gaia-params:

collect_nearby_Gaia_params
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L10554>`__

**Description:** Query Vizier (Gaia DR3 by default) for sources within the TPF/FFI footprint and restrict the returned catalog to those whose (optionally PM-corrected) positions project inside the cutout's rectangular bounding box.

.. code-block:: python

   def collect_nearby_Gaia_params(ID: int, hdu, magnitude_limit: Optional[float]=None, use_dr2: bool=False, pix_scale_arcsec: float=21.0, border_pixels: float=0.5, apply_pm_correction: bool=True, keep_target_regardless: bool=True, verbose: bool=False, downloadpath: Optional[str]=None) -> Tuple[pd.DataFrame, str, float]: ...

.. _api-nemesis-tess-pipeline-collect-nearby-tic-params:

collect_nearby_TIC_params
~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L9506>`__

**Description:** Queries the TESS Input Catalog (TIC) for stellar parameters of the target star and nearby stars within a given radius. The radius is defined by the number of TESS pixels.

.. code-block:: python

   def collect_nearby_TIC_params(ID, N_TESS_pixels, verbose=False, downloadpath=None): ...

.. _api-nemesis-tess-pipeline-compare-to-model:

compare_to_model
~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L21788>`__

**Description:** No description available in the source docstring.

.. code-block:: python

   def compare_to_model(phase, flux, model_phase, model_flux, transit_results_df): ...

.. _api-nemesis-tess-pipeline-compute-prf-aperture-weights:

compute_prf_aperture_weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L7917>`__

**Description:** Compute a PRF-based soft aperture weight map for a TESS TESScut cutout.

.. code-block:: python

   def compute_prf_aperture_weights(hdu: fits.HDUList, reference_pixel: list, nearby_star_catalog: Optional[pd.DataFrame]=None, target_tmag: Optional[float]=None, cutoutsize: int=11, prf_weight_threshold: float=0.01, decontaminate: bool=True, prf_cache_dir: Optional[str]=None, verbose: bool=False) -> Tuple[np.ndarray, np.ndarray]: ...

.. _api-nemesis-tess-pipeline-compute-sde-optimized:

compute_sde_optimized
~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L17071>`__

**Description:** Computes an array of SDE values, ensuring that max(SDE array) matches the classical SDE.

.. code-block:: python

   def compute_sde_optimized(periodogram_power): ...

.. _api-nemesis-tess-pipeline-compute-tce-coverage-metrics:

compute_tce_coverage_metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L23288>`__

**Description:** Compute coverage-related vetting metrics for a single-sector TCE.

.. code-block:: python

   def compute_tce_coverage_metrics(time: np.ndarray, period: float, t0: float, t14_hours: float, min_epochs: int=2) -> Dict[str, float]: ...

.. _api-nemesis-tess-pipeline-consecutive:

consecutive
~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L13210>`__

**Description:** This function will search input data for points that are consecutive.

.. code-block:: python

   def consecutive(data, stepsize=1): ...

.. _api-nemesis-tess-pipeline-convert-tessmag-to-flux:

convert_TessMag_to_Flux
~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L5789>`__

**Description:** This function will convert TESS Input Catalog (TIC) Magnitude into TESS Flux.

.. code-block:: python

   def convert_TessMag_to_Flux(Tmag, Zeropoint=20.44): ...

.. _api-nemesis-tess-pipeline-convert-window-size-in-days-to-points:

convert_window_size_in_days_to_points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L12111>`__

**Description:** This function converts a window size into a window size in number of data points, based on the cadence of timestamps in input time array.

.. code-block:: python

   def convert_window_size_in_days_to_points(window_size_in_days, time): ...

.. _api-nemesis-tess-pipeline-create-box-model:

create_box_model
~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L21450>`__

**Description:** Create boxcar model.

.. code-block:: python

   def create_box_model(times, period, epoch, duration, depth): ...

.. _api-nemesis-tess-pipeline-create-box-model-for-phasefolded:

create_box_model_for_phasefolded
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L21491>`__

**Description:** Create boxcar model.

.. code-block:: python

   def create_box_model_for_phasefolded(pf, period, epoch, duration, depth): ...

.. _api-nemesis-tess-pipeline-create-circular-mask:

create_circular_mask
~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L5238>`__

**Description:** Build a boolean circular aperture mask centered on reference\_pixel.

.. code-block:: python

   def create_circular_mask(median_image: np.ndarray, reference_pixel, radius: float=1.0, snap_to_pixel: bool=False) -> np.ndarray: ...

.. _api-nemesis-tess-pipeline-create-pdf-from-png:

create_PDF_from_PNG
~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L20090>`__

**Description:** See source for parameters and behavior.

.. code-block:: python

   def create_PDF_from_PNG(ID, Sector, BLSorTLS, cadence, input_file, default_settings): ...

.. _api-nemesis-tess-pipeline-current-tesscut-active:

current_tesscut_active
~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L2694>`__

**Description:** Return approximate number of active TESScut/TESSSearch slots.

.. code-block:: python

   def current_tesscut_active(cache_root) -> int: ...

.. _api-nemesis-tess-pipeline-cut-around-t0:

cut_around_T0
~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L23623>`__

**Description:** See source for parameters and behavior.

.. code-block:: python

   def cut_around_T0(T0, lc, time_chunk): ...

.. _api-nemesis-tess-pipeline-data-gapped:

data_gapped
~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L25275>`__

**Description:** FA check: too many transits fall near data gaps.

.. code-block:: python

   def data_gapped(metrics, thresholds, pad: int=8, float_round: int=2): ...

.. _api-nemesis-tess-pipeline-df-len-check:

df_len_check
~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L421>`__

**Description:** Print the row count of each column in a DataFrame for pipeline debugging.

.. code-block:: python

   def df_len_check(df, message): ...

.. _api-nemesis-tess-pipeline-dmm:

dmm
~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L24694>`__

**Description:** FA check: inconsistency between mean and median transit depths.

.. code-block:: python

   def dmm(metrics, thresholds, pad: int=4, float_round: int=2): ...

.. _api-nemesis-tess-pipeline-edge-check:

edge_check
~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L1346>`__

**Description:** Check whether a target star falls within 6 pixels of a TESS detector edge.

.. code-block:: python

   def edge_check(ID, star, Sector, verbose, return_outedgeWarn=False): ...

.. _api-nemesis-tess-pipeline-edgewarn-tesspt2px:

edgeWarn_tesspt2px
~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L1194>`__

**Description:** Checks if a star (based on RA/Dec) is near the edge of a TESS camera for the given sectors.

.. code-block:: python

   def edgeWarn_tesspt2px(outSector, ra, dec, scinfo, aberrate=False, combinedFits=False, noCollateral=False, verbose=False): ...

.. _api-nemesis-tess-pipeline-extract-background-flux:

extract_background_flux
~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L6578>`__

**Description:** Per-frame, sigma-clipped background estimation and raw SAP flux extraction.

.. code-block:: python

   def extract_background_flux(flux: np.ndarray, bkg_mask: np.ndarray, pix_mask: np.ndarray, Nsig: float=3.0, max_iter: int=5, estimator: str='median') -> tuple: ...

.. _api-nemesis-tess-pipeline-filter-phase-range:

filter_phase_range
~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L21653>`__

**Description:** No description available in the source docstring.

.. code-block:: python

   def filter_phase_range(phase, flux, transit_duration): ...

.. _api-nemesis-tess-pipeline-find-min-trial-duration:

find_min_trial_duration
~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L16981>`__

**Description:** No description available in the source docstring.

.. code-block:: python

   def find_min_trial_duration(input_LC, durations, periods): ...

.. _api-nemesis-tess-pipeline-find-tesscut-file:

find_tesscut_file
~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L1443>`__

**Description:** Return every file whose name matches tess-s00[sector]-[camera]-[ccd]\_\*[size]x[size]\*astrocut.fits anywhere under 'basedir'.

.. code-block:: python

   def find_tesscut_file(basedir: str, sector: int, camera: int, ccd: int, size: int=11) -> list[Path]: ...

.. _api-nemesis-tess-pipeline-find-tpf-file:

find_tpf_file
~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L1470>`__

**Description:** Recursively find SPOC Target Pixel Files (TPFs) on disk.

.. code-block:: python

   def find_tpf_file(basedir: Union[str, Path], tic: int, sector: int, cadence: Optional[Union[int, str]]=None, include_gz: bool=True) -> Union[Path, List[Path], None]: ...

.. _api-nemesis-tess-pipeline-fix-wcs-axis-labels:

fix_wcs_axis_labels
~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L9868>`__

**Description:** Configure WCS-projected axes so RA labels appear on the bottom edge and Dec labels appear on the left edge, with ~3 tick marks per axis.

.. code-block:: python

   def fix_wcs_axis_labels(ax, ra_label='RA', dec_label='DEC', fontsize=10, show_grid=False, tick_spacing_arcmin=1.0, minpad=1): ...

.. _api-nemesis-tess-pipeline-format-ax:

format_ax
~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L9922>`__

**Description:** This function will format the subplot used for plotting the input TESS Full Frame Image or Target Pixel File (FFI/TPF).

.. code-block:: python

   def format_ax(ax, hdu): ...

.. _api-nemesis-tess-pipeline-format-conditions-block:

format_conditions_block
~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L24270>`__

**Description:** Format a multi-line LEO-Vetter condition trace as an indented text block.

.. code-block:: python

   def format_conditions_block(header: str, lines: list[str], padding: int | float | str=4) -> str: ...

.. _api-nemesis-tess-pipeline-fullphasefold:

fullphasefold
~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L18399>`__

**Description:** Phase-fold a light curve onto the full ''[0, 1)'' orbital phase interval.

.. code-block:: python

   def fullphasefold(time, T0, period, flux, offset): ...

.. _api-nemesis-tess-pipeline-generate-injection-path:

generate_injection_path
~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L14340>`__

**Description:** Generates the appropriate file path based on the given parameters.

.. code-block:: python

   def generate_injection_path(Sector, default_settings): ...

.. _api-nemesis-tess-pipeline-get-coordinates:

get_coordinates
~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L9342>`__

**Description:** Returns two 3D arrays of RA and Dec values in decimal degrees. If cadence number is given, returns 2D arrays for that cadence. If cadence is 'all' returns one RA, Dec value for each pixel in every cadence. Uses the…

.. code-block:: python

   def get_coordinates(hdu, cadence='all'): ...

.. _api-nemesis-tess-pipeline-get-flux-at-tn:

get_flux_at_TN
~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L21398>`__

**Description:** Get the flux value closest to a specific transit contact point (T1, T4, etc.) based on a phase-folded light curve (pf, ff). The window is used to limit the search to values close to the contact point (TN).

.. code-block:: python

   def get_flux_at_TN(pf, ff, TN, window=None): ...

.. _api-nemesis-tess-pipeline-get-gaia-dr2-sources:

get_GAIA_DR2_sources
~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L9428>`__

**Description:** This function will query the GAIA catalog for nearby sources using a radial cone search.

.. code-block:: python

   def get_GAIA_DR2_sources(ID, hdu, downloadpath, magnitude_limit): ...

.. _api-nemesis-tess-pipeline-get-gain-and-readout-from-hdu:

get_gain_and_readout_from_hdu
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L6432>`__

**Description:** No description available in the source docstring.

.. code-block:: python

   def get_gain_and_readout_from_hdu(hdu): ...

.. _api-nemesis-tess-pipeline-get-lc-from-ffi:

get_LC_from_FFI
~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L14767>`__

**Description:** Run the full per-sector NEMESIS LC extraction pipeline on a single HDU.

.. code-block:: python

   def get_LC_from_FFI(ID, nearby_star_catalog, Sector, cadence, hdu, CCD, Camera, reference_pixel, default_settings, **kwargs): ...

.. _api-nemesis-tess-pipeline-get-momentum-dump-times:

get_momentum_dump_times
~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L11433>`__

**Description:** This function will do a bitmask search in the input TESS Full Frame Image or Target Pixel File (FFI/TPF) image headers for momentum dump timestamps that occurred during TESS observations.

.. code-block:: python

   def get_momentum_dump_times(hdu): ...

.. _api-nemesis-tess-pipeline-get-pix-and-bkg-masks-and-image-fluxes:

get_pix_and_bkg_masks_and_image_fluxes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L4904>`__

**Description:** This function retrieves the pixel and background masks and the TESS image fluxes for a target star.

.. code-block:: python

   def get_pix_and_bkg_masks_and_image_fluxes(ID, Sector, cadence, default_settings): ...

.. _api-nemesis-tess-pipeline-get-pld-model:

get_PLD_model
~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L12693>`__

**Description:** Compute the additive PLD systematics model via weighted least squares.

.. code-block:: python

   def get_PLD_model(time: np.ndarray, input_flux: np.ndarray, error: np.ndarray, flux: np.ndarray, pix_mask: np.ndarray, pld_order: int, n_pca_terms: int, lambda_reg: float, ivar: Optional[np.ndarray]=None, var_explained_thr: float=0.95) -> np.ndarray: ...

.. _api-nemesis-tess-pipeline-get-polygon:

get_polygon
~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L6025>`__

**Description:** This function will produce a Shapely Polygon object that outlines the input pixel mask.

.. code-block:: python

   def get_polygon(mask, hdu): ...

.. _api-nemesis-tess-pipeline-get-qdata:

get_qdata
~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L6720>`__

**Description:** Download or load the TESS engineering quaternion FITS for a given sector from the MAST archive.

.. code-block:: python

   def get_qdata(sector: int, savepath: str, overwrite: bool=False) -> 'fits.HDUList': ...

.. _api-nemesis-tess-pipeline-get-qld:

get_qld
~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L9711>`__

**Description:** Retrieves quadratic limb darkening coefficients 'a' and 'b' for a given star, based on its effective temperature and surface gravity.

.. code-block:: python

   def get_qld(star, verbose, mdwarf_fallback_mass: float=0.4, mdwarf_fallback_radius: float=0.4): ...

.. _api-nemesis-tess-pipeline-get-scalar:

get_scalar
~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L24210>`__

**Description:** Coerce any scalar-like object to a plain Python scalar.

.. code-block:: python

   def get_scalar(value): ...

.. _api-nemesis-tess-pipeline-get-sources-in-aperture-mask:

get_sources_in_aperture_mask
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L6102>`__

**Description:** This function will query the TESS Input Catalog (TIC) for nearby sources using a radial cone search and check which sources are inside the input aperture pixel mask.

.. code-block:: python

   def get_sources_in_aperture_mask(ID, hdu, pix_mask, nearby_star_catalog, verbose=False): ...

.. _api-nemesis-tess-pipeline-get-tess-reference-pixel:

get_tess_reference_pixel
~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L4524>`__

**Description:** Return 0-indexed pixel-center coordinates [col, row] from a TESS binary-table HDU header.

.. code-block:: python

   def get_tess_reference_pixel(header) -> list: ...

.. _api-nemesis-tess-pipeline-get-tic-id:

get_TIC_ID
~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L526>`__

**Description:** Retrieves the TIC (TESS Input Catalog) ID of a given target star by querying the MAST Catalogs. If the star has a duplicate ID, the duplicate is returned. The search is conducted within a 5-arcsecond radius of the…

.. code-block:: python

   def get_TIC_ID(starName, verbose): ...

.. _api-nemesis-tess-pipeline-get-toi-and-ctoi-catalogs:

get_TOI_and_CTOI_catalogs
~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L19720>`__

**Description:** See source for parameters and behavior.

.. code-block:: python

   def get_TOI_and_CTOI_catalogs(): ...

.. _api-nemesis-tess-pipeline-gethdu:

gethdu
~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L4569>`__

**Description:** Download TESS FFI/TPF cutouts from MAST for one or more sectors and return all metadata needed for downstream SAP.

.. code-block:: python

   def gethdu(star, ID, Sector, cadence, default_settings): ...

.. _api-nemesis-tess-pipeline-gethdu-failure-summary:

gethdu_failure_summary
~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L4268>`__

**Description:** One line saying why a TIC and sector ended with no image data.

.. code-block:: python

   def gethdu_failure_summary(ID: Any, Sector: Any) -> str: ...

.. _api-nemesis-tess-pipeline-gethdu-lksearch:

gethdu_lksearch
~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L4309>`__

**Description:** Return the verified cutout for one TIC ID and one sector, exactly as ''\_gethdu\_lksearch\_verified'' does, and log why when none comes back.

.. code-block:: python

   def gethdu_lksearch(star: pd.DataFrame, ID: int, Sector: int, cadence: str, default_settings: dict) -> 'fits.HDUList | None': ...

.. _api-nemesis-tess-pipeline-hwhm:

HWHM
~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L16708>`__

**Description:** This function will calculated the Half-Width Half-Max (HWHM) of an input BLS\_periods, BLS\_SDEs data. This is used to estimate the uncertainty of the orbital period in the BLS power spectrum.

.. code-block:: python

   def HWHM(BLS_periods, BLS_SDEs): ...

.. _api-nemesis-tess-pipeline-in-versus-out-of-transit-test:

in_versus_out_of_transit_test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L22592>`__

**Description:** See source for parameters and behavior.

.. code-block:: python

   def in_versus_out_of_transit_test(ID, Sector, LC_df, transit_results_df, BLSorTLSmodel_df, verbose): ...

.. _api-nemesis-tess-pipeline-intransit-stats:

intransit_stats
~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L16744>`__

**Description:** Compute per-transit and odd/even transit statistics for a BLS or TLS result.

.. code-block:: python

   def intransit_stats(input_LC, TransitParams_df): ...

.. _api-nemesis-tess-pipeline-invalid-transits:

invalid_transits
~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L24387>`__

**Description:** FA check: insufficient valid transits after removing bad events.

.. code-block:: python

   def invalid_transits(metrics, thresholds, pad: int=0, float_round: int=2): ...

.. _api-nemesis-tess-pipeline-is-point-inside-polygon:

is_point_inside_polygon
~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L6066>`__

**Description:** This function will check if the input pixel coordinates are within the input Shapely polygon object.

.. code-block:: python

   def is_point_inside_polygon(pix_coords, poly): ...

.. _api-nemesis-tess-pipeline-is-toi-or-ctoi:

is_TOI_or_CTOI
~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L19680>`__

**Description:** See source for parameters and behavior.

.. code-block:: python

   def is_TOI_or_CTOI(ID, TOIs=None, CTOIs=None): ...

.. _api-nemesis-tess-pipeline-large:

large
~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L25516>`__

**Description:** FP check: planet candidate radius too large (likely an eclipsing binary).

.. code-block:: python

   def large(metrics, thresholds, pad: int=6, float_round: int=2): ...

.. _api-nemesis-tess-pipeline-lc-to-ls:

LC_to_LS
~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L20502>`__

**Description:** Render the LS periodogram and phase-folded LC panels for vetting report page 3.

.. code-block:: python

   def LC_to_LS(ax, ax1, ax2, ax3, ID, Sector, cadence, LC_df, TLS_bestfit_df, default_settings, filter_period: Optional[float]=None, nterms: int=1, fap_bootstrap: bool=False, n_bootstrap: int=100): ...

.. _api-nemesis-tess-pipeline-lin-interp:

lin_interp
~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L16692>`__

**Description:** This function is used to perform a simple linear interpolation.

.. code-block:: python

   def lin_interp(x, y, i, half): ...

.. _api-nemesis-tess-pipeline-load-camera-quaternions:

load_camera_quaternions
~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L6774>`__

**Description:** Extract TIME, Q1, Q2, Q3 for a given camera from a quaternion HDUList.

.. code-block:: python

   def load_camera_quaternions(qdata: 'fits.HDUList', camera: int) -> 'pd.DataFrame': ...

.. _api-nemesis-tess-compat-load-triceratops:

load_triceratops
~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.compat`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/compat.py#L8>`__

**Description:** Import TRICERATOPS with compatibility fixes for PyTransit 2.2.0.

.. code-block:: python

   def load_triceratops() -> ModuleType: ...

.. _api-nemesis-tess-pipeline-ls:

LS
~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L20208>`__

**Description:** Compute a generalized Lomb-Scargle periodogram with optional multi-term fitting.

.. code-block:: python

   def LS(time: np.ndarray, flux: np.ndarray, error: np.ndarray | None=None, min_period: float | None=None, max_period: float | None=None, filter_period: float | None=None, nterms: int=1) -> tuple: ...

.. _api-nemesis-tess-pipeline-make-dirs:

Make_dirs
~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L678>`__

**Description:** Create all NEMESIS output directories and write resolved path strings back into ''default\_settings''.

.. code-block:: python

   def Make_dirs(Sector, cadence, default_settings, return_paths=False): ...

.. _api-nemesis-tess-pipeline-mann2019-mass-radius:

mann2019_mass_radius
~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L9562>`__

**Description:** Estimate M-dwarf stellar mass and radius from absolute 2MASS K\_s magnitude.

.. code-block:: python

   def mann2019_mass_radius(Kmag: float, dist_pc: float, feh: float=0.0, use_metallicity: bool=False, fallback_mass: float=0.4, fallback_radius: float=0.4, verbose: bool=False) -> Tuple[float, float, bool]: ...

.. _api-nemesis-tess-pipeline-mask-out-neighbor-pixels:

mask_out_neighbor_pixels
~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L8198>`__

**Description:** Excise aperture pixels contaminated by neighboring stars.

.. code-block:: python

   def mask_out_neighbor_pixels(hdu, pix_mask: np.ndarray, nearby_star_catalog: pd.DataFrame, ID: int, reference_pixel: list, delta_tmag_threshold: float=5.0, neighbor_excision_radius: float=1.5, blend_threshold_px: float=2.0, blend_shrink_radius: float=1.5, min_aperture_pixels: int=2, verbose: bool=False) -> tuple: ...

.. _api-nemesis-tess-pipeline-mass-radius-mdwarf-relation:

mass_radius_mdwarf_relation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L590>`__

**Description:** Adjusts the mass and radius of an M-dwarf star based on the given inputs. This function uses an OVERLY simple linear mass-radius relationship for M-dwarfs, where if one of the stellar parameters (mass or radius) is…

.. code-block:: python

   def mass_radius_mdwarf_relation(M_star, R_star): ...

.. _api-nemesis-tess-pipeline-mirror-and-model-left-right-transit:

mirror_and_model_left_right_transit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L22440>`__

**Description:** Performs a comprehensive symmetry analysis for a transit event by comparing the left and right mirrored transits to the model and computing various metrics such as RMSE, residuals, and chi-squared values. Also…

.. code-block:: python

   def mirror_and_model_left_right_transit(ID, Sector, input_LC, transit_results_df, BLSorTLSmodel_df, verbose): ...

.. _api-nemesis-tess-pipeline-momentumdump-removal:

momentumdump_removal
~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L11621>`__

**Description:** This function will remove windows of data around momentum dump time stamps and return a mask.

.. code-block:: python

   def momentumdump_removal(SectorNum: int, before_after_in_minutes: float, time: np.ndarray, hdu, combined_mask=None, keep_baddata: bool=False, verbose: bool=False, apply_sector_jitter_mask: bool=True): ...

.. _api-nemesis-tess-pipeline-nan-symmetry-result:

nan_symmetry_result
~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L22300>`__

**Description:** Return a NaN-filled result with the exact 16-element shape that ''mirror\_and\_model\_left\_right\_transit'' returns on success.

.. code-block:: python

   def nan_symmetry_result() -> tuple: ...

.. _api-nemesis-tess-pipeline-nan-vetting-columns:

nan_vetting_columns
~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L22371>`__

**Description:** NaN-filled replacement for the vetting-metric column block that ''in\_versus\_out\_of\_transit\_test'' normally appends to the bestfit frame.

.. code-block:: python

   def nan_vetting_columns(transit_results_df) -> dict: ...

.. _api-nemesis-tess-pipeline-nanchecker:

nanchecker
~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L21550>`__

**Description:** Adjusts the standard deviation threshold to remove NaNs from mirrored fluxes.

.. code-block:: python

   def nanchecker(mirrored_phases, mirrored_fluxes, check_mirrored_fluxes, Nstd, duration, verbose=False): ...

.. _api-nemesis-tess-pipeline-non-unique:

non_unique
~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L24526>`__

**Description:** FA check: events not unique in the phased light curve.

.. code-block:: python

   def non_unique(metrics, thresholds, pad: int=8, float_round: int=2): ...

.. _api-nemesis-tess-pipeline-odd-even:

odd_even
~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L25352>`__

**Description:** FP check: significant odd/even transit depth mismatch (EB signature).

.. code-block:: python

   def odd_even(metrics, thresholds, pad: int=8, float_round: int=2): ...

.. _api-nemesis-tess-pipeline-odd-even-cuts:

odd_even_cuts
~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L20840>`__

**Description:** See source for parameters and behavior.

.. code-block:: python

   def odd_even_cuts(ID, Sector, LC_df, P, T0, Dur, Depth): ...

.. _api-nemesis-tess-pipeline-odd-even-mismatch-plot:

odd_even_mismatch_plot
~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L20902>`__

**Description:** See source for parameters and behavior.

.. code-block:: python

   def odd_even_mismatch_plot(ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ID, Sector, cadence, LC_df, TLSbestfit_df, path): ...

.. _api-nemesis-tess-pipeline-offset:

offset
~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L25700>`__

**Description:** FP check: centroid offset indicates an off-target eclipsing binary.

.. code-block:: python

   def offset(metrics, thresholds, pad: int=4, float_round: int=2): ...

.. _api-nemesis-tess-pipeline-optimize-period-duration-grid-sizes:

optimize_period_duration_grid_sizes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L17055>`__

**Description:** No description available in the source docstring.

.. code-block:: python

   def optimize_period_duration_grid_sizes(periods, durations, downsample_by_N_percent, cadence_of_observations): ...

.. _api-nemesis-tess-pipeline-outlier-removal:

outlier_removal
~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L13233>`__

**Description:** Sliding-window sigma-clipping outlier removal on the detrended light curve.

.. code-block:: python

   def outlier_removal(ID, nearby_star_catalog, Sector, input_LC, default_settings, method='local', window_size_in_days=None): ...

.. _api-nemesis-tess-pipeline-p2p-rms:

p2p_rms
~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L16206>`__

**Description:** Compute the point-to-point RMS of a light curve as a robust noise proxy.

.. code-block:: python

   def p2p_rms(flux): ...

.. _api-nemesis-tess-pipeline-patch-edivetter-unplugged:

patch_EDIVetter_Unplugged
~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L8>`__

**Description:** Patch the EDIunplugged package by replacing deprecated 'np.float' with 'float' in Python scripts. This patch is only applied if the installed version of EDIunplugged is "0.1.3".

.. code-block:: python

   def patch_EDIVetter_Unplugged(): ...

.. _api-nemesis-tess-pipeline-pd-to-lk:

pd_to_lk
~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L27387>`__

**Description:** Convert a NEMESIS ''LC\_df'' DataFrame to a ''lightkurve.LightCurve'' object.

.. code-block:: python

   def pd_to_lk(LC_df, flux_type='PLD'): ...

.. _api-nemesis-tess-pipeline-phasefold:

phasefold
~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L16128>`__

**Description:** Phase-fold a light curve with the transit centred at phase zero.

.. code-block:: python

   def phasefold(T0, time, period, flux): ...

.. _api-nemesis-tess-pipeline-pipeline-quality-mask:

pipeline_quality_mask
~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L1086>`__

**Description:** Applies a quality mask to a given TESS data quality array based on the desired bitmask setting.

.. code-block:: python

   def pipeline_quality_mask(bitmask_string, qual_arr, verbose): ...

.. _api-nemesis-tess-pipeline-pld-model:

PLD_model
~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L12845>`__

**Description:** Two-pass Pixel Level Decorrelation (PLD) with GP-proxy variance inflation.

.. code-block:: python

   def PLD_model(ID, Sector, flux: np.ndarray, pix_mask: np.ndarray, input_LC, default_settings: dict, window: int=10, max_dur: float=1.0, Nsigma: float=15.0, N_transits=None, lambda_reg: float=0.0, cadence_mask=None, do_plot: bool=False) -> 'pd.DataFrame': ...

.. _api-nemesis-tess-pipeline-plot-centroids-in-phase:

plot_centroids_in_phase
~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L18753>`__

**Description:** See source for parameters and behavior.

.. code-block:: python

   def plot_centroids_in_phase(LC_df, BLSorTLSbestfit_df, ax_placement): ...

.. _api-nemesis-tess-pipeline-plot-cutouts-alt:

plot_cutouts_ALT
~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L10828>`__

**Description:** Drop-in alternative to ''plot\_cutouts'' with a ''catalog\_source'' switch.

.. code-block:: python

   def plot_cutouts_ALT(ID, nearby_star_catalog, Sector, cadence, hdu, pix_mask, bkg_mask, reference_pixel, ncols, fig, ax, cbarpad, default_settings, catalog_source: str='TIC'): ...

.. _api-nemesis-tess-pipeline-plot-dss-orientation:

plot_dss_orientation
~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L19122>`__

**Description:** Fetch a DSS optical stamp, reproject it onto the TESS WCS pixel grid, and render it with N/E arrows and TESS aperture/background mask overlays.

.. code-block:: python

   def plot_dss_orientation(ax, ID, nearby_star_catalog, downloadpath, hdu=None, pix_mask=None, bkg_mask=None, cutoutsize: int=11, do_DSS_plot: bool=True, remove_fits: bool=True, show_mask_overlay: bool=True, pix_color: str='red', bkg_color: str='pink', verbose: bool=True): ...

.. _api-nemesis-tess-pipeline-plot-dss-sources:

plot_dss_sources
~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L11210>`__

**Description:** Overlay catalog sources on the reprojected DSS panel.

.. code-block:: python

   def plot_dss_sources(ax, hdu=None, nearby_star_catalog=None, gaia_df=None, target_id=None, target_gaia_id=None, pix_mask=None, oversample: int=21, source_color: str='gold', in_aperture_edge_color: str='deepskyblue', target_marker_color: str='white', cone_color: str='cyan', label_dmag_max: float=5.0, fontsize: int=8): ...

.. _api-nemesis-tess-pipeline-plot-image:

plot_image
~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L18620>`__

**Description:** See source for parameters and behavior.

.. code-block:: python

   def plot_image(ID, nearby_star_catalog, hdu, Sector, cadence, ax_placement, fig, fs, ncols, cbarpad, default_settings, return_masks=False): ...

.. _api-nemesis-tess-pipeline-plot-image-alt:

plot_image_ALT
~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L10958>`__

**Description:** Plot the TESS FFI/TPF median-image cutout with aperture/background masks and catalog source overlay (TIC or Gaia DR3).

.. code-block:: python

   def plot_image_ALT(ID, nearby_star_catalog, hdu, Sector, cadence, ax_placement, fig, fs, ncols, cbarpad, default_settings, return_masks=False, draw_legend=True): ...

.. _api-nemesis-tess-pipeline-plot-image-and-mask:

plot_image_and_mask
~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L9983>`__

**Description:** This function will plot the Full Frame Image or Target Pixel File (FFI/TPF) images and input aperture and background pixel masks.

.. code-block:: python

   def plot_image_and_mask(hdu, ID, nearby_star_catalog, pix_mask, bkg_mask, ax, fig, cbarpad, cmap=None, pix_color=None, bkg_color=None): ...

.. _api-nemesis-tess-pipeline-plot-it-all-up:

plot_it_all_up
~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L13636>`__

**Description:** Unified LC summary figure for both single-sector and multi-sector runs.

.. code-block:: python

   def plot_it_all_up(ID, nearby_star_catalog, Sector, cadence, hdu, RAWLC_df, clippedRAWLC_df, LC_df, good_ind_DF, bad_ind_DF, preclipLC_df, default_settings, median_image=None, pix_mask=None, bkg_mask=None, magnitude_limit=18, dot_scale=20, cbarpad=0.01, fs=16, savefigname=None, bkg_residuals=None, bkg_trend=None, bkg_level_per_pix=None, quat_jitter=None, quat_status='not provided', qlp_result=None, quality_flag=None, target_page_height: float | None=None): ...

.. _api-nemesis-tess-pipeline-plot-leo-vetter-text-results:

plot_LEO_Vetter_text_results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L25899>`__

**Description:** No description available in the source docstring.

.. code-block:: python

   def plot_LEO_Vetter_text_results(fig, metrics, fontx=0.615): ...

.. _api-nemesis-tess-pipeline-plot-mask:

plot_mask
~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L9957>`__

**Description:** This function is used to plot a pixel mask on a subplot.

.. code-block:: python

   def plot_mask(mask, mask_color, ax): ...

.. _api-nemesis-tess-pipeline-plot-mirrored-transits-and-collect-vetting-metrics:

plot_mirrored_transits_and_collect_vetting_metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L23053>`__

**Description:** See source for parameters and behavior.

.. code-block:: python

   def plot_mirrored_transits_and_collect_vetting_metrics(ID, Sector, LC_df, transit_results_df, BLSorTLSmodel_df, ax1, ax2, ax3, verbose): ...

.. _api-nemesis-tess-pipeline-plot-modshift-nemesis:

plot_modshift_NEMESIS
~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L25968>`__

**Description:** Compute LEO-Vetter transit metrics and render the modshift diagnostic figure.

.. code-block:: python

   def plot_modshift_NEMESIS(ID, star, LC_df, period, dur, T0, save_file=None, verbose=False): ...

.. _api-nemesis-tess-pipeline-plot-odd-even-transits:

plot_odd_even_transits
~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L18136>`__

**Description:** This function will plot the odd and even numbered transits that are modeled by TLS in the input subplot of a pyplot figure.

.. code-block:: python

   def plot_odd_even_transits(LC_df, TLSbestfit_df, TLSTCs_df, TLSmodel_df, ax, fig): ...

.. _api-nemesis-tess-pipeline-plot-orientation:

plot_orientation
~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L9390>`__

**Description:** This function will use the TESS Full Frame Image (FFI) or Target Pixel File (TPF) image headers to determine the North and East orientation for plotting the median image flux.

.. code-block:: python

   def plot_orientation(hdu, ax): ...

.. _api-nemesis-tess-pipeline-plot-page3-text:

plot_page3_text
~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L23182>`__

**Description:** See source for parameters and behavior.

.. code-block:: python

   def plot_page3_text(ID, Sector, LC_df, appended_transit_results, fig): ...

.. _api-nemesis-tess-pipeline-plot-phasefold-lcs:

plot_phasefold_LCs
~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L18448>`__

**Description:** See source for parameters and behavior.

.. code-block:: python

   def plot_phasefold_LCs(ID, hdu, LC_df, TLS_df, TLSmodel_df, TLSbestfit_df, axa, axb, axc, axd, axe): ...

.. _api-nemesis-tess-pipeline-plot-power-spectra:

plot_power_spectra
~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L18331>`__

**Description:** This function will plot the TLS power spectrum resulting from the TLS transit search.

.. code-block:: python

   def plot_power_spectra(TLS_df, TLSbestfit_df, ax): ...

.. _api-nemesis-tess-pipeline-plot-sources:

plot_sources
~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L10051>`__

**Description:** This function will plot catalog sources and centroid positions.

.. code-block:: python

   def plot_sources(starname, catalogData, cut_catalogData, hdu, ax, Sector, cadence, reference_pixel, pix_mask, default_settings): ...

.. _api-nemesis-tess-pipeline-plot-sources-gaia:

plot_sources_gaia
~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L10371>`__

**Description:** Overlay Gaia DR3 sources on a TESS cutout with PM correction to the cutout epoch.

.. code-block:: python

   def plot_sources_gaia(ID: int, hdu, ax: plt.Axes, gaia_catalog: pd.DataFrame, target_gaia_id: str, target_gmag: float, pix_mask: Optional[np.ndarray]=None, reference_pixel: Optional[list]=None, Sector: Optional[int]=None, cadence: Optional[str]=None, default_settings: Optional[dict]=None, show_pm: bool=False, show_centroids: bool=True, apply_pm_correction: bool=True, label_dmag_max: float=6.0, legend_loc: str='lower right', fontsize: int=12, source_color: str='gold', in_aperture_edge_color: str='deepskyblue', target_marker_color: str='white', draw_legend=True) -> dict: ...

.. _api-nemesis-tess-pipeline-plot-sources-tic:

plot_sources_TIC
~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L10172>`__

**Description:** Tpfplotter-style TIC source overlay on a TESS cutout.

.. code-block:: python

   def plot_sources_TIC(ID: int, hdu, ax, tic_catalog, pix_mask=None, reference_pixel=None, Sector=None, cadence=None, default_settings=None, show_centroids: bool=True, label_dtmag_max: float=6.0, legend_loc: str='lower right', fontsize: int=12, source_color: str='cyan', in_aperture_edge_color: str='deepskyblue', target_marker_color: str='white', draw_legend=True): ...

.. _api-nemesis-tess-pipeline-plot-text:

plot_text
~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L19489>`__

**Description:** See source for parameters and behavior.

.. code-block:: python

   def plot_text(ID, Sector, TLSbestfit_df, TLSTCs_df, fig, BLSorTLS): ...

.. _api-nemesis-tess-pipeline-preclean-sap-bkg-centroids:

preclean_SAP_BKG_Centroids
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L6142>`__

**Description:** Pre-clean SAP flux, background, and centroids before PLD/detrending.

.. code-block:: python

   def preclean_SAP_BKG_Centroids(preclip, time_col: str='Time', columns: tuple=('RAW SAP Flux', 'Background Flux', 'Centroid X Positions', 'Centroid Y Positions'), Nsig: float=5.0, side: 'str | dict'={'RAW SAP Flux': 'upper', 'Background Flux': 'both', 'Centroid X Positions': 'both', 'Centroid Y Positions': 'both'}, center: str='median', smooth_map: 'dict | None'={'RAW SAP Flux': False, 'Background Flux': True, 'Centroid X Positions': True, 'Centroid Y Positions': True}, window_length: float=0.5, method: str='biweight', edge_cutoff: float=0.0, break_tolerance: float=0.5, plot: bool=False, return_details: bool=False, consecutive_keep_col: str='RAW SAP Flux', consecutive_min_len: int=2): ...

.. _api-nemesis-tess-pipeline-prefetch-prf:

prefetch_prf
~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L7031>`__

**Description:** Pre-download TESS PRF FITS files from MAST for the epochs required by ''sectors''.

.. code-block:: python

   def prefetch_prf(sectors: List[int], default_settings: dict, cameras: tuple=(1, 2, 3, 4), overwrite: bool=False, verbose: bool=True) -> dict: ...

.. _api-nemesis-tess-pipeline-prefetch-quaternions:

prefetch_quaternions
~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L6860>`__

**Description:** Pre-download and cache TESS engineering quaternion FITS files for every sector in \*sectors\* before LC creation begins.

.. code-block:: python

   def prefetch_quaternions(sectors, default_settings: dict, *, overwrite: bool=False, verbose: bool=False) -> dict: ...

.. _api-nemesis-tess-pipeline-quick-outlier-removal:

quick_outlier_removal
~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L12504>`__

**Description:** Fast sliding-window outlier mask for a single flux column.

.. code-block:: python

   def quick_outlier_removal(input_LC, flux_type, Nsigma=5, window_size_in_days=1, method='local'): ...

.. _api-nemesis-tess-pipeline-readndarr:

readNDarr
~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L5737>`__

**Description:** Deserialize a pickle (.pkl) file written by ''saveNDarr''.

.. code-block:: python

   def readNDarr(path, filename): ...

.. _api-nemesis-tess-pipeline-recursive-search:

recursive_search
~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L328>`__

**Description:** Recursively find files under a directory tree matching a filename pattern.

.. code-block:: python

   def recursive_search(search_path, file_string): ...

.. _api-nemesis-tess-pipeline-refine-transit-epoch-post-search:

refine_transit_epoch_post_search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L17098>`__

**Description:** Refine a detected transit epoch after BLS/TLS using a guarded box-matched depth/SNR score.

.. code-block:: python

   def refine_transit_epoch_post_search(time: np.ndarray, flux: np.ndarray, t0_initial: float, period_days: float, duration_days: float, error: Optional[np.ndarray]=None, search_width_factor: float=3.0, max_search_days: Optional[float]=None, max_search_period_fraction: float=0.1, samples_per_duration: int=15, min_in_transit_cadences: int=3, min_depth: float=0.0, min_snr: float=0.0, min_score_improvement: float=0.0, return_diagnostics: bool=True) -> Tuple[float, float, Dict[str, Any]]: ...

.. _api-nemesis-tess-pipeline-resamplelinear1d:

ResampleLinear1D
~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L17042>`__

**Description:** No description available in the source docstring.

.. code-block:: python

   def ResampleLinear1D(periods, targetLength): ...

.. _api-nemesis-tess-pipeline-robust-transit-depth:

robust_transit_depth
~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L21710>`__

**Description:** Robust estimation of transit depth using sigma clipping, quantiles, and fallback strategies.

.. code-block:: python

   def robust_transit_depth(phase, flux, transit_duration, quantile=0.01, sigma_clip_thresh=3.0, fallback_strategy='weighted'): ...

.. _api-nemesis-tess-pipeline-running-mean:

running_mean
~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L5578>`__

**Description:** This function will calculate a mean value for a given window of time.

.. code-block:: python

   def running_mean(data, window_size): ...

.. _api-nemesis-tess-pipeline-running-median:

running_median
~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L5531>`__

**Description:** This function will calculate a median value for a given window of time.

.. code-block:: python

   def running_median(flux, window_size): ...

.. _api-nemesis-tess-pipeline-safe-cleanup-cache:

safe_cleanup_cache
~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L2459>`__

**Description:** Best-effort cleanup of files under the ONE cache root indicated by default\_settings. Mirrors your old behavior but without hardcoded /mnt/... paths. When ''shard\_cache\_by\_tic'' is enabled and ''tic\_id'' is provided…

.. code-block:: python

   def safe_cleanup_cache(default_settings, tic_id=None, remove_tesscut=True, remove_tpf=False, verbose=False): ...

.. _api-nemesis-tess-pipeline-safe-panel-normalization:

safe_panel_normalization
~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L23650>`__

**Description:** Normalize a per-panel flux/error pair by a guaranteed-positive scale.

.. code-block:: python

   def safe_panel_normalization(flux: np.ndarray, error: np.ndarray, fallback_scale: float=1.0, min_scale: float=1e-06) -> Tuple[np.ndarray, np.ndarray, float]: ...

.. _api-nemesis-tess-pipeline-sap-alt5:

SAP_ALT5
~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L8522>`__

**Description:** Simple Aperture Photometry with fully coupled aperture evaluation.

.. code-block:: python

   def SAP_ALT5(ID: int, nearby_star_catalog: 'pd.DataFrame', Sector: int, cadence: str, hdu: 'fits.HDUList', quality_mask: 'np.ndarray', reference_pixel: list, default_settings: dict): ...

.. _api-nemesis-tess-pipeline-sap-and-bkg-uncertainty:

SAP_and_BKG_uncertainty
~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L6455>`__

**Description:** Propagate photon and readout noise into SAP and background flux uncertainties.

.. code-block:: python

   def SAP_and_BKG_uncertainty(rawflux, bkgFlux, pix_mask, bkg_mask, rawtime, read_noise, gain): ...

.. _api-nemesis-tess-pipeline-savendarr:

saveNDarr
~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L5687>`__

**Description:** Serialize an arbitrary Python object to a pickle (.pkl) file.

.. code-block:: python

   def saveNDarr(multiNDarr, path, filename): ...

.. _api-nemesis-tess-pipeline-scramble-hdu-orbits:

scramble_hdu_orbits
~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L14638>`__

**Description:** Swap the two TESS orbit image-frame blocks within a single-sector HDU, preserving the original time array.

.. code-block:: python

   def scramble_hdu_orbits(hdu, Sector: int, verbose: bool=True) -> None: ...

.. _api-nemesis-tess-pipeline-search-sectors-from-filetype-and-path:

search_sectors_from_filetype_and_path
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L4746>`__

**Description:** Discover which TESS sectors have processed light curves on disk.

.. code-block:: python

   def search_sectors_from_filetype_and_path(ID, default_settings, restrict_to=None, require_masks: bool=False): ...

.. _api-nemesis-tess-pipeline-secondary:

secondary
~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L25581>`__

**Description:** FP check: significant secondary eclipse detected.

.. code-block:: python

   def secondary(metrics, thresholds, pad: int=6, float_round: int=2): ...

.. _api-nemesis-tess-pipeline-single-event:

single_event
~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L24761>`__

**Description:** FA check: signal dominated by a single event.

.. code-block:: python

   def single_event(metrics, thresholds, pad: int=8, float_round: int=2): ...

.. _api-nemesis-tess-pipeline-singleoutliers:

singleoutliers
~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L13182>`__

**Description:** This function will search input data for points that are not consecutive.

.. code-block:: python

   def singleoutliers(data, stepsize=1): ...

.. _api-nemesis-tess-pipeline-sinusoidal:

sinusoidal
~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L24940>`__

**Description:** FA check: signal consistent with sinusoidal variability.

.. code-block:: python

   def sinusoidal(metrics, thresholds, pad: int=6, float_round: int=2): ...

.. _api-nemesis-tess-pipeline-smoother:

Smoother
~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L12135>`__

**Description:** Smooth the PLD-corrected light curve with a biweight running window.

.. code-block:: python

   def Smoother(ID: int, nearby_star_catalog, Sector, input_LC, default_settings: dict, break_tolerance_in_days: Optional[float]=None, mode: str='uniform', transit_dur_factor: float=3.0, rot_divisor: float=5.0, min_prot_days: float=0.5, min_finite_for_smoothing: int=15): ...

.. _api-nemesis-tess-pipeline-symmetry-window-is-degenerate:

symmetry_window_is_degenerate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L22163>`__

**Description:** Decide whether the left/right mirror symmetry test can run at all.

.. code-block:: python

   def symmetry_window_is_degenerate(phase: np.ndarray, flux: np.ndarray, model_phase: np.ndarray, model_flux: np.ndarray, transit_duration: float, cadence: float, n_cad_min: float=0.0, verbose: bool=False) -> tuple[bool, str]: ...

.. _api-nemesis-tess-pipeline-t14:

T14
~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L11996>`__

**Description:** Estimate the total transit duration (T14) for a planet using Kepler's third law and transit geometry.

.. code-block:: python

   def T14(P, R_star, M_star, R_planet, b=0, i=90): ...

.. _api-nemesis-tess-pipeline-t23:

T23
~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L12053>`__

**Description:** Estimate the transit duration between the second and third contact points (T23).

.. code-block:: python

   def T23(P, R_star, M_star, R_planet, b=0, i=90): ...

.. _api-nemesis-tess-pipeline-tess-sector-observation-check:

TESS_Sector_observation_check
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L11474>`__

**Description:** This function will return the TESS timestamps corresponding to the beginning of the first to the end of the second orbit per TESS sector of observations. Values are interpretted from the TESS Data Release Notes…

.. code-block:: python

   def TESS_Sector_observation_check(SectorNum=None, T0=None): ...

.. _api-nemesis-tess-pipeline-tesscutcountersemaphore:

TesscutCounterSemaphore
~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L2606>`__

**Description:** Cross-process semaphore using a single integer counter file.

.. code-block:: python

   class TesscutCounterSemaphore

.. _api-nemesis-tess-pipeline-tesssearch-with-radius-growth:

tesssearch_with_radius_growth
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L2714>`__

**Description:** Perform lksearch.TESSSearch with an expanding search radius when the initial sector-constrained query raises a SearchError or yields no matching rows.

.. code-block:: python

   def tesssearch_with_radius_growth(ID: int, *, Sector: int, exptime: int, ffi_or_tpf: str, start_pixels: int=2, step_pixels: int=3, max_pixels: int=32, attempts: int | None=None, verbose: bool=False, search_fn=None) -> tuple: ...

.. _api-nemesis-tess-pipeline-thresholdmask:

thresholdmask
~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L5298>`__

**Description:** Build a source aperture mask from pixels above a MAD-scaled threshold.

.. code-block:: python

   def thresholdmask(hdu, reference_pixel, threshold, use_centroid=False, use_circ=True, max_aperture_radius=None): ...

.. _api-nemesis-tess-pipeline-timer:

timer
~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L373>`__

**Description:** Print the wall-clock runtime of a pipeline step in human-readable units.

.. code-block:: python

   def timer(start, end, message): ...

.. _api-nemesis-tess-pipeline-tls-func:

TLS_func
~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L16262>`__

**Description:** This function will perform a transit search on the input light curve using the Transit Least Squares (TLS, Hippke & Heller et al. 2019, https://arxiv.org/pdf/1901.02015.pdf) algorithm. Run the Transit Least Squares…

.. code-block:: python

   def TLS_func(ID, nearby_star_catalog, Sector, cadence, hdu, time, flux, error, default_settings): ...

.. _api-nemesis-tess-pipeline-transit-injection:

transit_injection
~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L14390>`__

**Description:** Inject a limb-darkened transit into a TESS-like light curve using BATMAN.

.. code-block:: python

   def transit_injection(input_LC, Period, T0, ID, nearby_star_catalog, Sector, default_settings, R_planet_RE=None, transit_depth=None, flux_contamination_ratio=0.0, inclination=90, eccentricity=0, ang_periastron=90): ...

.. _api-nemesis-tess-pipeline-transit-plot:

Transit_plot
~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L17882>`__

**Description:** Save a 2×2 transit search summary figure for the best-fit BLS or TLS signal.

.. code-block:: python

   def Transit_plot(ID, nearby_star_catalog, Sector, cadence, hdu, method, input_LC, PowerSpectrum_df, TransitModel_df, TransitParams_df, default_settings): ...

.. _api-nemesis-tess-pipeline-transitsearch:

TransitSearch
~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L17774>`__

**Description:** Dispatcher that runs either BLS or TLS on the input light curve.

.. code-block:: python

   def TransitSearch(method, ID, nearby_star_catalog, Sector, cadence, hdu, input_LC, default_settings): ...

.. _api-nemesis-tess-pipeline-try-apertures:

try_apertures
~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L7656>`__

**Description:** Grid-search over aperture definitions and select the one minimising CDPP × (1 + contamination ratio).

.. code-block:: python

   def try_apertures(hdu, reference_pixel, quality_mask, nearby_star_catalog, ID, rawtime, default_settings, thresholds=(3.0, 5.0, 7.5, 10.0), radii=(1.0, 1.5, 2.0, 2.5)): ...

.. _api-nemesis-tess-pipeline-unphysical-duration:

unphysical_duration
~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L25011>`__

**Description:** FA check: transit duration is unphysically short or long.

.. code-block:: python

   def unphysical_duration(metrics, thresholds, pad: int=13, float_round: int=2): ...

.. _api-nemesis-tess-pipeline-verify-cutout-identity:

verify_cutout_identity
~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L3105>`__

**Description:** Check that a cutout (TESScut or TPF) is the requested sector and the requested star.

.. code-block:: python

   def verify_cutout_identity(header, sector: int, *, ra: float | None=None, dec: float | None=None, pos_tol_arcsec: float=21.0, expected_camera: int | None=None, expected_ccd: int | None=None, filename: str | None=None, require_position: bool=False, times=None, sector_of_time: 'Callable[[float], int] | None'=None) -> dict: ...

.. _api-nemesis-tess-pipeline-verify-lc-matches-cutout:

verify_lc_matches_cutout
~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L2926>`__

**Description:** Check that a light curve's time stamps come from the cutout it was extracted from.

.. code-block:: python

   def verify_lc_matches_cutout(lc_time, cutout_time, *, atol_days: float=1e-05, min_frac: float=0.9) -> dict: ...

.. _api-nemesis-tess-pipeline-verify-lc-sector-by-time:

verify_lc_sector_by_time
~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L3041>`__

**Description:** Check that a light curve's first and last times fall in the requested sector. This is your ''TESS\_Sector\_observation\_check(T0=np.min(...))'' and ''(T0=np.max(...))'' test, run automatically.

.. code-block:: python

   def verify_lc_sector_by_time(lc_time, sector: int, *, sector_of_time: 'Callable[[float], int] | None'=None) -> dict: ...

.. _api-nemesis-tess-pipeline-verify-written-lc:

verify_written_lc
~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L2989>`__

**Description:** Re-read the Time column of a just-written light curve and compare it with the times held in memory. Catches a file on disk that is not the one this process wrote. A file that is not on disk is noted, not failed.

.. code-block:: python

   def verify_written_lc(path: str, lc_time, *, atol_days: float=1e-07) -> dict: ...

.. _api-nemesis-tess-pipeline-vet-with-edivetter:

Vet_with_EDIVetter
~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L16172>`__

**Description:** See source for parameters and behavior.

.. code-block:: python

   def Vet_with_EDIVetter(ID, Sector, cadence, TLS_OUTPUT, qld, SDE_threshold, N_transits, path=None): ...

.. _api-nemesis-tess-pipeline-vetting-report:

Vetting_Report
~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L19747>`__

**Description:** Produce page 1 of the NEMESIS vetting report for a BLS or TLS transit signal.

.. code-block:: python

   def Vetting_Report(ID, nearby_star_catalog, Sector, cadence, hdu, default_settings, BLSorTLS): ...

.. _api-nemesis-tess-pipeline-vetting-report-2nd-page:

Vetting_Report_2nd_page
~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L21080>`__

**Description:** Produce page 2 of the NEMESIS vetting report: the full LC summary figure.

.. code-block:: python

   def Vetting_Report_2nd_page(ID, nearby_star_catalog, Sector, LC_df, hdu, path, cadence, TLSbestfit_df, BLSorTLS, default_settings, bkg_residuals=None, bkg_trend=None, bkg_level_per_pix=None, quat_jitter=None, quat_status='not provided', qlp_result=None, quality_flag=None): ...

.. _api-nemesis-tess-pipeline-vetting-report-3rd-page:

Vetting_Report_3rd_page
~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L21274>`__

**Description:** Produce page 3 of the NEMESIS vetting report: LS periodogram and phase-folded light curves.

.. code-block:: python

   def Vetting_Report_3rd_page(ID, Sector, LC_df, path, cadence, TLSbestfit_df, BLSorTLS, default_settings): ...

.. _api-nemesis-tess-pipeline-vetting-report-4th-page:

Vetting_Report_4th_page
~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L23427>`__

**Description:** Produce page 4 of the NEMESIS vetting report: centroid analysis and transit symmetry tests.

.. code-block:: python

   def Vetting_Report_4th_page(ID, Sector, LC_df, hdu, path, cadence, transit_results_df, BLSorTLS, BLSorTLSmodel_df, default_settings, verbose=False): ...

.. _api-nemesis-tess-pipeline-vetting-report-5th-page-plot-individual-transits:

Vetting_Report_5th_page_plot_individual_transits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L23889>`__

**Description:** Produce page 5 of the NEMESIS vetting report: individual transit zoom-ins.

.. code-block:: python

   def Vetting_Report_5th_page_plot_individual_transits(ID, Sector, cadence, LC_df, path, bestfit_df, model_df, BLSorTLS, default_settings, max_panels: int=9, n_dur_window: float=3.5, snr_threshold_for_stratify: int=9, n_best: int=3, n_mid: int=3, n_worst: int=3, intransit_half_width_factor: float=1.0): ...

.. _api-nemesis-tess-pipeline-vetting-report-6th-page-leovetter:

Vetting_Report_6th_page_LEOVetter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L26217>`__

**Description:** Produce page 6 of the NEMESIS vetting report: LEO-Vetter modshift analysis.

.. code-block:: python

   def Vetting_Report_6th_page_LEOVetter(ID, nearby_star_catalog, Sector, cadence, LC_df, path, bestfit_df, BLSorTLS, default_settings, verbose): ...

.. _api-nemesis-tess-pipeline-vetting-report-multi-page:

Vetting_Report_multi_page
~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L26338>`__

**Description:** Produce pages 2–6 of the NEMESIS vetting report and merge all pages into a single multi-page PDF.

.. code-block:: python

   def Vetting_Report_multi_page(ID, nearby_star_catalog, Sector, cadence, hdu, BLSorTLS, default_settings, bkg_residuals=None, bkg_trend=None, bkg_level_per_pix=None, quat_jitter=None, quat_status='not provided', qlp_result=None, quality_flag=None): ...

.. _api-nemesis-tess-pipeline-vshaped:

vshaped
~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L25447>`__

**Description:** FP check: V-shaped transit event (consistent with a grazing eclipsing binary).

.. code-block:: python

   def vshaped(metrics, thresholds, pad: int=8, float_round: int=2): ...

.. _api-nemesis-tess-pipeline-weak:

weak
~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L24327>`__

**Description:** FA check: signal too weak (MES below minimum threshold).

.. code-block:: python

   def weak(metrics, thresholds, pad: int=4, float_round: int=2): ...

.. _api-nemesis-tess-pipeline-while-loop:

while_loop
~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L467>`__

**Description:** Call 'command(\*args, \*\*kwargs)' with exponential back‑off. Stops after 'max\_retries' failures (None = infinite). Returns the function’s result or raises the last exception.

.. code-block:: python

   def while_loop(command, message, *args, base_delay=1, max_delay=64, max_retries=10, total_timeout=None, **kwargs): ...

Private and internal helpers
----------------------------

.. _api-nemesis-tess-pipeline-apply-outlier-flag:

_apply_outlier_flag
~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L7471>`__

**Description:** Set QUALITY\_FLAG = 3 on cadences identified by outlier\_removal().

.. code-block:: python

   def _apply_outlier_flag(SAP_LC_full: 'pd.DataFrame', outlier_indices: 'np.ndarray') -> 'pd.DataFrame': ...

.. _api-nemesis-tess-pipeline-bitmask-to-flag-ids:

_bitmask_to_flag_ids
~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L7417>`__

**Description:** Convert stored bitmask value into flag identities.

.. code-block:: python

   def _bitmask_to_flag_ids(bitmask) -> list[int]: ...

.. _api-nemesis-tess-pipeline-bitmask-to-flag-string:

_bitmask_to_flag_string
~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L7441>`__

**Description:** Human-readable representation for CSV export.

.. code-block:: python

   def _bitmask_to_flag_string(bitmask) -> str: ...

.. _api-nemesis-tess-pipeline-bootstrap-fap:

_bootstrap_fap
~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L20331>`__

**Description:** Estimate FAP by counting how often a shuffled light curve exceeds the observed peak power. O(n\_bootstrap × N\_freq) — keep n\_bootstrap ≤ 200.

.. code-block:: python

   def _bootstrap_fap(time, flux, error, observed_peak: float, n_bootstrap: int, min_period: Optional[float], max_period: Optional[float], nterms_use: int) -> float: ...

.. _api-nemesis-tess-pipeline-cache-lock-name:

_cache_lock_name
~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L2543>`__

**Description:** Build a narrow-scope lock filename so only jobs touching the same resource serialize. Examples: flatten\_TESSCut\_s0001\_cam1\_ccd4.lock flatten\_TESS\_s0001\_tic\_24195569.lock flatten\_TESSCut\_generic.lock (fallback)

.. code-block:: python

   def _cache_lock_name(*, product_tag: str, sector: int | None=None, camera: int | None=None, ccd: int | None=None, tic_id: int | str | None=None) -> str: ...

.. _api-nemesis-tess-pipeline-cache-root-from-settings:

_cache_root_from_settings
~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L2450>`__

**Description:** Resolve the canonical cache root (your existing cache path). All cache activity (TESScut/TPF/DSS) uses this path.

.. code-block:: python

   def _cache_root_from_settings(default_settings): ...

.. _api-nemesis-tess-pipeline-cachelock:

_CacheLock
~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L2566>`__

**Description:** Best-effort file lock based on creating a lockfile. Scope is controlled by the lock name you pass in. Lock file is removed on exit; failures to remove are ignored.

.. code-block:: python

   class _CacheLock

.. _api-nemesis-tess-pipeline-check-harmonic:

_check_harmonic
~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L20307>`__

**Description:** Check whether the 2P candidate has higher power than P, which would indicate p\_peak is actually P\_rot/2.

.. code-block:: python

   def _check_harmonic(periods: np.ndarray, power: np.ndarray, p_peak: float, frac: float=0.05) -> tuple: ...

.. _api-nemesis-tess-pipeline-decorate-quality-columns:

_decorate_quality_columns
~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L7451>`__

**Description:** Ensure QUALITY\_FLAG is int16 bitmask and attach readable QUALITY\_FLAGS.

.. code-block:: python

   def _decorate_quality_columns(df: 'pd.DataFrame', quality_col: str='QUALITY_FLAG', label_col: str='QUALITY_FLAGS') -> 'pd.DataFrame': ...

.. _api-nemesis-tess-pipeline-detect-deep-dips:

_detect_deep_dips
~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L12622>`__

**Description:** Flag downward outliers in PLD residuals as candidate transit cadences.

.. code-block:: python

   def _detect_deep_dips(residuals: np.ndarray, sigma: float=4.0, grow_by: int=1) -> np.ndarray: ...

.. _api-nemesis-tess-pipeline-ensure-hdu-or-raise:

_ensure_hdu_or_raise
~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L283>`__

**Description:** Raise NoImageDataError if nothing usable is in hdu.

.. code-block:: python

   def _ensure_hdu_or_raise(hdu, ID, Sector, cadence): ...

.. _api-nemesis-tess-pipeline-ensure-hdu-or-raise-old:

_ensure_hdu_or_raise_OLD
~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L275>`__

**Description:** Raise NoImageDataError if nothing usable is in hdu.

.. code-block:: python

   def _ensure_hdu_or_raise_OLD(hdu, ID, Sector, cadence): ...

.. _api-nemesis-tess-pipeline-ensure-tesscut-dirs:

_ensure_tesscut_dirs
~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L1670>`__

**Description:** Make sure astroquery's expected dirs exist under the cache tree. Avoids FileNotFoundError for .../mastDownload/TESSCut/tesscut\_\*.zip

.. code-block:: python

   def _ensure_tesscut_dirs(downloadpath, tic_id=None, shard_by_tic=False): ...

.. _api-nemesis-tess-pipeline-filter-lc-by-quality-flags:

_filter_lc_by_quality_flags
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L7518>`__

**Description:** Return a copy of lc\_df keeping only rows whose QUALITY\_FLAG is consistent with the requested allowed flag-ids.

.. code-block:: python

   def _filter_lc_by_quality_flags(lc_df, allowed_quality_flags=None, quality_col='QUALITY_FLAG', verbose=False): ...

.. _api-nemesis-tess-pipeline-flag-ids-to-bitmask:

_flag_ids_to_bitmask
~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L7395>`__

**Description:** Convert flag identities like [1,2,3] into bitmask sum 1+2+4 = 7.

.. code-block:: python

   def _flag_ids_to_bitmask(flag_ids) -> np.int16: ...

.. _api-nemesis-tess-pipeline-flatten-to-cache:

_flatten_to_cache
~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L1551>`__

**Description:** Move files downloaded by 'lksearch' into a cache/mastDownload directory.

.. code-block:: python

   def _flatten_to_cache(df, cache_root, product_tag, tic_id=None, shard_by_tic=False): ...

.. _api-nemesis-tess-pipeline-fmt-elapsed:

_fmt_elapsed
~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L4092>`__

**Description:** Format a duration for log lines: seconds below 2 minutes, else minutes.

.. code-block:: python

   def _fmt_elapsed(seconds: float) -> str: ...

.. _api-nemesis-tess-pipeline-get-cached-fits:

_get_cached_fits
~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L2034>`__

**Description:** Fast lookup of a cached TESScut FITS using deterministic paths first, then a tiny per-key SQLite index to avoid directory-wide scans.

.. code-block:: python

   def _get_cached_fits(cache_root, *, product_tag, sector, camera, ccd, size, target_coord, tic_id=None, shard_by_tic=False, migrate_flat_to_sharded=True, migrate_sharded_to_flat=True, tol=1 * u.arcsec): ...

.. _api-nemesis-tess-pipeline-gethdu-diag-key:

_gethdu_diag_key
~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L4116>`__

**Description:** Build the (TIC, sector) key used by the retrieval record.

.. code-block:: python

   def _gethdu_diag_key(ID: Any, Sector: Any) -> Tuple[int, int]: ...

.. _api-nemesis-tess-pipeline-gethdu-lksearch-unverified:

_gethdu_lksearch_unverified
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L3276>`__

**Description:** Download a single TESS FFI cutout or TPF from MAST via lksearch.

.. code-block:: python

   def _gethdu_lksearch_unverified(star, ID, Sector, cadence, default_settings): ...

.. _api-nemesis-tess-pipeline-gethdu-lksearch-verified:

_gethdu_lksearch_verified
~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L3781>`__

**Description:** Return the cutout for one TIC ID and one sector, verified to be that star and that sector, or None if no verified cutout can be obtained.

.. code-block:: python

   def _gethdu_lksearch_verified(star: pd.DataFrame, ID: int, Sector: int, cadence: str, default_settings: dict) -> 'fits.HDUList | None': ...

.. _api-nemesis-tess-pipeline-ls-results-for-flux:

_ls_results_for_flux
~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L20353>`__

**Description:** Compute all Lomb-Scargle metrics for a single flux array.

.. code-block:: python

   def _ls_results_for_flux(time: np.ndarray, flux: np.ndarray, error: Optional[np.ndarray]=None, min_period: float=0.5, max_period: Optional[float]=None, nterms: int=1, n_bootstrap: int=100, fap_bootstrap: bool=False, harmonic_frac: float=0.05) -> dict: ...

.. _api-nemesis-tess-pipeline-nemesis-skyview-stamp:

_nemesis_skyview_stamp
~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L18838>`__

**Description:** Fetch a sky survey image stamp and return ''(stamp, header)''.

.. code-block:: python

   def _nemesis_skyview_stamp(ra: float, dec: float, survey: str='DSS2 Red', scaling: str='Linear', sizepix: int=300, flip: bool=True, convolvewith=None, forcefetch: bool=False, cachedir=None, timeout: float=30.0, retry_failed: bool=False, savewcsheader: bool=True, fallback_surveys: tuple=('DSS2 Blue', 'DSS'), max_retries: int=2, verbose: bool=False) -> tuple: ...

.. _api-nemesis-tess-pipeline-normalize-gethdu-outputs:

_normalize_gethdu_outputs
~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L4435>`__

**Description:** Ensure that all five ''gethdu'' return values are Python \*lists\* of equal length, regardless of whether one or multiple sectors were returned.

.. code-block:: python

   def _normalize_gethdu_outputs(hdus, CCDs, Cameras, quality_masks, reference_pixels) -> tuple: ...

.. _api-nemesis-tess-pipeline-per-transit-snr:

_per_transit_snr
~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L23716>`__

**Description:** Compute a simple per-transit SNR for a single event.

.. code-block:: python

   def _per_transit_snr(t0: float, lc_time: np.ndarray, lc_flux: np.ndarray, dur_days: float, window_days: float, intransit_half_width_factor: float=1.0) -> float: ...

.. _api-nemesis-tess-pipeline-record-gethdu-diag:

_record_gethdu_diag
~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L4149>`__

**Description:** Store the latest retrieval outcome for one (TIC, sector).

.. code-block:: python

   def _record_gethdu_diag(key: Tuple[int, int], cause: str, detail: str, elapsed_s: float) -> None: ...

.. _api-nemesis-tess-pipeline-resolve-prf-cache-dir:

_resolve_prf_cache_dir
~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L6980>`__

**Description:** Resolve the PRF cache directory from NEMESIS settings.

.. code-block:: python

   def _resolve_prf_cache_dir(default_settings: dict) -> str: ...

.. _api-nemesis-tess-pipeline-resolve-quat-savepath:

_resolve_quat_savepath
~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L6814>`__

**Description:** Derive a canonical quaternion cache directory from default\_settings and write it back as default\_settings['quat\_savepath'].

.. code-block:: python

   def _resolve_quat_savepath(default_settings: dict) -> str: ...

.. _api-nemesis-tess-pipeline-resolve-target-sectors:

_resolve_target_sectors
~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L7201>`__

**Description:** Return the concrete list of sector ints that full\_pipeline will process.

.. code-block:: python

   def _resolve_target_sectors(ID: int, Sector, star, default_settings: dict) -> list[int]: ...

.. _api-nemesis-tess-pipeline-restore-full-length-lc:

_restore_full_length_lc
~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L7603>`__

**Description:** Rebuild a full-length LC dataframe from compact good-cadence results.

.. code-block:: python

   def _restore_full_length_lc(SAP_LC_full: 'pd.DataFrame', compact_df: 'pd.DataFrame', preserve_columns=('CADENCE_INDEX', 'Time', 'RAW SAP Flux', 'SAP Flux', 'SAP Error', 'Background Flux', 'Background Error', 'QUALITY_FLAG')) -> 'pd.DataFrame': ...

.. _api-nemesis-tess-pipeline-run-cutouts-alt:

_run_cutouts_ALT
~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L11120>`__

**Description:** Internal helper called by plot\_image\_ALT. Runs the full cutout plot and returns (gaia\_df, target\_gaia\_id).

.. code-block:: python

   def _run_cutouts_ALT(ID, nearby_star_catalog, Sector, cadence, hdu, pix_mask, bkg_mask, reference_pixel, ncols, fig, ax, cbarpad, default_settings, catalog_source, draw_legend): ...

.. _api-nemesis-tess-pipeline-sec-str:

_sec_str
~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L4413>`__

**Description:** Safe sector string for logs; won’t blow up on numpy/list types.

.. code-block:: python

   def _sec_str(sector): ...

.. _api-nemesis-tess-pipeline-select-transit-indices:

_select_transit_indices
~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L23778>`__

**Description:** Return (selected\_indices, tier\_labels) for stratified SNR-ranked selection.

.. code-block:: python

   def _select_transit_indices(snr_array: np.ndarray, n_best: int=3, n_mid: int=3, n_worst: int=3) -> tuple: ...

.. _api-nemesis-tess-pipeline-tesspoint-note:

_tesspoint_note
~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L4222>`__

**Description:** State tess-point's prediction for one star and sector, as one sentence.

.. code-block:: python

   def _tesspoint_note(star: pd.DataFrame, ID: Any, Sector: Any) -> str: ...

.. _api-nemesis-tess-pipeline-wait-for-paths:

_wait_for_paths
~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L2506>`__

**Description:** Wait until each path in 'paths' exists and has stable size for 'stable\_reads' polls. Early-exits as soon as condition is met for any path (you'll open just one). Returns True if any path stabilized; False otherwise.

.. code-block:: python

   def _wait_for_paths(paths, timeout: float=0.5, poll: float=0.05, stable_reads: int=2): ...

Legacy-named definitions
------------------------

.. _api-nemesis-tess-pipeline-create-circular-mask-old:

create_circular_mask_OLD
~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L5195>`__

**Description:** Create a circular boolean aperture mask centred on a reference pixel.

.. code-block:: python

   def create_circular_mask_OLD(median_image, reference_pixel, radius=1): ...

.. _api-nemesis-tess-pipeline-gethdu-lksearch-old:

gethdu_lksearch_OLD
~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L3930>`__

**Description:** Return the cutout for one TIC ID and one sector, verified to be that star and that sector, or None if no verified cutout can be obtained.

.. code-block:: python

   def gethdu_lksearch_OLD(star: pd.DataFrame, ID: int, Sector: int, cadence: str, default_settings: dict) -> 'fits.HDUList | None': ...

.. _api-nemesis-tess-pipeline-search-sectors-from-filetype-and-path-old:

search_sectors_from_filetype_and_path_OLD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L4837>`__

**Description:** Discover which TESS sectors have processed light curves on disk.

.. code-block:: python

   def search_sectors_from_filetype_and_path_OLD(ID, default_settings): ...

Class methods
-------------

.. _api-nemesis-tess-pipeline-cachelock-enter:

_CacheLock.__enter__
~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L2577>`__

**Description:** No description available in the source docstring.

.. code-block:: python

   def _CacheLock.__enter__(self): ...

.. _api-nemesis-tess-pipeline-cachelock-exit:

_CacheLock.__exit__
~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L2591>`__

**Description:** No description available in the source docstring.

.. code-block:: python

   def _CacheLock.__exit__(self, exc_type, exc, tb): ...

.. _api-nemesis-tess-pipeline-cachelock-init:

_CacheLock.__init__
~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L2572>`__

**Description:** No description available in the source docstring.

.. code-block:: python

   def _CacheLock.__init__(self, cache_root: str, name: str, max_wait_s: float=8.0): ...

.. _api-nemesis-tess-pipeline-tesscutcountersemaphore-enter:

TesscutCounterSemaphore.__enter__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L2665>`__

**Description:** No description available in the source docstring.

.. code-block:: python

   def TesscutCounterSemaphore.__enter__(self): ...

.. _api-nemesis-tess-pipeline-tesscutcountersemaphore-exit:

TesscutCounterSemaphore.__exit__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L2685>`__

**Description:** No description available in the source docstring.

.. code-block:: python

   def TesscutCounterSemaphore.__exit__(self, exc_type, exc, tb) -> None: ...

.. _api-nemesis-tess-pipeline-tesscutcountersemaphore-init:

TesscutCounterSemaphore.__init__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L2635>`__

**Description:** No description available in the source docstring.

.. code-block:: python

   def TesscutCounterSemaphore.__init__(self, cache_root, max_slots: int=4, poll_interval: float=0.5, max_wait: float | None=None) -> None: ...

.. _api-nemesis-tess-pipeline-tesscutcountersemaphore-read-count:

TesscutCounterSemaphore._read_count
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L2651>`__

**Description:** No description available in the source docstring.

.. code-block:: python

   def TesscutCounterSemaphore._read_count(self) -> int: ...

.. _api-nemesis-tess-pipeline-tesscutcountersemaphore-write-count:

TesscutCounterSemaphore._write_count
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location:** ``nemesis_tess.pipeline`` — `View source <https://github.com/daxfeliz/NEMESIS/blob/main/src/nemesis_tess/pipeline.py#L2661>`__

**Description:** No description available in the source docstring.

.. code-block:: python

   def TesscutCounterSemaphore._write_count(self, value: int) -> None: ...
