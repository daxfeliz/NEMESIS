Alphabetical function and class index
=====================================

An alphabetical inventory of the function names, classes, and class
methods defined in the installed NEMESIS source. Names link to the
:doc:`API reference <api_reference>`, which shows signatures and
GitHub source locations. This is an **inventory, not a guarantee of
public API stability**; leading-underscore and legacy-named entries
are generally internal. The separate example wrappers are excluded.

A
-

- :ref:`_apply_outlier_flag <api-nemesis-tess-pipeline-apply-outlier-flag>` — Set QUALITY\_FLAG = 3 on cadences identified by outlier\_removal().
- :ref:`align_arrays <api-nemesis-tess-pipeline-align-arrays>` — Align the model flux to the data flux by interpolating the model flux to match the phase values of the data.
- :ref:`all_transit_times <api-nemesis-tess-pipeline-all-transit-times>` — Return all predicted mid-transit times within the time span of t.
- :ref:`all_transit_times_OLD <api-nemesis-tess-pipeline-all-transit-times-old>` — Return all mid-transit times within t
- :ref:`append_PDF <api-nemesis-tess-pipeline-append-pdf>` — See source for parameters and behavior.
- :ref:`apply_nan_to_flagged_rows <api-nemesis-tess-pipeline-apply-nan-to-flagged-rows>` — NaN non-preserved columns on cadences whose QUALITY\_FLAG indicates flux-corrupting events.
- :ref:`Applying_Mdump_removal <api-nemesis-tess-pipeline-applying-mdump-removal>` — Apply momentum dump and sector jitter masking to a raw SAP light curve.
- :ref:`asymmetric <api-nemesis-tess-pipeline-asymmetric>` — FA check: left/right transit asymmetry.

B
-

- :ref:`_bitmask_to_flag_ids <api-nemesis-tess-pipeline-bitmask-to-flag-ids>` — Convert stored bitmask value into flag identities.
- :ref:`_bitmask_to_flag_string <api-nemesis-tess-pipeline-bitmask-to-flag-string>` — Human-readable representation for CSV export.
- :ref:`_bootstrap_fap <api-nemesis-tess-pipeline-bootstrap-fap>` — Estimate FAP by counting how often a shuffled light curve exceeds the observed peak power. O(n\_bootstrap × N\_freq) — keep n\_bootstrap ≤ 200.
- :ref:`bad_fit <api-nemesis-tess-pipeline-bad-fit>` — FA check: poor fit of the trapezoidal transit model.
- :ref:`bad_shape <api-nemesis-tess-pipeline-bad-shape>` — FA check: flux increases within the putative transit (bad transit shape).
- :ref:`bin_ff <api-nemesis-tess-pipeline-bin-ff>` — No description available in the source docstring.
- :ref:`Bin_func <api-nemesis-tess-pipeline-bin-func>` — This function will produced a light curve that is binned with a weighted bins.
- :ref:`bin_quaternion_features <api-nemesis-tess-pipeline-bin-quaternion-features>` — Bin 2-sec quaternion data to LC cadence and compute the 36 QLP feature series following Kunimoto et al. (2022), arXiv:2211.04387.
- :ref:`BLS_func <api-nemesis-tess-pipeline-bls-func>` — Run the Astropy BLS algorithm and assemble all NEMESIS transit search outputs.

C
-

- :ref:`_cache_lock_name <api-nemesis-tess-pipeline-cache-lock-name>` — Build a narrow-scope lock filename so only jobs touching the same resource serialize. Examples: flatten\_TESSCut\_s0001\_cam1\_ccd4.lock flatten\_TESS\_s0001\_tic\_24195569.lock flatten\_TESSCut\_generic.lock (fallback)
- :ref:`_cache_root_from_settings <api-nemesis-tess-pipeline-cache-root-from-settings>` — Resolve the canonical cache root (your existing cache path). All cache activity (TESScut/TPF/DSS) uses this path.
- :ref:`_CacheLock <api-nemesis-tess-pipeline-cachelock>` — Best-effort file lock based on creating a lockfile. Scope is controlled by the lock name you pass in. Lock file is removed on exit; failures to remove are ignored.
- :ref:`_CacheLock.__enter__ <api-nemesis-tess-pipeline-cachelock-enter>` — No description available in the source docstring.
- :ref:`_CacheLock.__exit__ <api-nemesis-tess-pipeline-cachelock-exit>` — No description available in the source docstring.
- :ref:`_CacheLock.__init__ <api-nemesis-tess-pipeline-cachelock-init>` — No description available in the source docstring.
- :ref:`_check_harmonic <api-nemesis-tess-pipeline-check-harmonic>` — Check whether the 2P candidate has higher power than P, which would indicate p\_peak is actually P\_rot/2.
- :ref:`_compute_per_cadence_centroid <api-nemesis-tess-pipeline-compute-per-cadence-centroid>` — Compute per-cadence quadratic centroids using an independent loose threshold mask, NOT the PRF pix\_mask, eliminating the self-referential bias present in the original SAP\_ALT2 Step 4 inline centroid loop.
- :ref:`calc_flux_contamination <api-nemesis-tess-pipeline-calc-flux-contamination>` — Compute the flux contamination ratio and its 1-sigma uncertainty for a target star given nearby TIC sources.
- :ref:`calc_mean_depths <api-nemesis-tess-pipeline-calc-mean-depths>` — See source for parameters and behavior.
- :ref:`calc_transit_times <api-nemesis-tess-pipeline-calc-transit-times>` — Return all mid-transit times within t
- :ref:`calculate_semi_major_axis <api-nemesis-tess-pipeline-calculate-semi-major-axis>` — Calculate the semi-major axis using Kepler's third law.
- :ref:`CDPP <api-nemesis-tess-pipeline-cdpp>` — Compute the Combined Differential Photometric Precision (CDPP) of a light curve.
- :ref:`center_cutout <api-nemesis-tess-pipeline-center-cutout>` — For TESS Full Frame Images (FFI), this function will try to center the FFI on the target star, much like how Target Pixel Files (TPF) are centered on the target star.
- :ref:`centroid_analysis <api-nemesis-tess-pipeline-centroid-analysis>` — Compute a centroid motion diagnostic for a NEMESIS TCE and render it on the Page-4 vetting report axes.
- :ref:`centroid_analysis_OLD <api-nemesis-tess-pipeline-centroid-analysis-old>` — See source for parameters and behavior.
- :ref:`centroid_quadratic <api-nemesis-tess-pipeline-centroid-quadratic>` — This function computes the quadratic estimate of the centroid in a 2d-array. This method will fit a simple 2D second-order polynomial $P(x, y) = a + bx + cy + dx^2 + exy + fy^2$ to the 3x3 patch of pixels centered on…
- :ref:`chases <api-nemesis-tess-pipeline-chases>` — FA check: events not unique in the local light curve (few transits).
- :ref:`check_ALL_thresholds <api-nemesis-tess-pipeline-check-all-thresholds>` — Run all 17–18 LEO-Vetter FA/FP checks and return an aggregate flag.
- :ref:`check_centroids <api-nemesis-tess-pipeline-check-centroids>` — This function will compare the location of the centroids produced by the 'SAP' function with the location of the reference pixel.
- :ref:`check_edge <api-nemesis-tess-pipeline-check-edge>` — This function is used to check if a polygon contains the edges of a Full Frame Image (FFI) or Target Pixel File (TPF).
- :ref:`check_imports_are_installed <api-nemesis-tess-pipeline-check-imports-are-installed>` — Try to import every module (and submodule) that NEMESIS uses in NEMESIS\_pipeline\_new\_temp\_2.py. Prints a compact report and returns (missing, imported) where each is a list of module strings.
- :ref:`check_transit_symmetry <api-nemesis-tess-pipeline-check-transit-symmetry>` — Assess left/right transit symmetry using uncertainty-normalised (SNR-on-the-asymmetry) tests for both transit depth and fractional transit duration (q\_tran).
- :ref:`check_transit_symmetry_OLD <api-nemesis-tess-pipeline-check-transit-symmetry-old>` — Checks the symmetry of the transit by comparing the left and right mirrored transit depths and durations. Returns a pass/fail result based on predefined thresholds.
- :ref:`check_wcs <api-nemesis-tess-pipeline-check-wcs>` — Returns an astropy.wcs.WCS for the input TESS FFI/TPF.
- :ref:`chi <api-nemesis-tess-pipeline-chi>` — FA check: inconsistent per-transit SNRs (χ² statistic).
- :ref:`classify_retrieval_error <api-nemesis-tess-pipeline-classify-retrieval-error>` — Sort a cutout-retrieval exception into a plain-language cause.
- :ref:`cleanup <api-nemesis-tess-pipeline-cleanup>` — Conditionally delete NEMESIS intermediate and ancillary data products at the end of a pipeline run.
- :ref:`collect_nearby_Gaia_params <api-nemesis-tess-pipeline-collect-nearby-gaia-params>` — Query Vizier (Gaia DR3 by default) for sources within the TPF/FFI footprint and restrict the returned catalog to those whose (optionally PM-corrected) positions project inside the cutout's rectangular bounding box.
- :ref:`collect_nearby_TIC_params <api-nemesis-tess-pipeline-collect-nearby-tic-params>` — Queries the TESS Input Catalog (TIC) for stellar parameters of the target star and nearby stars within a given radius. The radius is defined by the number of TESS pixels.
- :ref:`compare_to_model <api-nemesis-tess-pipeline-compare-to-model>` — No description available in the source docstring.
- :ref:`compute_prf_aperture_weights <api-nemesis-tess-pipeline-compute-prf-aperture-weights>` — Compute a PRF-based soft aperture weight map for a TESS TESScut cutout.
- :ref:`compute_sde_optimized <api-nemesis-tess-pipeline-compute-sde-optimized>` — Computes an array of SDE values, ensuring that max(SDE array) matches the classical SDE.
- :ref:`compute_tce_coverage_metrics <api-nemesis-tess-pipeline-compute-tce-coverage-metrics>` — Compute coverage-related vetting metrics for a single-sector TCE.
- :ref:`consecutive <api-nemesis-tess-pipeline-consecutive>` — This function will search input data for points that are consecutive.
- :ref:`convert_TessMag_to_Flux <api-nemesis-tess-pipeline-convert-tessmag-to-flux>` — This function will convert TESS Input Catalog (TIC) Magnitude into TESS Flux.
- :ref:`convert_window_size_in_days_to_points <api-nemesis-tess-pipeline-convert-window-size-in-days-to-points>` — This function converts a window size into a window size in number of data points, based on the cadence of timestamps in input time array.
- :ref:`create_box_model <api-nemesis-tess-pipeline-create-box-model>` — Create boxcar model.
- :ref:`create_box_model_for_phasefolded <api-nemesis-tess-pipeline-create-box-model-for-phasefolded>` — Create boxcar model.
- :ref:`create_circular_mask <api-nemesis-tess-pipeline-create-circular-mask>` — Build a boolean circular aperture mask centered on reference\_pixel.
- :ref:`create_circular_mask_OLD <api-nemesis-tess-pipeline-create-circular-mask-old>` — Create a circular boolean aperture mask centred on a reference pixel.
- :ref:`create_PDF_from_PNG <api-nemesis-tess-pipeline-create-pdf-from-png>` — See source for parameters and behavior.
- :ref:`current_tesscut_active <api-nemesis-tess-pipeline-current-tesscut-active>` — Return approximate number of active TESScut/TESSSearch slots.
- :ref:`cut_around_T0 <api-nemesis-tess-pipeline-cut-around-t0>` — See source for parameters and behavior.

D
-

- :ref:`_decorate_quality_columns <api-nemesis-tess-pipeline-decorate-quality-columns>` — Ensure QUALITY\_FLAG is int16 bitmask and attach readable QUALITY\_FLAGS.
- :ref:`_detect_deep_dips <api-nemesis-tess-pipeline-detect-deep-dips>` — Flag downward outliers in PLD residuals as candidate transit cadences.
- :ref:`data_gapped <api-nemesis-tess-pipeline-data-gapped>` — FA check: too many transits fall near data gaps.
- :ref:`default_NEMESIS_pipeline_settings <api-nemesis-tess-pipeline-default-nemesis-pipeline-settings>` **[public]** — Return the default settings dictionary for the NEMESIS pipeline.
- :ref:`df_len_check <api-nemesis-tess-pipeline-df-len-check>` — Print the row count of each column in a DataFrame for pipeline debugging.
- :ref:`dmm <api-nemesis-tess-pipeline-dmm>` — FA check: inconsistency between mean and median transit depths.

E
-

- :ref:`_ensure_hdu_or_raise <api-nemesis-tess-pipeline-ensure-hdu-or-raise>` — Raise NoImageDataError if nothing usable is in hdu.
- :ref:`_ensure_hdu_or_raise_OLD <api-nemesis-tess-pipeline-ensure-hdu-or-raise-old>` — Raise NoImageDataError if nothing usable is in hdu.
- :ref:`_ensure_tesscut_dirs <api-nemesis-tess-pipeline-ensure-tesscut-dirs>` — Make sure astroquery's expected dirs exist under the cache tree. Avoids FileNotFoundError for .../mastDownload/TESSCut/tesscut\_\*.zip
- :ref:`edge_check <api-nemesis-tess-pipeline-edge-check>` — Check whether a target star falls within 6 pixels of a TESS detector edge.
- :ref:`edgeWarn_tesspt2px <api-nemesis-tess-pipeline-edgewarn-tesspt2px>` — Checks if a star (based on RA/Dec) is near the edge of a TESS camera for the given sectors.
- :ref:`extract_background_flux <api-nemesis-tess-pipeline-extract-background-flux>` — Per-frame, sigma-clipped background estimation and raw SAP flux extraction.

F
-

- :ref:`_filter_lc_by_quality_flags <api-nemesis-tess-pipeline-filter-lc-by-quality-flags>` — Return a copy of lc\_df keeping only rows whose QUALITY\_FLAG is consistent with the requested allowed flag-ids.
- :ref:`_flag_ids_to_bitmask <api-nemesis-tess-pipeline-flag-ids-to-bitmask>` — Convert flag identities like [1,2,3] into bitmask sum 1+2+4 = 7.
- :ref:`_flatten_to_cache <api-nemesis-tess-pipeline-flatten-to-cache>` — Move files downloaded by 'lksearch' into a cache/mastDownload directory.
- :ref:`_fmt_elapsed <api-nemesis-tess-pipeline-fmt-elapsed>` — Format a duration for log lines: seconds below 2 minutes, else minutes.
- :ref:`filter_phase_range <api-nemesis-tess-pipeline-filter-phase-range>` — No description available in the source docstring.
- :ref:`find_min_trial_duration <api-nemesis-tess-pipeline-find-min-trial-duration>` — No description available in the source docstring.
- :ref:`find_tesscut_file <api-nemesis-tess-pipeline-find-tesscut-file>` — Return every file whose name matches tess-s00[sector]-[camera]-[ccd]\_\*[size]x[size]\*astrocut.fits anywhere under 'basedir'.
- :ref:`find_tpf_file <api-nemesis-tess-pipeline-find-tpf-file>` — Recursively find SPOC Target Pixel Files (TPFs) on disk.
- :ref:`fix_wcs_axis_labels <api-nemesis-tess-pipeline-fix-wcs-axis-labels>` — Configure WCS-projected axes so RA labels appear on the bottom edge and Dec labels appear on the left edge, with ~3 tick marks per axis.
- :ref:`format_ax <api-nemesis-tess-pipeline-format-ax>` — This function will format the subplot used for plotting the input TESS Full Frame Image or Target Pixel File (FFI/TPF).
- :ref:`format_conditions_block <api-nemesis-tess-pipeline-format-conditions-block>` — Format a multi-line LEO-Vetter condition trace as an indented text block.
- :ref:`full_pipeline <api-nemesis-tess-pipeline-full-pipeline>` **[public]** — End-to-end NEMESIS pipeline: FFI/TPF retrieval → SAP → PLD → detrending → outlier removal → BLS/TLS transit search → vetting reports.
- :ref:`fullphasefold <api-nemesis-tess-pipeline-fullphasefold>` — Phase-fold a light curve onto the full ''[0, 1)'' orbital phase interval.

G
-

- :ref:`_get_cached_fits <api-nemesis-tess-pipeline-get-cached-fits>` — Fast lookup of a cached TESScut FITS using deterministic paths first, then a tiny per-key SQLite index to avoid directory-wide scans.
- :ref:`_gethdu_diag_key <api-nemesis-tess-pipeline-gethdu-diag-key>` — Build the (TIC, sector) key used by the retrieval record.
- :ref:`_gethdu_lksearch_unverified <api-nemesis-tess-pipeline-gethdu-lksearch-unverified>` — Download a single TESS FFI cutout or TPF from MAST via lksearch.
- :ref:`_gethdu_lksearch_verified <api-nemesis-tess-pipeline-gethdu-lksearch-verified>` — Return the cutout for one TIC ID and one sector, verified to be that star and that sector, or None if no verified cutout can be obtained.
- :ref:`generate_injection_path <api-nemesis-tess-pipeline-generate-injection-path>` — Generates the appropriate file path based on the given parameters.
- :ref:`get_coordinates <api-nemesis-tess-pipeline-get-coordinates>` — Returns two 3D arrays of RA and Dec values in decimal degrees. If cadence number is given, returns 2D arrays for that cadence. If cadence is 'all' returns one RA, Dec value for each pixel in every cadence. Uses the…
- :ref:`get_flux_at_TN <api-nemesis-tess-pipeline-get-flux-at-tn>` — Get the flux value closest to a specific transit contact point (T1, T4, etc.) based on a phase-folded light curve (pf, ff). The window is used to limit the search to values close to the contact point (TN).
- :ref:`get_GAIA_DR2_sources <api-nemesis-tess-pipeline-get-gaia-dr2-sources>` — This function will query the GAIA catalog for nearby sources using a radial cone search.
- :ref:`get_gain_and_readout_from_hdu <api-nemesis-tess-pipeline-get-gain-and-readout-from-hdu>` — No description available in the source docstring.
- :ref:`get_LC_from_FFI <api-nemesis-tess-pipeline-get-lc-from-ffi>` — Run the full per-sector NEMESIS LC extraction pipeline on a single HDU.
- :ref:`get_momentum_dump_times <api-nemesis-tess-pipeline-get-momentum-dump-times>` — This function will do a bitmask search in the input TESS Full Frame Image or Target Pixel File (FFI/TPF) image headers for momentum dump timestamps that occurred during TESS observations.
- :ref:`get_pix_and_bkg_masks_and_image_fluxes <api-nemesis-tess-pipeline-get-pix-and-bkg-masks-and-image-fluxes>` — This function retrieves the pixel and background masks and the TESS image fluxes for a target star.
- :ref:`get_PLD_model <api-nemesis-tess-pipeline-get-pld-model>` — Compute the additive PLD systematics model via weighted least squares.
- :ref:`get_polygon <api-nemesis-tess-pipeline-get-polygon>` — This function will produce a Shapely Polygon object that outlines the input pixel mask.
- :ref:`get_qdata <api-nemesis-tess-pipeline-get-qdata>` — Download or load the TESS engineering quaternion FITS for a given sector from the MAST archive.
- :ref:`get_qld <api-nemesis-tess-pipeline-get-qld>` — Retrieves quadratic limb darkening coefficients 'a' and 'b' for a given star, based on its effective temperature and surface gravity.
- :ref:`get_scalar <api-nemesis-tess-pipeline-get-scalar>` — Coerce any scalar-like object to a plain Python scalar.
- :ref:`get_sources_in_aperture_mask <api-nemesis-tess-pipeline-get-sources-in-aperture-mask>` — This function will query the TESS Input Catalog (TIC) for nearby sources using a radial cone search and check which sources are inside the input aperture pixel mask.
- :ref:`get_tess_reference_pixel <api-nemesis-tess-pipeline-get-tess-reference-pixel>` — Return 0-indexed pixel-center coordinates [col, row] from a TESS binary-table HDU header.
- :ref:`get_TIC_ID <api-nemesis-tess-pipeline-get-tic-id>` — Retrieves the TIC (TESS Input Catalog) ID of a given target star by querying the MAST Catalogs. If the star has a duplicate ID, the duplicate is returned. The search is conducted within a 5-arcsecond radius of the…
- :ref:`get_TOI_and_CTOI_catalogs <api-nemesis-tess-pipeline-get-toi-and-ctoi-catalogs>` — See source for parameters and behavior.
- :ref:`gethdu <api-nemesis-tess-pipeline-gethdu>` — Download TESS FFI/TPF cutouts from MAST for one or more sectors and return all metadata needed for downstream SAP.
- :ref:`gethdu_failure_summary <api-nemesis-tess-pipeline-gethdu-failure-summary>` — One line saying why a TIC and sector ended with no image data.
- :ref:`gethdu_lksearch <api-nemesis-tess-pipeline-gethdu-lksearch>` — Return the verified cutout for one TIC ID and one sector, exactly as ''\_gethdu\_lksearch\_verified'' does, and log why when none comes back.
- :ref:`gethdu_lksearch_OLD <api-nemesis-tess-pipeline-gethdu-lksearch-old>` — Return the cutout for one TIC ID and one sector, verified to be that star and that sector, or None if no verified cutout can be obtained.

H
-

- :ref:`HWHM <api-nemesis-tess-pipeline-hwhm>` — This function will calculated the Half-Width Half-Max (HWHM) of an input BLS\_periods, BLS\_SDEs data. This is used to estimate the uncertainty of the orbital period in the BLS power spectrum.

I
-

- :ref:`in_versus_out_of_transit_test <api-nemesis-tess-pipeline-in-versus-out-of-transit-test>` — See source for parameters and behavior.
- :ref:`intransit_stats <api-nemesis-tess-pipeline-intransit-stats>` — Compute per-transit and odd/even transit statistics for a BLS or TLS result.
- :ref:`invalid_transits <api-nemesis-tess-pipeline-invalid-transits>` — FA check: insufficient valid transits after removing bad events.
- :ref:`is_point_inside_polygon <api-nemesis-tess-pipeline-is-point-inside-polygon>` — This function will check if the input pixel coordinates are within the input Shapely polygon object.
- :ref:`is_TOI_or_CTOI <api-nemesis-tess-pipeline-is-toi-or-ctoi>` — See source for parameters and behavior.

L
-

- :ref:`_ls_results_for_flux <api-nemesis-tess-pipeline-ls-results-for-flux>` — Compute all Lomb-Scargle metrics for a single flux array.
- :ref:`large <api-nemesis-tess-pipeline-large>` — FP check: planet candidate radius too large (likely an eclipsing binary).
- :ref:`LC_creation <api-nemesis-tess-pipeline-lc-creation>` **[public]** — Produce processed TESS light curves for one or more sectors using the NEMESIS pipeline (Feliz et al. 2021, https://arxiv.org/abs/2103.05647).
- :ref:`LC_creation_OLD <api-nemesis-tess-pipeline-lc-creation-old>` — Produce processed TESS light curves for one or more sectors using the NEMESIS pipeline (Feliz et al. 2021, https://arxiv.org/abs/2103.05647).
- :ref:`LC_to_LS <api-nemesis-tess-pipeline-lc-to-ls>` — Render the LS periodogram and phase-folded LC panels for vetting report page 3.
- :ref:`lin_interp <api-nemesis-tess-pipeline-lin-interp>` — This function is used to perform a simple linear interpolation.
- :ref:`load_camera_quaternions <api-nemesis-tess-pipeline-load-camera-quaternions>` — Extract TIME, Q1, Q2, Q3 for a given camera from a quaternion HDUList.
- :ref:`load_triceratops <api-nemesis-tess-compat-load-triceratops>` — Import TRICERATOPS with compatibility fixes for PyTransit 2.2.0.
- :ref:`LS <api-nemesis-tess-pipeline-ls>` — Compute a generalized Lomb-Scargle periodogram with optional multi-term fitting.

M
-

- :ref:`Make_dirs <api-nemesis-tess-pipeline-make-dirs>` — Create all NEMESIS output directories and write resolved path strings back into ''default\_settings''.
- :ref:`mann2019_mass_radius <api-nemesis-tess-pipeline-mann2019-mass-radius>` — Estimate M-dwarf stellar mass and radius from absolute 2MASS K\_s magnitude.
- :ref:`mann2019_mass_radius_OLD <api-nemesis-tess-pipeline-mann2019-mass-radius-old>` — Estimate M-dwarf stellar mass and radius from absolute K-band magnitude using the empirical polynomial relations of Mann et al. 2019 (mass) and Mann et al. 2015 (radius).
- :ref:`mask_out_neighbor_pixels <api-nemesis-tess-pipeline-mask-out-neighbor-pixels>` — Excise aperture pixels contaminated by neighboring stars.
- :ref:`mask_out_neighbor_pixels_OLD <api-nemesis-tess-pipeline-mask-out-neighbor-pixels-old>` — Excise aperture pixels contaminated by neighboring stars, with a separation-based gate that escalates sub-pixel blends to PRF decontamination rather than attempting pixel excision.
- :ref:`mass_radius_mdwarf_relation <api-nemesis-tess-pipeline-mass-radius-mdwarf-relation>` — Adjusts the mass and radius of an M-dwarf star based on the given inputs. This function uses an OVERLY simple linear mass-radius relationship for M-dwarfs, where if one of the stellar parameters (mass or radius) is…
- :ref:`mirror_and_model_left_right_transit <api-nemesis-tess-pipeline-mirror-and-model-left-right-transit>` — Performs a comprehensive symmetry analysis for a transit event by comparing the left and right mirrored transits to the model and computing various metrics such as RMSE, residuals, and chi-squared values. Also…
- :ref:`momentumdump_removal <api-nemesis-tess-pipeline-momentumdump-removal>` — This function will remove windows of data around momentum dump time stamps and return a mask.

N
-

- :ref:`_nemesis_skyview_stamp <api-nemesis-tess-pipeline-nemesis-skyview-stamp>` — Fetch a sky survey image stamp and return ''(stamp, header)''.
- :ref:`_normalize_gethdu_outputs <api-nemesis-tess-pipeline-normalize-gethdu-outputs>` — Ensure that all five ''gethdu'' return values are Python \*lists\* of equal length, regardless of whether one or multiple sectors were returned.
- :ref:`nan_symmetry_result <api-nemesis-tess-pipeline-nan-symmetry-result>` — Return a NaN-filled result with the exact 16-element shape that ''mirror\_and\_model\_left\_right\_transit'' returns on success.
- :ref:`nan_vetting_columns <api-nemesis-tess-pipeline-nan-vetting-columns>` — NaN-filled replacement for the vetting-metric column block that ''in\_versus\_out\_of\_transit\_test'' normally appends to the bestfit frame.
- :ref:`nanchecker <api-nemesis-tess-pipeline-nanchecker>` — Adjusts the standard deviation threshold to remove NaNs from mirrored fluxes.
- :ref:`NoImageDataError <api-nemesis-tess-pipeline-noimagedataerror>` **[public]** — Raised when no TESScut/TPF/FFI image data are available for a TIC+Sector+cadence.
- :ref:`non_unique <api-nemesis-tess-pipeline-non-unique>` — FA check: events not unique in the phased light curve.

O
-

- :ref:`odd_even <api-nemesis-tess-pipeline-odd-even>` — FP check: significant odd/even transit depth mismatch (EB signature).
- :ref:`odd_even_cuts <api-nemesis-tess-pipeline-odd-even-cuts>` — See source for parameters and behavior.
- :ref:`odd_even_mismatch_plot <api-nemesis-tess-pipeline-odd-even-mismatch-plot>` — See source for parameters and behavior.
- :ref:`offset <api-nemesis-tess-pipeline-offset>` — FP check: centroid offset indicates an off-target eclipsing binary.
- :ref:`optimize_period_duration_grid_sizes <api-nemesis-tess-pipeline-optimize-period-duration-grid-sizes>` — No description available in the source docstring.
- :ref:`outlier_removal <api-nemesis-tess-pipeline-outlier-removal>` — Sliding-window sigma-clipping outlier removal on the detrended light curve.

P
-

- :ref:`_per_transit_snr <api-nemesis-tess-pipeline-per-transit-snr>` — Compute a simple per-transit SNR for a single event.
- :ref:`_prf_weighted_flux_pass <api-nemesis-tess-pipeline-prf-weighted-flux-pass>` — Per-cadence PRF-weighted flux extraction with corrected normalization and improved acceptance criteria. Replaces the Step 4b inline block in the original SAP\_ALT2.
- :ref:`p2p_rms <api-nemesis-tess-pipeline-p2p-rms>` — Compute the point-to-point RMS of a light curve as a robust noise proxy.
- :ref:`patch_EDIVetter_Unplugged <api-nemesis-tess-pipeline-patch-edivetter-unplugged>` — Patch the EDIunplugged package by replacing deprecated 'np.float' with 'float' in Python scripts. This patch is only applied if the installed version of EDIunplugged is "0.1.3".
- :ref:`pd_to_lk <api-nemesis-tess-pipeline-pd-to-lk>` — Convert a NEMESIS ''LC\_df'' DataFrame to a ''lightkurve.LightCurve'' object.
- :ref:`phasefold <api-nemesis-tess-pipeline-phasefold>` — Phase-fold a light curve with the transit centred at phase zero.
- :ref:`pipeline_quality_mask <api-nemesis-tess-pipeline-pipeline-quality-mask>` — Applies a quality mask to a given TESS data quality array based on the desired bitmask setting.
- :ref:`PLD_model <api-nemesis-tess-pipeline-pld-model>` — Two-pass Pixel Level Decorrelation (PLD) with GP-proxy variance inflation.
- :ref:`plot_centroids_in_phase <api-nemesis-tess-pipeline-plot-centroids-in-phase>` — See source for parameters and behavior.
- :ref:`plot_cutouts_ALT <api-nemesis-tess-pipeline-plot-cutouts-alt>` — Drop-in alternative to ''plot\_cutouts'' with a ''catalog\_source'' switch.
- :ref:`plot_cutouts_ALTb <api-nemesis-tess-pipeline-plot-cutouts-altb>` — Drop-in alternative to ''plot\_cutouts'' with a ''catalog\_source'' switch.
- :ref:`plot_cutouts_OLD <api-nemesis-tess-pipeline-plot-cutouts-old>` — This function will plot the median flux image of the input Full Frame Image or Target Pixel File (FFI/TPF) along with the input photometric aperture and background flux pixel masks.
- :ref:`plot_dss_orientation <api-nemesis-tess-pipeline-plot-dss-orientation>` — Fetch a DSS optical stamp, reproject it onto the TESS WCS pixel grid, and render it with N/E arrows and TESS aperture/background mask overlays.
- :ref:`plot_dss_orientation_OLD <api-nemesis-tess-pipeline-plot-dss-orientation-old>` — Fetch DSS and draw N/E. If anything fails (DNS/HTTP/WCS/FITS), place 'No DSS image' text and return cleanly. Uses the SAME cache root as TESScut/TPF (downloadpath).
- :ref:`plot_dss_sources <api-nemesis-tess-pipeline-plot-dss-sources>` — Overlay catalog sources on the reprojected DSS panel.
- :ref:`plot_image <api-nemesis-tess-pipeline-plot-image>` — See source for parameters and behavior.
- :ref:`plot_image_ALT <api-nemesis-tess-pipeline-plot-image-alt>` — Plot the TESS FFI/TPF median-image cutout with aperture/background masks and catalog source overlay (TIC or Gaia DR3).
- :ref:`plot_image_and_mask <api-nemesis-tess-pipeline-plot-image-and-mask>` — This function will plot the Full Frame Image or Target Pixel File (FFI/TPF) images and input aperture and background pixel masks.
- :ref:`plot_image_and_mask_OLD <api-nemesis-tess-pipeline-plot-image-and-mask-old>` — This function will plot the Full Frame Image or Target Pixel File (FFI/TPF) images and input aperture and background pixel masks.
- :ref:`plot_it_all_up <api-nemesis-tess-pipeline-plot-it-all-up>` — Unified LC summary figure for both single-sector and multi-sector runs.
- :ref:`plot_LEO_Vetter_text_results <api-nemesis-tess-pipeline-plot-leo-vetter-text-results>` — No description available in the source docstring.
- :ref:`plot_mask <api-nemesis-tess-pipeline-plot-mask>` — This function is used to plot a pixel mask on a subplot.
- :ref:`plot_mirrored_transits_and_collect_vetting_metrics <api-nemesis-tess-pipeline-plot-mirrored-transits-and-collect-vetting-metrics>` — See source for parameters and behavior.
- :ref:`plot_modshift_NEMESIS <api-nemesis-tess-pipeline-plot-modshift-nemesis>` — Compute LEO-Vetter transit metrics and render the modshift diagnostic figure.
- :ref:`plot_odd_even_transits <api-nemesis-tess-pipeline-plot-odd-even-transits>` — This function will plot the odd and even numbered transits that are modeled by TLS in the input subplot of a pyplot figure.
- :ref:`plot_orientation <api-nemesis-tess-pipeline-plot-orientation>` — This function will use the TESS Full Frame Image (FFI) or Target Pixel File (TPF) image headers to determine the North and East orientation for plotting the median image flux.
- :ref:`plot_page3_text <api-nemesis-tess-pipeline-plot-page3-text>` — See source for parameters and behavior.
- :ref:`plot_phasefold_LCs <api-nemesis-tess-pipeline-plot-phasefold-lcs>` — See source for parameters and behavior.
- :ref:`plot_power_spectra <api-nemesis-tess-pipeline-plot-power-spectra>` — This function will plot the TLS power spectrum resulting from the TLS transit search.
- :ref:`plot_sources <api-nemesis-tess-pipeline-plot-sources>` — This function will plot catalog sources and centroid positions.
- :ref:`plot_sources_gaia <api-nemesis-tess-pipeline-plot-sources-gaia>` — Overlay Gaia DR3 sources on a TESS cutout with PM correction to the cutout epoch.
- :ref:`plot_sources_TIC <api-nemesis-tess-pipeline-plot-sources-tic>` — Tpfplotter-style TIC source overlay on a TESS cutout.
- :ref:`plot_text <api-nemesis-tess-pipeline-plot-text>` — See source for parameters and behavior.
- :ref:`preclean_SAP_BKG_Centroids <api-nemesis-tess-pipeline-preclean-sap-bkg-centroids>` — Pre-clean SAP flux, background, and centroids before PLD/detrending.
- :ref:`prefetch_prf <api-nemesis-tess-pipeline-prefetch-prf>` — Pre-download TESS PRF FITS files from MAST for the epochs required by ''sectors''.
- :ref:`prefetch_quaternions <api-nemesis-tess-pipeline-prefetch-quaternions>` — Pre-download and cache TESS engineering quaternion FITS files for every sector in \*sectors\* before LC creation begins.

Q
-

- :ref:`quick_outlier_removal <api-nemesis-tess-pipeline-quick-outlier-removal>` — Fast sliding-window outlier mask for a single flux column.

R
-

- :ref:`_record_gethdu_diag <api-nemesis-tess-pipeline-record-gethdu-diag>` — Store the latest retrieval outcome for one (TIC, sector).
- :ref:`_resolve_prf_cache_dir <api-nemesis-tess-pipeline-resolve-prf-cache-dir>` — Resolve the PRF cache directory from NEMESIS settings.
- :ref:`_resolve_quat_savepath <api-nemesis-tess-pipeline-resolve-quat-savepath>` — Derive a canonical quaternion cache directory from default\_settings and write it back as default\_settings['quat\_savepath'].
- :ref:`_resolve_target_sectors <api-nemesis-tess-pipeline-resolve-target-sectors>` — Return the concrete list of sector ints that full\_pipeline will process.
- :ref:`_restore_full_length_lc <api-nemesis-tess-pipeline-restore-full-length-lc>` — Rebuild a full-length LC dataframe from compact good-cadence results.
- :ref:`_run_cutouts_ALT <api-nemesis-tess-pipeline-run-cutouts-alt>` — Internal helper called by plot\_image\_ALT. Runs the full cutout plot and returns (gaia\_df, target\_gaia\_id).
- :ref:`readNDarr <api-nemesis-tess-pipeline-readndarr>` — Deserialize a pickle (.pkl) file written by ''saveNDarr''.
- :ref:`recursive_search <api-nemesis-tess-pipeline-recursive-search>` — Recursively find files under a directory tree matching a filename pattern.
- :ref:`refine_transit_epoch_post_search <api-nemesis-tess-pipeline-refine-transit-epoch-post-search>` — Refine a detected transit epoch after BLS/TLS using a guarded box-matched depth/SNR score.
- :ref:`ResampleLinear1D <api-nemesis-tess-pipeline-resamplelinear1d>` — No description available in the source docstring.
- :ref:`robust_transit_depth <api-nemesis-tess-pipeline-robust-transit-depth>` — Robust estimation of transit depth using sigma clipping, quantiles, and fallback strategies.
- :ref:`running_mean <api-nemesis-tess-pipeline-running-mean>` — This function will calculate a mean value for a given window of time.
- :ref:`running_median <api-nemesis-tess-pipeline-running-median>` — This function will calculate a median value for a given window of time.

S
-

- :ref:`_sec_str <api-nemesis-tess-pipeline-sec-str>` — Safe sector string for logs; won’t blow up on numpy/list types.
- :ref:`_select_transit_indices <api-nemesis-tess-pipeline-select-transit-indices>` — Return (selected\_indices, tier\_labels) for stratified SNR-ranked selection.
- :ref:`safe_cleanup_cache <api-nemesis-tess-pipeline-safe-cleanup-cache>` — Best-effort cleanup of files under the ONE cache root indicated by default\_settings. Mirrors your old behavior but without hardcoded /mnt/... paths. When ''shard\_cache\_by\_tic'' is enabled and ''tic\_id'' is provided…
- :ref:`safe_panel_normalization <api-nemesis-tess-pipeline-safe-panel-normalization>` — Normalize a per-panel flux/error pair by a guaranteed-positive scale.
- :ref:`safe_skyview_stamp_OLD <api-nemesis-tess-pipeline-safe-skyview-stamp-old>` — Wrapper around astrobase.plotbase.skyview\_stamp that ALWAYS returns (stamp, hdr) and NEVER raises. On any failure, returns (None, None).
- :ref:`SAP <api-nemesis-tess-pipeline-sap>` — Simple Aperture Photometry with simultaneous, flux-independent quality flagging.
- :ref:`SAP_ALT5 <api-nemesis-tess-pipeline-sap-alt5>` — Simple Aperture Photometry with fully coupled aperture evaluation.
- :ref:`SAP_and_BKG_uncertainty <api-nemesis-tess-pipeline-sap-and-bkg-uncertainty>` — Propagate photon and readout noise into SAP and background flux uncertainties.
- :ref:`saveNDarr <api-nemesis-tess-pipeline-savendarr>` — Serialize an arbitrary Python object to a pickle (.pkl) file.
- :ref:`scramble_hdu_orbits <api-nemesis-tess-pipeline-scramble-hdu-orbits>` — Swap the two TESS orbit image-frame blocks within a single-sector HDU, preserving the original time array.
- :ref:`search_sectors_from_filetype_and_path <api-nemesis-tess-pipeline-search-sectors-from-filetype-and-path>` — Discover which TESS sectors have processed light curves on disk.
- :ref:`search_sectors_from_filetype_and_path_OLD <api-nemesis-tess-pipeline-search-sectors-from-filetype-and-path-old>` — Discover which TESS sectors have processed light curves on disk.
- :ref:`secondary <api-nemesis-tess-pipeline-secondary>` — FP check: significant secondary eclipse detected.
- :ref:`single_event <api-nemesis-tess-pipeline-single-event>` — FA check: signal dominated by a single event.
- :ref:`singleoutliers <api-nemesis-tess-pipeline-singleoutliers>` — This function will search input data for points that are not consecutive.
- :ref:`sinusoidal <api-nemesis-tess-pipeline-sinusoidal>` — FA check: signal consistent with sinusoidal variability.
- :ref:`Smoother <api-nemesis-tess-pipeline-smoother>` — Smooth the PLD-corrected light curve with a biweight running window.
- :ref:`Smoother_OLD <api-nemesis-tess-pipeline-smoother-old>` — Smooth the PLD-corrected light curve with a biweight running window.
- :ref:`symmetry_window_is_degenerate <api-nemesis-tess-pipeline-symmetry-window-is-degenerate>` — Decide whether the left/right mirror symmetry test can run at all.

T
-

- :ref:`_tesspoint_note <api-nemesis-tess-pipeline-tesspoint-note>` — State tess-point's prediction for one star and sector, as one sentence.
- :ref:`T14 <api-nemesis-tess-pipeline-t14>` — Estimate the total transit duration (T14) for a planet using Kepler's third law and transit geometry.
- :ref:`T23 <api-nemesis-tess-pipeline-t23>` — Estimate the transit duration between the second and third contact points (T23).
- :ref:`TESS_Sector_observation_check <api-nemesis-tess-pipeline-tess-sector-observation-check>` — This function will return the TESS timestamps corresponding to the beginning of the first to the end of the second orbit per TESS sector of observations. Values are interpretted from the TESS Data Release Notes…
- :ref:`TesscutCounterSemaphore <api-nemesis-tess-pipeline-tesscutcountersemaphore>` — Cross-process semaphore using a single integer counter file.
- :ref:`TesscutCounterSemaphore.__enter__ <api-nemesis-tess-pipeline-tesscutcountersemaphore-enter>` — No description available in the source docstring.
- :ref:`TesscutCounterSemaphore.__exit__ <api-nemesis-tess-pipeline-tesscutcountersemaphore-exit>` — No description available in the source docstring.
- :ref:`TesscutCounterSemaphore.__init__ <api-nemesis-tess-pipeline-tesscutcountersemaphore-init>` — No description available in the source docstring.
- :ref:`TesscutCounterSemaphore._read_count <api-nemesis-tess-pipeline-tesscutcountersemaphore-read-count>` — No description available in the source docstring.
- :ref:`TesscutCounterSemaphore._write_count <api-nemesis-tess-pipeline-tesscutcountersemaphore-write-count>` — No description available in the source docstring.
- :ref:`tesssearch_with_radius_growth <api-nemesis-tess-pipeline-tesssearch-with-radius-growth>` — Perform lksearch.TESSSearch with an expanding search radius when the initial sector-constrained query raises a SearchError or yields no matching rows.
- :ref:`thresholdmask <api-nemesis-tess-pipeline-thresholdmask>` — Build a source aperture mask from pixels above a MAD-scaled threshold.
- :ref:`timer <api-nemesis-tess-pipeline-timer>` — Print the wall-clock runtime of a pipeline step in human-readable units.
- :ref:`TLS_func <api-nemesis-tess-pipeline-tls-func>` — This function will perform a transit search on the input light curve using the Transit Least Squares (TLS, Hippke & Heller et al. 2019, https://arxiv.org/pdf/1901.02015.pdf) algorithm. Run the Transit Least Squares…
- :ref:`transit_injection <api-nemesis-tess-pipeline-transit-injection>` — Inject a limb-darkened transit into a TESS-like light curve using BATMAN.
- :ref:`transit_injection_OLD <api-nemesis-tess-pipeline-transit-injection-old>` — Injects a limb-darkened transit into a TESS-like light curve using the BATMAN package.
- :ref:`Transit_Pipeline <api-nemesis-tess-pipeline-transit-pipeline>` **[public]** — Run BLS and optionally TLS transit searches, produce transit plots, and generate multi-page vetting reports.
- :ref:`Transit_plot <api-nemesis-tess-pipeline-transit-plot>` — Save a 2×2 transit search summary figure for the best-fit BLS or TLS signal.
- :ref:`TransitSearch <api-nemesis-tess-pipeline-transitsearch>` — Dispatcher that runs either BLS or TLS on the input light curve.
- :ref:`try_apertures <api-nemesis-tess-pipeline-try-apertures>` — Grid-search over aperture definitions and select the one minimising CDPP × (1 + contamination ratio).

U
-

- :ref:`unphysical_duration <api-nemesis-tess-pipeline-unphysical-duration>` — FA check: transit duration is unphysically short or long.

V
-

- :ref:`verify_cutout_identity <api-nemesis-tess-pipeline-verify-cutout-identity>` — Check that a cutout (TESScut or TPF) is the requested sector and the requested star.
- :ref:`verify_lc_matches_cutout <api-nemesis-tess-pipeline-verify-lc-matches-cutout>` — Check that a light curve's time stamps come from the cutout it was extracted from.
- :ref:`verify_lc_sector_by_time <api-nemesis-tess-pipeline-verify-lc-sector-by-time>` — Check that a light curve's first and last times fall in the requested sector. This is your ''TESS\_Sector\_observation\_check(T0=np.min(...))'' and ''(T0=np.max(...))'' test, run automatically.
- :ref:`verify_written_lc <api-nemesis-tess-pipeline-verify-written-lc>` — Re-read the Time column of a just-written light curve and compare it with the times held in memory. Catches a file on disk that is not the one this process wrote. A file that is not on disk is noted, not failed.
- :ref:`Vet_with_EDIVetter <api-nemesis-tess-pipeline-vet-with-edivetter>` — See source for parameters and behavior.
- :ref:`Vetting_Report <api-nemesis-tess-pipeline-vetting-report>` — Produce page 1 of the NEMESIS vetting report for a BLS or TLS transit signal.
- :ref:`Vetting_Report_2nd_page <api-nemesis-tess-pipeline-vetting-report-2nd-page>` — Produce page 2 of the NEMESIS vetting report: the full LC summary figure.
- :ref:`Vetting_Report_3rd_page <api-nemesis-tess-pipeline-vetting-report-3rd-page>` — Produce page 3 of the NEMESIS vetting report: LS periodogram and phase-folded light curves.
- :ref:`Vetting_Report_4th_page <api-nemesis-tess-pipeline-vetting-report-4th-page>` — Produce page 4 of the NEMESIS vetting report: centroid analysis and transit symmetry tests.
- :ref:`Vetting_Report_5th_page_plot_individual_transits <api-nemesis-tess-pipeline-vetting-report-5th-page-plot-individual-transits>` — Produce page 5 of the NEMESIS vetting report: individual transit zoom-ins.
- :ref:`Vetting_Report_5th_page_plot_individual_transits_OLD <api-nemesis-tess-pipeline-vetting-report-5th-page-plot-individual-transits-old>` — Produce page 5 of the NEMESIS vetting report: individual transit zoom-ins.
- :ref:`Vetting_Report_6th_page_LEOVetter <api-nemesis-tess-pipeline-vetting-report-6th-page-leovetter>` — Produce page 6 of the NEMESIS vetting report: LEO-Vetter modshift analysis.
- :ref:`Vetting_Report_multi_page <api-nemesis-tess-pipeline-vetting-report-multi-page>` — Produce pages 2–6 of the NEMESIS vetting report and merge all pages into a single multi-page PDF.
- :ref:`vshaped <api-nemesis-tess-pipeline-vshaped>` — FP check: V-shaped transit event (consistent with a grazing eclipsing binary).

W
-

- :ref:`_wait_for_paths <api-nemesis-tess-pipeline-wait-for-paths>` — Wait until each path in 'paths' exists and has stable size for 'stable\_reads' polls. Early-exits as soon as condition is met for any path (you'll open just one). Returns True if any path stabilized; False otherwise.
- :ref:`weak <api-nemesis-tess-pipeline-weak>` — FA check: signal too weak (MES below minimum threshold).
- :ref:`while_loop <api-nemesis-tess-pipeline-while-loop>` — Call 'command(\*args, \*\*kwargs)' with exponential back‑off. Stops after 'max\_retries' failures (None = infinite). Returns the function’s result or raises the last exception.
