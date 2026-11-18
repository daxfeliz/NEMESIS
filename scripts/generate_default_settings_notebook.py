#!/usr/bin/env python3
"""Generate Notebook 00 directly from NEMESIS's active settings dictionary.

The generator reads `src/nemesis_tess/pipeline.py` with Python's AST, without
importing scientific dependencies or invoking NEMESIS. Every effective key is
documented; duplicate dictionary keys are detected and reported as warnings.

Usage
-----
python scripts/generate_default_settings_notebook.py
python scripts/generate_default_settings_notebook.py --check

Example
-------
>>> isinstance(document_default_settings, type(lambda: None))
True
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, OrderedDict
import json
import operator
from pathlib import Path
import sys
from typing import Any


# Every dictionary key must have a description. If the pipeline changes and a
# new key appears, update the glossary: no undocumented settings are published.
# Key groups follow the sections of the pipeline's settings dictionary.
GLOSSARY: dict[str, tuple[str, str]] = {
    # Paths and archive queries
    "base_path": ("Paths and archive access", "Optional root path for NEMESIS data products; None lets the caller or pipeline determine it."),
    "cache_dir_path": ("Paths and archive access", "Optional directory for downloaded/cached TESS resources."),
    "shard_cache_by_tic": ("Paths and archive access", "Organize cached data into TIC-specific subdirectories when enabled."),
    "mast_query_attempts": ("Paths and archive access", "Number of attempts permitted for an archive query."),
    # Sector control
    "get_all_sectors": ("Targets and execution", "Process all selected available TESS sectors rather than one requested sector."),
    "last_updated_sector": ("Targets and execution", "Upper sector limit used when discovering/iterating TESS observations."),
    "do_transit_search": ("Targets and execution", "Run the transit-search stage after light-curve processing."),
    "scramble_orbits": ("Targets and execution", "Enable orbit/sector scrambling for the survey false-alarm control; not intended for normal target analysis."),
    # Aperture extraction
    "cutoutsize": ("Aperture photometry", "Width/height of the requested square FFI pixel cutout."),
    "minimum_photon_counts": ("Aperture photometry", "Minimum acceptable median photon-count criterion for input image data."),
    "threshold": ("Aperture photometry", "Source-detection threshold for threshold-based aperture selection."),
    "use_SPOC_aperture": ("Aperture photometry", "Select an available SPOC aperture mask instead of constructing a custom one."),
    "use_centroid": ("Aperture photometry", "Use a centroid-based aperture approach where supported."),
    "use_circ": ("Aperture photometry", "Allow a circular aperture choice/fallback where supported."),
    "use_sources_in_aperture": ("Aperture photometry", "Include cataloged nearby sources in aperture/contamination handling."),
    "do_aperture_optimization": ("Aperture photometry", "Test candidate apertures and select an optimized aperture rather than relying only on the configured method."),
    "plot_catalog_source": ("Aperture photometry", "Catalog used to annotate nearby sources on diagnostic figures (e.g. Gaia or TIC); remote catalog access may be required."),
    "aperture_auto_select": ("Aperture photometry", "Enable automatic aperture-method choice using contamination and brightness criteria where supported."),
    "aperture_auto_contam_thresh": ("Aperture photometry", "Contamination-ratio threshold for automatic aperture-method switching."),
    "aperture_auto_tmag_thresh": ("Aperture photometry", "TESS magnitude threshold for automatic aperture-method switching."),
    "prf_decontaminate": ("Aperture photometry", "Enable PRF-based treatment of contaminating sources for supported PRF methods."),
    "prf_cache_dir": ("Aperture photometry", "Directory for PRF files (None lets the pipeline choose a location)."),
    "prf_jitter_sigma": ("Aperture photometry", "Optional PRF jitter broadening, in the units used by the PRF model."),
    "use_prf_pos_corr": ("Aperture photometry", "Use positional corrections in PRF modeling when supported."),
    "use_centroid_prf": ("Aperture photometry", "Permit PRF-derived centroiding when supported."),
    "use_prf_in_optimization": ("Aperture photometry", "Include PRF-assisted aperture choices in the optimizer."),
    "opt_thresholds": ("Aperture photometry", "Trial detection thresholds considered by aperture optimization."),
    "opt_radii": ("Aperture photometry", "Trial circular-aperture radii considered by aperture optimization."),
    "aperture_method": ("Aperture photometry", "Selected aperture method, such as threshold-based photometry."),
    "aperture_method_param": ("Aperture photometry", "Optional numerical/configuration parameter for the selected aperture method."),
    "circular_aperture_radius": ("Aperture photometry", "Radius in pixels used by the circular aperture method."),
    "neighbor_catalog_source": ("Aperture photometry", "Catalog used to identify nearby stars for aperture contamination handling."),
    "neighbor_excision_dtmag": ("Aperture photometry", "Magnitude-difference cut for considering nearby stars in pixel excision."),
    "neighbor_excision_radius": ("Aperture photometry", "Radius in pixels around a resolved neighbor to excise from the aperture."),
    "neighbor_blend_threshold_px": ("Aperture photometry", "Neighbor-separation threshold in pixels distinguishing blended versus resolved sources."),
    "neighbor_excision_min_pix": ("Aperture photometry", "Minimum number of target aperture pixels that must survive neighbor excision."),
    "neighbor_blend_shrink_radius": ("Aperture photometry", "Radius in pixels retained around the target when the field contains a blended neighbor."),
    "max_aperture_radius": ("Aperture photometry", "Maximum supported target-aperture extent in pixels in applicable selection methods."),
    "circular_fallback_radii": ("Aperture photometry", "Candidate fallback circular-aperture radii in pixels."),
    "use_prf_weights": ("Aperture photometry", "Weight aperture pixels using a PRF model, when selected and supported."),
    "prf_weight_threshold": ("Aperture photometry", "Minimum PRF pixel weight retained in PRF-weighted aperture construction."),
    "bitmask_string": ("Quality and jitter masking", "TESS QUALITY bitmask prescription used to reject flagged spacecraft cadences."),
    # Quality masking
    "do_mdump_removal": ("Quality and jitter masking", "Mask cadences associated with momentum dumps."),
    "before_after_in_minutes": ("Quality and jitter masking", "Extra exclusion window, in minutes, around a momentum dump."),
    "keep_mdumps": ("Quality and jitter masking", "Retain momentum-dump cadences rather than excluding them under the usual rule."),
    "keep_baddata": ("Quality and jitter masking", "Retain data normally removed by quality screening (use for diagnostics with care)."),
    "do_sector_jitter_mask": ("Quality and jitter masking", "Apply the sector-level jitter mask option when enabled."),
    "use_quat_jitter_mask": ("Quality and jitter masking", "Use spacecraft quaternion time series to detect and flag pointing jitter."),
    "quat_savepath": ("Quality and jitter masking", "Optional location for quaternion/engineering telemetry data."),
    "quat_threshold_sigma": ("Quality and jitter masking", "Robust-sigma threshold for declaring quaternion jitter anomalous."),
    "quat_min_samples": ("Quality and jitter masking", "Minimum quaternion samples in a cadence bin; None selects the pipeline's automatic rule."),
    "bkg_threshold_sigma": ("Quality and jitter masking", "Robust-sigma threshold for anomalous background flux."),
    "centroid_threshold_sigma": ("Quality and jitter masking", "Robust-sigma threshold for abnormal centroid displacement."),
    "bkg_detrend_window": ("Quality and jitter masking", "Window in days used to remove slow changes from background flux before flagging."),
    # PLD
    "pld_order": ("Pixel-level decorrelation", "Maximum order of pixel-flux products in the PLD systematics model."),
    "n_pca_terms": ("Pixel-level decorrelation", "Maximum retained singular/PCA components per higher-order PLD block."),
    "pld_gp_proxy_window": ("Pixel-level decorrelation", "Time window in days for the stellar-variability proxy in weighted PLD; None selects the simpler pass."),
    "pld_transit_mask_sigma": ("Pixel-level decorrelation", "Robust-sigma threshold for identifying downward flux deviations masked during iterative PLD fitting."),
    "pld_transit_mask_iter": ("Pixel-level decorrelation", "Maximum PLD transit-mask refinement iterations; zero disables this refinement."),
    "pld_transit_mask_grow": ("Pixel-level decorrelation", "Number of neighboring cadences on each side added to a PLD transit mask."),
    # Smoothing
    "window_size_in_days": ("Smoothing and outliers", "Time window in days for detrending; None requests the stellar/transit-duration-based automatic window."),
    "smoother_mode": ("Smoothing and outliers", "Detrending/smoothing mode chosen by the light-curve-processing stage."),
    "remove_outliers": ("Smoothing and outliers", "Enable the pipeline's outlier-rejection stage."),
    "Nsigma": ("Smoothing and outliers", "Outlier-removal deviation threshold used by the pipeline's clipping rule."),
    # File output
    "keep_FITS": ("Output retention", "Retain FITS intermediates and/or cutout files after processing."),
    "keep_raw_and_preclip": ("Output retention", "Retain additional preclipping and raw light-curve products."),
    "keep_imagedata": ("Output retention", "Retain image data products rather than deleting temporary image products."),
    "keep_extra_files": ("Output retention", "Keep additional diagnostic/intermediate files."),
    "keep_extra_transit_files": ("Output retention", "Keep additional transit-search intermediate files."),
    # Transit search
    "SDE_threshold": ("BLS/TLS search", "Minimum signal-detection-efficiency threshold for relevant search-gating paths; the default of zero imposes no positive SDE cut."),
    "do_TLS_Search": ("BLS/TLS search", "Enable Transit Least Squares in the transit search pipeline."),
    "N_transits": ("BLS/TLS search", "Minimum number of distinct transits used to set the search baseline/period bound."),
    "minP": ("BLS/TLS search", "Minimum trial orbital period in days."),
    "maxP": ("BLS/TLS search", "Maximum trial orbital period in days; None derives it from baseline and transit-count requirement."),
    "oversampling_factor": ("BLS/TLS search", "Frequency/period-grid oversampling factor."),
    "duration_grid_step": ("BLS/TLS search", "Multiplicative spacing between trial transit durations."),
    "N_TLS_threads": ("BLS/TLS search", "Optional thread count for TLS computation; None leaves the choice automatic."),
    "transit_search_quality_flags": ("BLS/TLS search", "Which NEMESIS quality-flag values are eligible for transit search; [0] means unflagged cadences only."),
    "vetting_report_quality_flags": ("BLS/TLS search", "Eligible quality-flag values for report panels; None inherits transit_search_quality_flags."),
    # Survey control
    "for_injections": ("Injection and scramble experiments", "Run photometry/search in injection mode rather than a normal target run."),
    "inject_param_P_RP_T0": ("Injection and scramble experiments", "Injection specification of orbital period, planet radius, and transit epoch."),
    # Reports
    "do_DSS_plot": ("Diagnostics and report switches", "Include the Digitized Sky Survey field image; disabling can avoid remote SkyView requests."),
    "do_vetting_report": ("Diagnostics and report switches", "Proposed control for vetting-report generation; may be inactive in some source versions—verify call sites before relying on it."),
    "do_transit_plot": ("Diagnostics and report switches", "Proposed control for transit plotting; may be inactive in some source versions—verify call sites."),
    "do_save_pdf": ("Diagnostics and report switches", "Proposed control for PDF merging; may be inactive in some source versions—verify call sites."),
    # Output
    "verbose": ("Display and verbosity", "Enable detailed status/debug messages."),
    "show_fig": ("Display and verbosity", "Display figures interactively, where supported."),
    "return_default_settings": ("Display and verbosity", "Legacy switch included in the settings dictionary; the function's return_dict parameter controls its own return type."),
}


OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Pow: operator.pow,
}


def _static_value(node: ast.AST) -> Any:
    """Evaluate only simple literal defaults and numeric arithmetic, never code."""
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        pass
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        left = _static_value(node.left)
        right = _static_value(node.right)
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            raise ValueError("Only numeric arithmetic is supported in defaults")
        return OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _static_value(node.operand)
        if not isinstance(value, (int, float)):
            raise ValueError("Only numeric unary operations are supported")
        return value if isinstance(node.op, ast.UAdd) else -value
    raise ValueError(f"Unexpected default expression: {ast.dump(node, include_attributes=False)}")


def extract_defaults(source: Path) -> tuple[OrderedDict[str, Any], dict[str, int]]:
    """Read effective defaults from a pipeline source file without importing it."""
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    funcs = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "default_NEMESIS_pipeline_settings"
    ]
    if len(funcs) != 1:
        raise ValueError("Expected exactly one default_NEMESIS_pipeline_settings definition")
    assignments = [
        node.value for node in ast.walk(funcs[0])
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "settings" for target in node.targets)
        and isinstance(node.value, ast.Dict)
    ]
    if len(assignments) != 1:
        raise ValueError("Expected exactly one literal settings dictionary")
    result: OrderedDict[str, Any] = OrderedDict()
    counts: Counter[str] = Counter()
    for key_node, value_node in zip(assignments[0].keys, assignments[0].values):
        if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
            raise ValueError("Nonliteral settings key encountered")
        key = key_node.value
        counts[key] += 1
        result[key] = _static_value(value_node)  # Last declaration wins, as in Python.
    return result, dict(counts)


def _markdown_cell(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def _code_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": source.splitlines(keepends=True),
    }


def document_default_settings(repo_root: Path) -> dict[str, Any]:
    """Build a notebook whose tables describe every active pipeline default.

    Args:
        repo_root: Git checkout root containing src/nemesis_tess/pipeline.py.

    Returns:
        Valid notebook-v4 JSON object with nbsphinx execution disabled.

    Raises:
        ValueError: If source structure changes or a key lacks a description.
    """
    source = repo_root / "src/nemesis_tess/pipeline.py"
    settings, counts = extract_defaults(source)
    entire_source = source.read_text(encoding="utf-8")
    # Ignore the definition itself when checking whether optional switches
    # occur at a use site; this is a warning heuristic, not reachability proof.
    source_tree = ast.parse(entire_source)
    default_function = next(node for node in source_tree.body if isinstance(node, ast.FunctionDef) and node.name == "default_NEMESIS_pipeline_settings")
    source_lines = entire_source.splitlines(keepends=True)
    outside_definition = "".join(source_lines[:default_function.lineno - 1] + source_lines[default_function.end_lineno:])
    possibly_inactive = [key for key in ("do_vetting_report", "do_transit_plot", "do_save_pdf")
                           if key in settings and (f"'{key}'" not in outside_definition and f'"{key}"' not in outside_definition)]
    missing = set(settings) - set(GLOSSARY)
    if missing:
        raise ValueError(
            "Missing descriptions for new pipeline settings: " + ", ".join(sorted(missing))
        )
    duplicate_keys = {key: n for key, n in counts.items() if n > 1}
    sections: OrderedDict[str, list[tuple[str, Any, str]]] = OrderedDict()
    for key, value in settings.items():
        section, description = GLOSSARY[key]
        sections.setdefault(section, []).append((key, value, description))

    cells: list[dict[str, Any]] = [
        _markdown_cell(
            "# 00 — NEMESIS default pipeline settings\n\n"
            "This tutorial explains the settings returned by "
            "`default_NEMESIS_pipeline_settings()`. Start here before running "
            "[Notebook 01](01_nemesis_leovetter.ipynb).\n\n"
            "**Source of truth:** The tables below are generated directly from the "
            "effective dictionary in `src/nemesis_tess/pipeline.py`. This document "
            "describes the **software defaults**, which are not necessarily the "
            "science configuration used for the NEMESIS II survey. "
            "Values are displayed without executing the pipeline.\n\n"
            "There are **" + str(len(settings)) + " effective settings** in the current source. "
            "You can execute the optional code cells in a Python environment with "
            "NEMESIS installed, but the Read the Docs build will not execute them.\n"
        ),
        _markdown_cell(
            "## 1. Retrieve the current defaults\n\n"
            "This function returns a Python dictionary. Access a value by its "
            "key, and copy the dictionary before changing options. The example "
            "imports the installed NEMESIS package; it does not download TESS data.\n"
        ),
        _code_cell(
            "from nemesis_tess import default_NEMESIS_pipeline_settings\n\n"
            "settings = default_NEMESIS_pipeline_settings()\n"
            "print(f'{len(settings)} settings loaded')\n"
            "print('Transit search enabled:', settings['do_transit_search'])\n"
            "print('Minimum period (days):', settings['minP'])\n"
            "print('Required transit count:', settings['N_transits'])\n"
        ),
        _markdown_cell(
            "## 2. Effective defaults, organized by processing stage\n\n"
            "`None` usually requests automatic behavior or an unset path; the exact "
            "meaning depends on the setting. The descriptions summarize the "
            "source implementation, not independent scientific validation. "
            "Features described as *optional* require their associated mode to "
            "be enabled.\n"
        ),
    ]

    for section, values in sections.items():
        rows = [f"### {section}\n", "| Setting | Default | What it controls |", "|:--|:--|:--|"]
        for key, value, description in values:
            rendered = repr(value).replace("|", "\\|")
            rows.append(f"| `{key}` | `{rendered}` | {description} |")
        cells.append(_markdown_cell("\n".join(rows) + "\n"))

    if duplicate_keys:
        inactive_note = (
            "The following switches are not referenced by name outside the "
            "settings definition in this source snapshot: "
            + ", ".join(f"`{key}`" for key in possibly_inactive)
            + ". Verify call sites before relying on them.\n"
            if possibly_inactive else ""
        )
        duplicate_text = ", ".join(
            f"`{key}` ({count} declarations; final value `{settings[key]!r}`)"
            for key, count in duplicate_keys.items()
        )
        cells.append(_markdown_cell(
            "## 3. Source-level cautions\n\n"
            f"The source dictionary currently declares the following keys more than once: {duplicate_text}. "
            "Python uses the last declaration. The tables above show those **effective** values.\n\n"
            "The function docstring may describe different historical defaults; "
            "for example, aperture optimization or catalog selection. "
            "**Trust the effective dictionary**, and check the actual implementation "
            "before relying on unusual or legacy controls.\n\n" + inactive_note
        ))
    else:
        cells.append(_markdown_cell(
            "## 3. Source-level cautions\n\n"
            "The table reflects the effective values in the current source. "
            "A default being present does not guarantee that every workflow "
            "consults it. Verify optional flags before relying on them.\n"
        ))

    cells.extend([
        _markdown_cell(
            "## 4. Override settings for a single run\n\n"
            "Changing a dictionary does not change the global defaults. "
            "Pass the modified settings to `full_pipeline(..., settings=settings)`. "
            "The example below enables verbose logs, retains FITS products, "
            "and chooses TIC rather than Gaia for plot annotations. This is "
            "useful where remote Gaia/VizieR queries are unavailable.\n"
        ),
        _code_cell(
            "settings = default_NEMESIS_pipeline_settings()\n"
            "settings.update({\n"
            "    'verbose': True,\n"
            "    'keep_FITS': True,\n"
            "    'plot_catalog_source': 'TIC',\n"
            "})\n"
            "for key in ('verbose', 'keep_FITS', 'plot_catalog_source'):\n"
            "    print(f'{key} = {settings[key]!r}')\n\n"
            "# Run only when you intend to download and process TESS data:\n"
            "# from nemesis_tess import full_pipeline\n"
            "# full_pipeline(ID=103633434, path='output/', cadence='30 minute',\n"
            "#               Sector=20, settings=settings)\n"
        ),
        _markdown_cell(
            "## 5. Compare your configuration with the defaults\n\n"
            "When debugging a run, print only values that differ from the defaults. "
            "This is safer than copying the whole settings dictionary into "
            "documentation and forgetting to update it.\n"
        ),
        _code_cell(
            "defaults = default_NEMESIS_pipeline_settings()\n"
            "changed = {k: (defaults[k], settings[k]) for k in defaults\n"
            "           if settings[k] != defaults[k]}\n"
            "for key, (before, after) in changed.items():\n"
            "    print(f'{key}: {before!r} -> {after!r}')\n"
            "assert set(settings) == set(defaults)\n"
        ),
        _markdown_cell(
            "## 6. Updating this reference\n\n"
            "After editing `default_NEMESIS_pipeline_settings()` in the repository, run:\n\n"
            "```bash\n"
            "python scripts/generate_default_settings_notebook.py\n"
            "python scripts/generate_default_settings_notebook.py --check\n"
            "```\n\n"
            "The generator uses the Python AST and does not import NEMESIS, "
            "download TESS observations, or run a transit search. "
            "If a new setting lacks a description, generation fails until "
            "the glossary in the generator is updated.\n\n"
            "See [Notebook 01](01_nemesis_leovetter.ipynb) for a real transit "
            "search and [the NEMESIS II preprint](https://arxiv.org/abs/2611.21758) "
            "for the survey configuration and methodology.\n"
        ),
    ])

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "nbsphinx": {"execute": "never"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    """Generate or verify the checked-in settings notebook."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the notebook is outdated")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = repo / "docs/notebooks/00_default_settings.ipynb"
    notebook = document_default_settings(repo)
    text = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
    if args.check:
        if not output.exists() or output.read_text(encoding="utf-8") != text:
            print("ERROR: Notebook 00 is missing/outdated; rerun the generator", file=sys.stderr)
            return 1
        print("PASS: Notebook 00 matches the NEMESIS settings dictionary")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    settings, dup = extract_defaults(repo / "src/nemesis_tess/pipeline.py")
    print(f"Wrote {output.relative_to(repo)} ({len(settings)} effective settings)")
    for key, count in dup.items():
        if count > 1:
            print(f"WARNING: '{key}' appears {count} times; final value is {settings[key]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
