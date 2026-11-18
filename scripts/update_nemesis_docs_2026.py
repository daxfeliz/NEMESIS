#!/usr/bin/env python3
"""Apply focused NEMESIS II publication and Notebook 00 documentation edits.

Unlike replacing README.md with an old snapshot, this script edits specific
outdated phrases in the *current local files*. It is idempotent: it will not
repeat changes already present. It writes nothing unless all expected source
patterns are found (or changes are already applied).

Usage:
    python scripts/update_nemesis_docs_2026.py --dry-run
    python scripts/update_nemesis_docs_2026.py
"""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path
import sys

ARXIV = "https://arxiv.org/abs/2611.21758"


def update_docs(repo_root: Path, dry_run: bool = False) -> None:
    """Apply local, idempotent text replacements to NEMESIS documentation.

    Args:
        repo_root: Root of the local NEMESIS Git checkout.
        dry_run: If true, print a diff without writing files.

    Raises:
        ValueError: If a file has unrecognized text that needs manual review.
    """
    updates: dict[Path, str] = {}

    def read(relpath: str) -> str:
        return (repo_root / relpath).read_text(encoding="utf-8")

    def replace_once(
        text: str, old: str, new: str, source: str,
        already: str | None = None,
    ) -> str:
        if old in text:
            if text.count(old) != 1:
                raise ValueError(f"{source}: expected one old phrase, found {text.count(old)}")
            return text.replace(old, new)
        if (already or new) in text:
            return text
        raise ValueError(f"{source}: expected phrase not found; edit manually instead")

    # README remains the canonical GitHub landing page.
    path = "README.md"
    old = read(path)
    revised = old
    revised = replace_once(
        revised,
        "**NEMESIS II (Feliz et al., in preparation)**",
        f"**NEMESIS II (Feliz et al. 2026; [arXiv:2611.21758]({ARXIV}))**",
        path,
    )
    revised = replace_once(
        revised,
        "The manuscript is **not yet posted or peer-reviewed**; posting to arXiv and submission to *The Astronomical Journal* are planned.",
        "The manuscript is available as an **arXiv preprint** and has been "
        "**submitted to *The Astronomical Journal* for peer review**; it is "
        "not yet a refereed journal publication.",
        path,
        already="The manuscript is available as an **arXiv preprint**",
    )
    revised = replace_once(
        revised,
        "- **NEMESIS II:** Feliz et al. (in preparation); planned arXiv posting and submission to *The Astronomical Journal* (link to follow)",
        f"- **NEMESIS II:** [Feliz et al. (2026), arXiv:2611.21758]({ARXIV}) — preprint; submitted to *The Astronomical Journal*",
        path,
    )
    revised = replace_once(
        revised,
        "and, once available, **Feliz et al. (2026, NEMESIS II; forthcoming)**. Until NEMESIS II is public, describe it as *in preparation* rather than citing a nonexistent preprint or publication.",
        f"and **[Feliz et al. (2026), NEMESIS II, arXiv:2611.21758]({ARXIV})**. "
        "NEMESIS II is publicly available as a preprint and has been submitted "
        "to *The Astronomical Journal*; do not cite it as an accepted journal article yet.",
        path,
    )
    if "00_default_settings.html" not in revised:
        anchor = "- [Getting Started / Installation](https://nemesis-tess.readthedocs.io/en/latest/installation.html)"
        if anchor not in revised:
            raise ValueError(f"{path}: documentation link anchor missing")
        revised = revised.replace(
            anchor,
            anchor + "\n" +
            "- [**00 — Default pipeline settings**](https://nemesis-tess.readthedocs.io/en/latest/notebooks/00_default_settings.html)",
        )
    updates[repo_root / path] = revised

    # Citations: preprint available, journal not yet refereed.
    path = "docs/citing.md"
    old = read(path)
    revised = old
    old_para = (
        "NEMESIS II (Feliz et al., **in preparation**) describes the extended TESS Primary Mission M-dwarf transit survey and its updated automatic vetting and selection-function analysis. Its authors plan to post a preprint on arXiv and submit to *The Astronomical Journal*. **No public NEMESIS II citation or DOI is available yet**; update your references when the manuscript becomes public."
    )
    new_para = (
        f"**NEMESIS II:** [Feliz et al. (2026), *NEMESIS: Exoplanet TraNsit SurvEy of Nearby M-dwarfs in TESS FFIs II — 262 Transiting Planet Candidates from the TESS Primary Mission*, arXiv:2611.21758]({ARXIV}). "
        "This **preprint has been submitted to *The Astronomical Journal* for peer review**; it is not yet a refereed AJ article. "
        "The paper describes the Primary Mission survey, updated photometry/vetting, injection–recovery sensitivity, and orbit-scrambling false-alarm analysis."
    )
    revised = replace_once(revised, old_para, new_para, path, already="arXiv:2611.21758")
    revised = replace_once(
        revised,
        "For survey-specific NEMESIS II methods and candidate results, reference the NEMESIS II paper after its public release rather than presenting draft results as refereed findings.",
        f"For survey-specific NEMESIS II methods or candidate results, cite the [2026 arXiv preprint]({ARXIV}) and update the reference when a journal version is available.",
        path,
    )
    updates[repo_root / path] = revised

    # Add an overview link and the first notebook in the Sphinx toctree.
    path = "docs/index.rst"
    old = read(path)
    revised = old
    toc_line = "   notebooks/01_nemesis_leovetter"
    if "   notebooks/00_default_settings" not in revised:
        if toc_line not in revised:
            raise ValueError(f"{path}: notebook list anchor missing")
        revised = revised.replace(toc_line, "   notebooks/00_default_settings\n" + toc_line, 1)
    if "arxiv.org/abs/2611.21758" not in revised:
        marker = ".. toctree::"
        if marker not in revised:
            raise ValueError(f"{path}: toctree marker missing")
        publication_note = (
            "**NEMESIS II:** `Feliz et al. (2026), arXiv:2611.21758 "
            f"<{ARXIV}>`_ is available as a **preprint** and "
            "has been submitted to *The Astronomical Journal* for peer review.\n\n"
        )
        revised = revised.replace(marker, publication_note + marker, 1)
    updates[repo_root / path] = revised

    # Avoid stale GitHub "Cite this repository" instructions.
    path = "CITATION.cff"
    old = read(path)
    revised = replace_once(
        old,
        "Also cite the NEMESIS II paper once it\n  is publicly available; at present it is in preparation.",
        "For NEMESIS II results and updated methods, cite\n  Feliz et al. (2026), arXiv:2611.21758 (submitted to AJ).",
        path,
        already="arXiv:2611.21758",
    )
    updates[repo_root / path] = revised

    # The older validator assumes exactly five notebook files.
    path = "scripts/validate_notebooks.py"
    old = read(path)
    revised = replace_once(
        old,
        'assert len(notebooks) == 5, f"Expected 5 notebooks in {folder}"',
        'assert len(notebooks) == 6, f"Expected 6 notebooks in {folder}"',
        path,
    )
    updates[repo_root / path] = revised

    for path, new_text in updates.items():
        previous = path.read_text(encoding="utf-8")
        if previous == new_text:
            print(f"UNCHANGED {path.relative_to(repo_root)}")
            continue
        print(f"{'WOULD UPDATE' if dry_run else 'UPDATED'} {path.relative_to(repo_root)}")
        if dry_run:
            diff = difflib.unified_diff(
                previous.splitlines(keepends=True), new_text.splitlines(keepends=True),
                fromfile=f"a/{path.relative_to(repo_root)}",
                tofile=f"b/{path.relative_to(repo_root)}",
            )
            print("".join(diff), end="")
        else:
            path.write_text(new_text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        update_docs(Path(__file__).resolve().parents[1], dry_run=args.dry_run)
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
