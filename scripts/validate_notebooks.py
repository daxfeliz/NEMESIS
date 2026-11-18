"""Validate the JSON, Python syntax, and non-execution metadata of tutorials."""

from __future__ import annotations

import ast
import json
from pathlib import Path


def validate_docs_examples(repo_root: Path) -> int:
    """Validate tutorial notebooks without executing scientific workflows.

    Saved execution results are permitted so Sphinx can display real plots
    and tables. Cells containing a saved exception traceback are rejected.

    Args:
        repo_root: Root directory of the NEMESIS Git checkout.

    Returns:
        Number of notebooks successfully validated.

    Raises:
        AssertionError: If a notebook has invalid metadata or saved errors.
        SyntaxError: If a code cell contains invalid Python syntax.

    Example:
        >>> isinstance(validate_docs_examples(Path.cwd()), int)  # doctest: +SKIP
        True
    """
    folder = repo_root / "docs" / "notebooks"
    notebooks = sorted(folder.glob("*.ipynb"))
    assert len(notebooks) == 6, f"Expected 6 notebooks in {folder}"

    for path in notebooks:
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["nbformat"] == 4, path
        assert document["metadata"]["nbsphinx"]["execute"] == "never", path

        for index, cell in enumerate(document["cells"]):
            if cell["cell_type"] != "code":
                continue

            # Saved outputs are allowed; Python source must remain valid.
            ast.parse("".join(cell["source"]), filename=f"{path}:{index}")

            # Reject only actual Jupyter error outputs (tracebacks).
            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    raise AssertionError(
                        f"{path}: code cell {index} contains a saved "
                        f"{output.get('ename', 'error')}: "
                        f"{output.get('evalue', '')}"
                    )

        print(f"PASS {path.relative_to(repo_root)}")

    return len(notebooks)


if __name__ == "__main__":
    n_valid = validate_docs_examples(Path(__file__).resolve().parents[1])
    print(f"PASS: {n_valid} notebooks validated")
