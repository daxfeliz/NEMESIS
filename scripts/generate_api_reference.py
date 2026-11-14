#!/usr/bin/env python3
"""Generate NEMESIS Sphinx API pages from source AST without importing NEMESIS.

Usage:
    python scripts/generate_api_reference.py
    python scripts/generate_api_reference.py --check

The second command exits nonzero if the committed reference is stale. This
keeps Read the Docs independent of the scientific runtime dependencies.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from pathlib import Path
import re
from typing import NamedTuple


class ApiItem(NamedTuple):
    """One source-level function, class, or class method."""

    name: str
    module: str
    relative_file: str
    line: int
    kind: str
    signature: str
    summary: str
    anchor: str
    visibility: str


def generate_api_reference(repo_root: Path, check: bool = False) -> int:
    """Generate or check source-derived Sphinx reference pages.

    Args:
        repo_root: Repository root containing ``src/nemesis_tess`` and ``docs``.
        check: Compare expected outputs instead of modifying files.

    Returns:
        Number of top-level function and class names documented.

    Raises:
        FileNotFoundError: When the expected source package is absent.
        RuntimeError: In check mode when a generated page needs updating.

    Examples:
        >>> callable(generate_api_reference)
        True
        >>> isinstance(Path.cwd(), Path)
        True
    """
    source_dir = repo_root / "src" / "nemesis_tess"
    paths = [source_dir / "pipeline.py", source_dir / "compat.py"]
    modules = ["nemesis_tess.pipeline", "nemesis_tess.compat"]
    exports = {
        "full_pipeline", "LC_creation", "Transit_Pipeline",
        "default_NEMESIS_pipeline_settings", "NoImageDataError",
    }

    def slug(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")

    def synopsis(node: ast.AST) -> str:
        """Return a compact, reasonably readable docstring description."""
        doc = ast.get_docstring(node) or ""
        if not doc:
            return "No description available in the source docstring."
        lines = doc.splitlines()
        headings = {"input parameters", "parameters", "arguments", "args",
                    "notes", "returns", "description", "examples"}
        paragraphs: list[str] = []
        current: list[str] = []
        for line in lines:
            clean = line.strip()
            if not clean:
                if current:
                    paragraphs.append(" ".join(current))
                    current = []
                continue
            if clean.lower().rstrip(":") in headings or set(clean) <= {"-", "="}:
                if current:
                    paragraphs.append(" ".join(current))
                    current = []
                continue
            current.append(clean)
        if current:
            paragraphs.append(" ".join(current))
        paragraphs = [p for p in paragraphs if re.search(r"[A-Za-z]{3}", p)]
        if not paragraphs:
            return "See source for parameters and behavior."
        first = paragraphs[0]
        first = re.sub(r"\s+", " ", first).replace("`", "'")
        if len(first) > 220:
            first = first[:217].rsplit(" ", 1)[0].rstrip(".,;:") + "…"
        # The synopsis is prose, not an RST expression: preserve literal glob
        # patterns, underscores, and asterisks in source docstrings.
        first = first.replace("\\", "\\\\")
        first = first.replace("*", r"\*").replace("_", r"\_")
        return first

    def signature(node: ast.AST, name: str) -> str:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return f"class {name}"
        arg_string = ast.unparse(node.args)
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        result = f"{prefix} {name}({arg_string})"
        if node.returns is not None:
            result += f" -> {ast.unparse(node.returns)}"
        return result + ": ..."

    items: list[ApiItem] = []
    top_count = 0
    shadowed: list[str] = []
    for path, module in zip(paths, modules):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        # The final top-level definition is the one bound at module import.
        definitions: dict[str, ast.AST] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name in definitions:
                    shadowed.append(f"{module}.{node.name}")
                definitions[node.name] = node
        rel = path.relative_to(repo_root).as_posix()
        for name, node in definitions.items():
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            visibility = (
                "public" if name in exports else
                "private" if name.startswith("_") else
                "legacy" if name.endswith("_OLD") else
                "module"
            )
            anchor = f"api-{slug(module)}-{slug(name)}"
            items.append(ApiItem(
                name=name, module=module, relative_file=rel,
                line=node.lineno, kind=kind, signature=signature(node, name),
                summary=synopsis(node), anchor=anchor, visibility=visibility,
            ))
            top_count += 1
            if isinstance(node, ast.ClassDef):
                for method in node.body:
                    if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    method_name = f"{name}.{method.name}"
                    items.append(ApiItem(
                        name=method_name, module=module, relative_file=rel,
                        line=method.lineno, kind="method",
                        signature=signature(method, method_name),
                        summary=synopsis(method),
                        anchor=f"api-{slug(module)}-{slug(method_name)}",
                        visibility="private" if method.name.startswith("_") or name.startswith("_") else "module",
                    ))

    def source_ref(item: ApiItem) -> str:
        url = (
            "https://github.com/daxfeliz/NEMESIS/blob/main/"
            f"{item.relative_file}#L{item.line}"
        )
        return f"`View source <{url}>`__"

    main: list[str] = [
        "API reference", "=============", "",
        "This reference is generated from NEMESIS source syntax and docstrings,",
        "**without importing or running the scientific pipeline**. Definitions,",
        "signatures, and source locations reflect the committed code at generation",
        "time. Details and behavior are described in the linked source docstrings.",
        "",
        "The **supported top-level exports** are ``full_pipeline``,",
        "``LC_creation``, ``Transit_Pipeline``,",
        "``default_NEMESIS_pipeline_settings``, and ``NoImageDataError``.",
        "The complete inventory below also includes internal helpers and",
        "legacy-named functions, which are **not promised as a stable public API**.",
        "The NEMESIS II jaxoplanet and TRICERATOPS reference wrappers live in",
        "``examples/reference_wrappers/`` and are not installed package exports.",
        "",
        "To update both reference pages after source edits, run",
        "``python scripts/generate_api_reference.py`` at the repository root.",
        "",
        f"This snapshot contains **{top_count} unique top-level functions/classes**",
        f"and **{len(items) - top_count} class methods**.",
        "", 
    ]
    if shadowed:
        main.extend([
            ".. note::", "",
            "   An earlier top-level definition is replaced by a later definition",
            "   of the same name at import time: " + ", ".join(f"``{n}``" for n in shadowed) + ".",
            "   The reference lists the final active definition.",
            "",
        ])

    categories = [
        ("Supported top-level exports", [i for i in items if i.visibility == "public"]),
        ("Other module-level functions and classes", [i for i in items if i.visibility == "module" and i.kind != "method"]),
        ("Private and internal helpers", [i for i in items if i.visibility == "private" and i.kind != "method"]),
        ("Legacy-named definitions", [i for i in items if i.visibility == "legacy"]),
        ("Class methods", [i for i in items if i.kind == "method"]),
    ]
    for label, rows in categories:
        main.extend([label, "-" * len(label), ""])
        for item in sorted(rows, key=lambda row: (row.name.casefold(), row.module)):
            main.extend([
                f".. _{item.anchor}:", "",
                item.name, "~" * len(item.name), "",
                f"**Location:** ``{item.module}`` — {source_ref(item)}", "",
                f"**Description:** {item.summary}", "",
                ".. code-block:: python", "",
            ])
            # Python's AST unparser produces a compact line; code blocks scroll
            # horizontally when a scientific signature has many arguments.
            for line in item.signature.splitlines():
                main.append("   " + line)
            main.append("")

    index: list[str] = [
        "Alphabetical function and class index",
        "=====================================", "",
        "An alphabetical inventory of the function names, classes, and class",
        "methods defined in the installed NEMESIS source. Names link to the",
        ":doc:`API reference <api_reference>`, which shows signatures and",
        "GitHub source locations. This is an **inventory, not a guarantee of",
        "public API stability**; leading-underscore and legacy-named entries",
        "are generally internal. The separate example wrappers are excluded.",
        "",
    ]
    grouped: dict[str, list[ApiItem]] = defaultdict(list)
    for item in items:
        initial = item.name.lstrip("_")[:1].upper() or "#"
        grouped[initial].append(item)
    for initial in sorted(grouped):
        index.extend([initial, "-" * len(initial), ""])
        for item in sorted(grouped[initial], key=lambda row: (row.name.casefold(), row.module)):
            mark = " **[public]**" if item.visibility == "public" else ""
            summary = item.summary.replace("|", "\\|")
            index.append(f"- :ref:`{item.name} <{item.anchor}>`{mark} — {summary}")
        index.append("")

    target_files = {
        repo_root / "docs" / "api_reference.rst": "\n".join(main).rstrip() + "\n",
        repo_root / "docs" / "function_index.rst": "\n".join(index).rstrip() + "\n",
    }
    outdated = []
    for target, desired in target_files.items():
        if check:
            if not target.is_file() or target.read_text(encoding="utf-8") != desired:
                outdated.append(target.relative_to(repo_root))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(desired, encoding="utf-8")
            print(f"Generated {target.relative_to(repo_root)}")
    if outdated:
        names = ", ".join(str(p) for p in outdated)
        raise RuntimeError(
            f"API pages are stale: {names}. "
            "Run python scripts/generate_api_reference.py"
        )
    if check:
        print("PASS: API reference and alphabetical index are up to date")
    print(f"Documented {top_count} unique top-level names and "
          f"{len(items)-top_count} class methods")
    return top_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Fail when committed API pages are stale")
    arguments = parser.parse_args()
    generate_api_reference(Path(__file__).resolve().parents[1], check=arguments.check)
