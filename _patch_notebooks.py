"""Patch all notebooks to add A/B/C subplot labels to multi-panel figures."""
import json, re, pathlib

NB_DIR = pathlib.Path("notebooks")
MULTI_RE = re.compile(r'plt\.subplots\(\s*\d+\s*,\s*[2-9]\d*')


def patch_source(source_lines):
    """Return patched source or None if no change needed."""
    source = "".join(source_lines)

    if not MULTI_RE.search(source):
        return None
    if "label_subplots" in source:
        return None

    lines = source.split("\n")

    # Find plt.tight_layout indent
    tight_idx = None
    indent = ""
    for i, ln in enumerate(lines):
        stripped = ln.lstrip()
        if stripped.startswith("plt.tight_layout") or stripped.startswith("fig.tight_layout"):
            tight_idx = i
            indent = ln[: len(ln) - len(stripped)]
            break

    if tight_idx is None:
        return None

    # Detect axes variable name
    axes_var = "axes"
    for ln in lines:
        m = re.search(r'(?:fig\s*,\s*)(\w+)\s*=\s*plt\.subplots', ln)
        if m:
            axes_var = m.group(1)
            break

    # Add import at top
    lines.insert(0, "from src.utils import label_subplots")

    # Re-find tight_layout after insert
    tight_idx = None
    for i, ln in enumerate(lines):
        stripped = ln.lstrip()
        if stripped.startswith("plt.tight_layout") or stripped.startswith("fig.tight_layout"):
            tight_idx = i
            indent = ln[: len(ln) - len(stripped)]
            break

    if tight_idx is None:
        return None

    lines.insert(tight_idx, f"{indent}label_subplots({axes_var})")
    return "\n".join(lines)


notebooks = sorted(NB_DIR.glob("*.ipynb"))
total_patched = 0

for nb_path in notebooks:
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        new_src = patch_source(src)
        if new_src is not None:
            # Convert back to list of strings with \n terminators
            raw_lines = new_src.split("\n")
            new_lines = [ln + "\n" for ln in raw_lines[:-1]] + [raw_lines[-1]]
            cell["source"] = new_lines
            changed = True
            total_patched += 1
            for ln in new_lines:
                if "label_subplots" in ln:
                    print(f"  {nb_path.name}: {ln.strip()}")
                    break

    if changed:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"  -> Saved {nb_path.name}")

print(f"\nDone: {total_patched} cells patched")
