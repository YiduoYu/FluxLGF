"""Write a sha256 manifest covering everything an experiment driver can reach, so a lock also catches changes in
bt_elim.py, ir_fp128.py, gen3t.py and the like (audit 2026-11-05, item 9). Follows local imports, including those
inside functions; add the protocol and driver scripts by name. Warns if a listed file has uncommitted changes.
usage (from regions/): python lock_manifest.py seq_isolated.py protocol_x.md test_x.sh > protocol_x.sha256
check: shasum -a 256 -c protocol_x.sha256"""
import ast
import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEARCH = (ROOT / "regions", ROOT / "solver_challenge", ROOT / "multifield")


def reachable(script):
    """The script and all project modules it imports, directly or not."""
    seen, todo = set(), [Path(script).resolve()]
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf8"))):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module]
            else:
                continue
            for name in names:
                for d in (path.parent, *SEARCH):
                    if (d / f"{name}.py").exists():
                        todo.append((d / f"{name}.py").resolve())
                        break
    return seen


if __name__ == "__main__":
    files = set()
    for arg in sys.argv[1:]:
        files |= reachable(arg) if arg.endswith(".py") else {Path(arg).resolve()}
    for f in sorted(files):
        print(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {os.path.relpath(f)}")
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *map(str, files)], capture_output=True, text=True).stdout
    if dirty:
        print("warning: uncommitted changes, the manifest will not match any commit:\n" + dirty, file=sys.stderr)
