"""Hash of the viewer's React sources, so tests can tell whether the committed build is stale.

`pnpm run build` (or npm) in src/specret/viewer/web writes this hash next to the build as
static/app/source-hash.txt; tests/test_phase3.py recomputes it. Line endings are normalized,
so a Windows and a Linux checkout hash the same.
"""

import hashlib
import sys
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "src/specret/viewer/web"
FILES = ["index.html", "package.json", "vite.config.ts", "tsconfig.json"]


def source_hash(web: Path = WEB) -> str:
    h = hashlib.sha256()
    paths = [web / f for f in FILES] + sorted(
        p for p in (web / "src").rglob("*") if p.is_file() and p.suffix != ".json"
    )
    for p in paths:
        h.update(p.relative_to(web).as_posix().encode())
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


if __name__ == "__main__":
    digest = source_hash()
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(digest + "\n", encoding="utf-8")
    print(digest)
