"""
Build step for the Vercel deployment (invoked via pyproject.toml's
[tool.vercel.scripts] build). Copies web/static/ into public/ — Vercel
serves anything in public/** straight from its CDN, without going through
the Python function at all, which is faster and keeps static assets out of
the Python bundle. web/static/ stays the single source of truth; this script
never needs to be run for local development (python -m web.server serves
web/static/ directly).
"""
from __future__ import annotations

import shutil
from pathlib import Path

SRC = Path(__file__).resolve().parent / "web" / "static"
DST = Path(__file__).resolve().parent / "public"


def main() -> None:
    if DST.exists():
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST)
    print(f"Copied {SRC} -> {DST}")


if __name__ == "__main__":
    main()
