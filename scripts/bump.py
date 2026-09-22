#!/usr/bin/env python3
"""Bump the version, in the one place it lives.

    python scripts/bump.py patch     0.1.0 -> 0.1.1
    python scripts/bump.py minor     0.1.0 -> 0.2.0
    python scripts/bump.py major     0.1.0 -> 1.0.0
    python scripts/bump.py 0.3.0     set it outright

Then commit and tag: the publish workflow refuses a tag that disagrees with
the package version, and refuses a version already on PyPI.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

INIT = Path(__file__).resolve().parent.parent / "decision_jef" / "__init__.py"
PATTERN = re.compile(r'(__version__ = ")([^"]+)(")')


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    arg = sys.argv[1]
    text = INIT.read_text()
    match = PATTERN.search(text)
    if match is None:
        print(f"no __version__ in {INIT}", file=sys.stderr)
        return 1
    current = match.group(2)
    major, minor, patch = (int(x) for x in current.split("."))

    if arg == "major":
        new = f"{major + 1}.0.0"
    elif arg == "minor":
        new = f"{major}.{minor + 1}.0"
    elif arg == "patch":
        new = f"{major}.{minor}.{patch + 1}"
    elif re.fullmatch(r"\d+\.\d+\.\d+", arg):
        new = arg
    else:
        print(f"not a bump or a version: {arg!r}", file=sys.stderr)
        return 2

    INIT.write_text(PATTERN.sub(rf"\g<1>{new}\g<3>", text, count=1))
    print(f"{current} -> {new}")
    print("\nnext:")
    print(f"  python verify.py --build")
    print(f"  git commit -am 'Release {new}' && git tag v{new}")
    print(f"  git push && git push --tags")
    print(f"  gh release create v{new} --generate-notes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
