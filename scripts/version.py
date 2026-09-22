#!/usr/bin/env python3
"""Print the package version without importing the package.

The publish workflow needs it before anything is installed, and reading the
literal avoids importing torch just to learn a version string.
"""
import pathlib
import re
import sys

INIT = pathlib.Path(__file__).resolve().parent.parent / "decision_jef" / "__init__.py"
match = re.search(r'__version__ = "([^"]+)"', INIT.read_text())
if match is None:
    print(f"no __version__ in {INIT}", file=sys.stderr)
    raise SystemExit(1)
print(match.group(1))
