#!/usr/bin/env python3
"""Everything that has to hold before a build. Run locally and in CI.

One script, two callers: what passes on a laptop is what gates the release.
A second copy of these checks inside a workflow file would drift from this one.

    python verify.py            checks the installed or source package
    python verify.py --build    also builds and validates the artifacts
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FAILURES: list[str] = []


def check(name: str):
    def deco(fn):
        try:
            fn()
        except Exception as exc:                        # noqa: BLE001
            FAILURES.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"  FAIL  {name}: {exc}")
        else:
            print(f"  ok    {name}")
        return fn
    return deco


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true",
                    help="also build sdist and wheel and run twine check")
    args = ap.parse_args()

    import decision_jef
    from decision_jef import Answer, Decider, Question, Request
    from decision_jef.model import temperature_bucket

    print(f"decision_jef {decision_jef.__version__} from "
          f"{Path(decision_jef.__file__).parent}")
    print("\npublic API")

    @check("version is a three-part string")
    def _():
        parts = decision_jef.__version__.split(".")
        assert len(parts) == 3 and all(p.isdigit() for p in parts), \
            decision_jef.__version__

    @check("the PyPI readme carries no YAML frontmatter")
    def _():
        readme = ROOT / "README.md"
        if not readme.exists():
            return
        first = readme.read_text().lstrip().splitlines()[0]
        assert first != "---", \
            "README.md starts with frontmatter; PyPI renders it as text"

    @check("pyproject leaves the version dynamic")
    def _():
        pp = ROOT / "pyproject.toml"
        if not pp.exists():
            return                                      # installed, no source
        # A plain text scan, not tomllib: that module is 3.11+ and this check
        # has to run on every Python the package claims to support.
        lines = [ln.strip() for ln in pp.read_text().splitlines()]
        assert 'dynamic = ["version"]' in lines, "pyproject must declare a dynamic version"
        assert not any(ln.startswith("version = \"") for ln in lines), \
            "pyproject pins a version; it must stay dynamic"

    @check("choice keys keep their given order")
    def _():
        q = Question("choice", "Which team?", {"b": "B", "a": "A"})
        assert q.keys == ["b", "a"], q.keys

    @check("score levels are indexed from zero")
    def _():
        s = Question("score", "How urgent?", ["low", "mid", "high"])
        assert s.keys == ["0", "1", "2"] and s.descriptions[2] == "high"

    @check("noul is keyed false then true")
    def _():
        n = Question("noul", "Urgent?", {"true": "yes", "false": "no"})
        assert n.keys == ["false", "true"] and n.descriptions == ["no", "yes"]

    @check("noul without criteria still answers")
    def _():
        assert Question("noul", "Urgent?").descriptions == ["no", "yes"]

    @check("the caps are enforced")
    def _():
        bad = [
            lambda: Question("choice", "x", {str(i): str(i) for i in range(256)}),
            lambda: Question("choice", "x", {"only": "one"}),
            lambda: Question("score", "x", [str(i) for i in range(11)]),
            lambda: Question("score", "x", ["only one"]),
            lambda: Question("noul", "x", {"maybe": "n", "true": "y"}),
            lambda: Question("nope", "x", None),
        ]
        for i, fn in enumerate(bad):
            try:
                fn()
            except ValueError:
                continue
            raise AssertionError(f"case {i} should have raised ValueError")

    @check("an option with no description falls back to its key")
    def _():
        # A real crash: a choice whose criteria values are all null reached the
        # tokenizer as None and raised "You need to specify either `text` or
        # `text_target`". The key is the label in that form.
        q = Question("choice", "What is it?", {"coding": None, "other": ""})
        assert q.descriptions == ["coding", "other"], q.descriptions
        s = Question("score", "How bad?", [None, "", "severe"])
        assert s.descriptions == ["level 0", "level 1", "severe"], s.descriptions
        n = Question("noul", "Urgent?", {"false": None, "true": ""})
        assert n.descriptions == ["no", "yes"], n.descriptions
        for d in q.descriptions + s.descriptions + n.descriptions:
            assert d != "None", "str(None) leaked through as an option label"

    @check("temperature buckets are stable")
    def _():
        assert temperature_bucket("noul", 2) == "noul"
        assert temperature_bucket("score", 5) == "score:5"
        assert temperature_bucket("choice", 3) == "choice:2-3"
        assert temperature_bucket("choice", 4) == "choice:4-6"
        assert temperature_bucket("choice", 20) == "choice:7+"

    @check("a request round-trips through JSON unchanged")
    def _():
        r = Request("state", {"q": Question("choice", "x?", {"a": "A", "b": "B"}),
                              "s": Question("score", "y?", ["lo", "hi"])})
        back = Request.from_json(r.to_json())
        assert back.to_json() == r.to_json()
        assert back.questions["s"].keys == ["0", "1"]

    @check("a noul answer expands to a distribution")
    def _():
        a = Answer.from_json({"type": "noul", "noul": 0.95})
        assert abs(a.p("true") - 0.95) < 1e-9
        assert abs(sum(a.probabilities.values()) - 1) < 1e-9

    @check("log odds clamp at the reporting floor")
    def _():
        a = Answer("choice", {"x": 0.0, "y": 1.0})
        assert a.log_odds("x", "y") < 0 and a.log_odds("x", "y") > -20

    @check("absent weights raise a message that says what to do")
    def _():
        try:
            Decider.from_pretrained(str(ROOT / "no-such-dir"))
        except FileNotFoundError as exc:
            assert "model.pt" in str(exc) and "hf auth login" in str(exc)
        else:
            raise AssertionError("expected FileNotFoundError")

    if args.build:
        print("\nbuild")

        @check("sdist and wheel build")
        def _():
            # `uv build` where available, so a uv-created virtualenv without
            # pip still verifies; CI installs `build` and takes the second.
            import shutil
            if shutil.which("uv"):
                cmd = ["uv", "build", "--out-dir", str(ROOT / "dist")]
            else:
                cmd = [sys.executable, "-m", "build"]
            subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)

        @check("twine accepts the artifacts")
        def _():
            dist = [p for p in sorted((ROOT / "dist").glob("*"))
                    if p.suffix in (".whl", ".gz")]
            assert dist, "nothing in dist/"
            import importlib.util
            if importlib.util.find_spec("twine") is None:
                raise AssertionError("twine is not installed")
            subprocess.run([sys.executable, "-m", "twine", "check",
                            *[str(p) for p in dist]],
                           check=True, capture_output=True)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
