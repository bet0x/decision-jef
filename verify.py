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


_TOK = []


def _tok():
    """The packer needs a real tokenizer. Cache it; skip if it is unreachable.

    verify.py runs in CI without network on some jobs, so a missing tokenizer
    has to skip the check rather than fail the suite. It returns None only
    when the download is impossible, never when the packing is wrong.
    """
    if not _TOK:
        try:
            from transformers import AutoTokenizer
            _TOK.append(AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base"))
        except Exception as e:                        # noqa: BLE001
            print(f"    (sin tokenizador: {type(e).__name__})")
            _TOK.append(None)
    return _TOK[0]


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

    @check("a live client's structured instructions and state become text")
    def _():
        # A browser agent sends instructions as an object, with its standing
        # policy and the tick's constraints alongside the task, and a
        # structured state as nested objects. Passing either to the model as a
        # Python repr would feed it punctuation.
        from decision_jef.wire import as_instructions, as_state_text
        got = as_instructions({"task": "Choose navigation for this tick.",
                               "policy": "Play Doom autonomously.", "tick": 41})
        assert got.startswith("task: Choose navigation"), got
        assert ".." not in got, got            # a value already ending in "."
        assert "{" not in got and "'" not in got, got
        assert as_instructions("plain") == "plain"
        assert as_instructions(None) == ""
        st = as_state_text({"health": 88, "ammo": {"shells": 12}})
        assert st == '{"health": 88, "ammo": {"shells": 12}}', st
        assert as_state_text("already text") == "already text"

    @check("a one-option choice is answered rather than refused")
    def _():
        # A live client builds its option list from the world: a game agent
        # asks which visible enemy to aim at and usually has one or none. The
        # wire format still requires two -- a choice among one is not a choice
        # -- so the server answers these itself instead of returning 422.
        from decision_jef.serve import _forced_choices, _wire_forced
        payload = {"state": "s", "questions": {
            "target": {"type": "choice", "criteria": {"Demon_1": "aim at it"},
                       "instructions": "Which enemy?"},
            "movement": {"type": "choice", "instructions": "Where?",
                         "criteria": {"a": "A", "b": "B"}}}}
        forced = _forced_choices(payload)
        assert forced == {"target": "Demon_1"}, forced
        assert list(payload["questions"]) == ["movement"], payload["questions"]
        wired = _wire_forced(forced)["answers"]["target"]
        assert wired["choice"] == "Demon_1" and wired["confidence"] == 1.0
        assert wired["probabilities"] == {"Demon_1": 1.0} and wired["forced"]
        # A two-option choice is left for the model.
        assert _forced_choices({"questions": {"m": {
            "type": "choice", "criteria": {"a": "A", "b": "B"}}}}) == {}

    @check("a question's packing does not depend on what precedes it")
    def _():
        # This shipped broken. Attention isolation was exact, but positions ran
        # straight through the sequence, so RoPE moved a question whenever
        # another was placed in front of it, and the sliding layers' band was
        # indexed absolutely, so a long preceding question cost the next one
        # sight of the shared state. One noul question answered 0.215 alone and
        # 0.104 behind an 18-token question. Nothing in this file caught it.
        import torch
        from decision_jef.pack import pack, segment_attention_mask
        tok = _tok()
        if tok is None:
            return                      # no tokenizer available offline
        u = Question("noul", "Does this need a reply today?")
        a = Question("choice", "Which queue?", {"x": "one two three",
                                                "y": "four five six"})
        big = Question("choice", "Which queue?", {"x": "alpha beta " * 20,
                                                  "y": "gamma delta " * 20})
        alone = pack([Request("s", {"u": u})], tok, 1024)
        want = alone.position_ids[0, alone.segment_ids[0] == 1].tolist()
        for name, qs in (("short first", {"a": a, "u": u}),
                         ("long first", {"a": big, "u": u}),
                         ("two first", {"a": a, "b": big, "u": u})):
            got = pack([Request("s", qs)], tok, 1024)
            seg = list(qs).index("u") + 1
            have = got.position_ids[0, got.segment_ids[0] == seg].tolist()
            assert have == want, f"{name}: {have[:6]} != {want[:6]}"
        # And the mask must still hide the other questions outright.
        m = segment_attention_mask(got.segment_ids, got.attention_mask)[0, 0]
        seg = got.segment_ids[0]
        mine, other = seg == 3, seg == 1
        assert not bool(m[mine][:, other].any()), "a question can see another"

    @check("a question passed as a plain dict says what to do")
    def _():
        # Without the check in Request.__post_init__ this reached the packer,
        # where `q.keys` resolves to dict.keys, and the user saw "object of
        # type 'builtin_function_or_method' has no len()".
        try:
            Request("state", {"queue": {"type": "choice", "options": {"a": "A"}}})
        except TypeError as e:
            assert "plain dict" in str(e) and "Question(" in str(e), str(e)
        else:
            raise AssertionError("a plain dict should have raised TypeError")
        try:
            Request("state", {})
        except ValueError:
            pass
        else:
            raise AssertionError("an empty request should have raised ValueError")

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

    @check("every optional head is inferred from the weights")
    def _():
        # A checkpoint written before a head existed must still load. The flags
        # are read off the state_dict, so this list has to stay in step with
        # the heads the model defines.
        import inspect

        from decision_jef.infer import Decider as _D
        from decision_jef.model import DecisionJef

        params = inspect.signature(DecisionJef.__init__).parameters
        heads = {n for n in params if n.endswith("_head")}
        src = inspect.getsource(_D.from_pretrained)
        for h in heads:
            assert f'"{h}"' in src, f"{h} is not inferred in from_pretrained"

    @check("email cleaning cuts the thread, the signature and the disclaimer")
    def _():
        from decision_jef import email
        raw = ("Please refund the duplicate.\n\n"
               "On Tue, 3 Jun 2025 at 14:02, S <a@b.com> wrote:\n"
               "> earlier text\n\n--\nDana\n\n"
               "This e-mail is confidential and intended solely for you.")
        assert email.clean(raw) == "Please refund the duplicate.", email.clean(raw)
        st = email.as_state("a@b.com", "Charge", raw)
        assert st.startswith("from: a@b.com") and "confidential" not in st

    @check("shortlisting keeps a lexically obvious label and passes small sets through")
    def _():
        from decision_jef.shortlist import rank_options
        labels = {f"i{n}": f"outcome number {n}" for n in range(40)}
        labels["billing_refund"] = "the customer wants a duplicate charge refunded"
        q = Question("choice", "What do they want?", labels)
        kept, scores = rank_options(None, "refund the duplicate charge", q, 5)
        assert kept[0] == "billing_refund", kept[:3]
        assert scores is not None and len(kept) == 5
        small = Question("choice", "x?", {"a": "A", "b": "B"})
        kept2, scores2 = rank_options(None, "anything", small, 5)
        assert kept2 == ["a", "b"] and scores2 is None

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
            msg = str(exc)
            # The message has to name the file the loader actually looks for.
            # It said model.pt for a version after the format changed, and this
            # check passed the whole time because it asserted the old name.
            assert "model.safetensors" in msg, msg
            assert "hf auth login" in msg, msg
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
