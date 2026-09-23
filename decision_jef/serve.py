"""A Jev-compatible HTTP endpoint for this model.

    decision-jef-serve --weights BarraHome/Decision-Jef-0.1 --port 8088

Existing Jev clients post to `/v1/systemone` with

    {"model": ..., "state": <text or object>,
     "questions": {qid: {"type": ..., "instructions": ..., "criteria": ...}}}

and read back `{"answers": {qid: {...}}}`. That is exactly what
`Request.from_json` accepts and `Decider.to_wire` produces, so this module is
a socket and a lock rather than a translation layer.

Standard library only. A game loop querying this at 10 Hz should not have to
install a web framework, and the work is one forward pass behind a lock.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from decision_jef.infer import Decider
from decision_jef.wire import Request

MAX_BODY = 4 * 1024 * 1024


def _forced_choices(payload: dict) -> dict:
    """Pull out and answer any choice question that offers a single option.

    A live client builds its option list from the world: a Doom agent asks
    which visible enemy to aim at, and most of the time there are one or none.
    The wire format requires two options for a `choice` and should keep doing
    so -- a choice among one alternative is not a choice -- but rejecting the
    request is the wrong response when the answer is forced and correct. These
    are answered here, removed from what reaches the model, and merged back
    into the response with probability 1.

    Mutates `payload["questions"]`, which is this request's own parsed body.
    """
    out = {}
    qs = payload.get("questions")
    if not isinstance(qs, dict):
        return out
    for qid in list(qs):
        q = qs[qid]
        if not isinstance(q, dict) or q.get("type") != "choice":
            continue
        crit = q.get("criteria")
        if isinstance(crit, dict) and len(crit) == 1:
            out[qid] = next(iter(crit))
            del qs[qid]
    return out


def _wire_forced(forced: dict) -> dict:
    return {"model": "Decision-Jef-0.1",
            "answers": {qid: {"type": "choice", "choice": key,
                              "probabilities": {key: 1.0}, "confidence": 1.0,
                              "forced": True}
                        for qid, key in forced.items()}}


class _Handler(BaseHTTPRequestHandler):
    server_version = "decision-jef"
    decider: Decider = None            # set by serve()
    lock: threading.Lock = None
    token: Optional[str] = None
    calibrated: bool = False
    quiet: bool = False

    def log_message(self, fmt, *args):     # noqa: A003
        if not self.quiet:
            sys.stderr.write("  %s\n" % (fmt % args))

    def _send(self, code: int, body: dict):
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):                      # noqa: N802
        if self.path.rstrip("/") in ("/health", "/healthz"):
            self._send(200, {"status": "ok", "model": "Decision-Jef-0.1"})
        else:
            self._send(404, {"error": f"no route for GET {self.path}"})

    def do_POST(self):                     # noqa: N802
        if self.path.rstrip("/") != "/v1/systemone":
            self._send(404, {"error": f"no route for POST {self.path}"})
            return
        if self.token is not None:
            sent = (self.headers.get("authorization") or "")
            if sent.removeprefix("Bearer ").strip() != self.token:
                self._send(401, {"error": "bad or missing bearer token"})
                return
        try:
            n = int(self.headers.get("content-length") or 0)
        except ValueError:
            n = 0
        if n <= 0 or n > MAX_BODY:
            self._send(413, {"error": f"content-length {n} out of range"})
            return
        try:
            payload = json.loads(self.rfile.read(n))
        except Exception as e:             # noqa: BLE001
            self._send(400, {"error": f"body is not JSON: {e}"})
            return
        forced = _forced_choices(payload)
        if forced and not payload.get("questions"):
            # Every question was a forced one-option choice. Building a Request
            # from what is left would fail its own empty-request check, and
            # there is nothing for the model to do.
            self._send(200, _wire_forced(forced))
            return
        try:
            req = Request.from_json(payload)
        except Exception as e:             # noqa: BLE001
            # The client's own error path only sees a status code, so say what
            # was wrong in the body: a malformed question is the common case.
            self._send(422, {"error": f"{type(e).__name__}: {e}"})
            return
        t0 = time.perf_counter()
        try:
            with self.lock:
                answers = self.decider.decide(
                    req.state, req.questions, calibrated=self.calibrated)
        except Exception as e:             # noqa: BLE001
            self._send(500, {"error": f"{type(e).__name__}: {e}"})
            return
        ms = (time.perf_counter() - t0) * 1000
        body = self.decider.to_wire(answers)
        body["answers"].update(_wire_forced(forced)["answers"])
        body["latency_ms"] = round(ms, 2)
        if not self.quiet:
            got = " ".join(
                f"{q}={a.choice if a.choice is not None else round(a.probabilities.get('true', 0.0), 3)}"
                for q, a in answers.items())
            sys.stderr.write(f"  {len(answers)}q {ms:6.1f}ms  {got}\n")
        self._send(200, body)


def serve(weights: str, host: str = "127.0.0.1", port: int = 8088,
          token: Optional[str] = None, calibrated: bool = False,
          device: str = "auto", quiet: bool = False) -> None:
    d = Decider.from_pretrained(weights, device=device)
    _Handler.decider = d
    # One forward pass at a time. The model is not re-entrant and a game loop
    # sending overlapping ticks would otherwise interleave two batches.
    _Handler.lock = threading.Lock()
    _Handler.token = token
    _Handler.calibrated = calibrated
    _Handler.quiet = quiet
    httpd = ThreadingHTTPServer((host, port), _Handler)
    print(f"decision-jef serving {weights} on http://{host}:{port}/v1/systemone"
          f"{' (bearer token required)' if token else ''}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping", flush=True)
    finally:
        httpd.server_close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="decision-jef-serve",
        description="Serve this model at the Jev /v1/systemone endpoint.")
    p.add_argument("--weights", default="BarraHome/Decision-Jef-0.1",
                   help="repository id or a local directory")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8088)
    p.add_argument("--token", default=None,
                   help="require this bearer token; omit to accept any")
    p.add_argument("--calibrated", action="store_true",
                   help="apply the shipped temperatures; read the model card first")
    p.add_argument("--device", default="auto")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)
    serve(a.weights, a.host, a.port, a.token, a.calibrated, a.device, a.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
