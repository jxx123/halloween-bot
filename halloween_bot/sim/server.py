#!/usr/bin/env python
"""Sim robot server: the MuJoCo twin behind the SAME HTTP API as halloween_bot/server.py, so the
web app, ctl.py and the Claude agent work unchanged against it.

  GET  /state  POST /move  GET /cam?name=  GET /stream?name=  POST /teleop  POST /cam_restart
  POST /stop                                   (same JSON shapes as the real server)
  POST /sim/reset {"randomize": bool}          arms to rest pose, toys re-placed
  POST /trajectory {"points": [...], "hz": 30} sys-ID: commanded sequence -> per-tick trace
  GET|POST /policy  POST /policy/stop          π₀-FAST via the shared PolicyRunner

Cameras: overhead, left_wrist, right_wrist (as on the rig) + scene (third-person view).
Start via:  python -m halloween_bot.ctl start --sim
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

from ..policy_runner import PI0FAST_CAMERAS, PI0FAST_IMAGE_HW, PolicyRunner, fit_image, handle_http
from .calib import KEYS
from .engine import SimEngine
from .model import CAMERA_NAMES

PORT = 8399
FRAMES_DIR = Path.home() / "lerobot/outputs/claude_robot/frames"


def policy_observation(engine: SimEngine) -> dict:
    """Joint state + the three rig cameras renamed/shaped the way the real pi0fast client sent them."""
    obs = engine.read_positions()
    for cam, feature in PI0FAST_CAMERAS.items():
        obs[feature] = fit_image(engine.get_frame(cam))
    return obs


def policy_features() -> dict:
    return {**{k: float for k in KEYS}, **{f: (*PI0FAST_IMAGE_HW, 3) for f in PI0FAST_CAMERAS.values()}}


def make_app(engine: SimEngine, runner=None, frames_dir: Path = FRAMES_DIR, on_stop=None):
    busy = threading.Lock()  # one motion command at a time, like the real server's lock

    def policy_running() -> bool:
        return runner is not None and runner.running

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _reply(self, obj, code=200):
            body = json.dumps(obj, indent=2, default=float).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _stream(self, name):
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    frame = engine.get_frame(name)
                    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                                           [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         + f"Content-Length: {len(jpg)}\r\n\r\n".encode()
                                         + jpg.tobytes() + b"\r\n")
                    time.sleep(1 / 15)
            except (BrokenPipeError, ConnectionResetError):
                pass  # viewer closed the tab
            except Exception as e:  # headers are already sent: end the stream, don't write a reply into it
                print(f"stream {name} ended: {type(e).__name__}: {e}", flush=True)

        def do_GET(self):
            url = urlparse(self.path)
            name = parse_qs(url.query).get("name", ["left_wrist"])[0]
            try:
                if url.path == "/state":
                    self._reply({"ok": True, "positions": engine.read_positions(), "teleop": False, "sim": True,
                                 "paused": engine.paused,
                                 "policy": runner.status() if runner is not None else None})
                elif url.path == "/teleop":
                    self._reply({"ok": True, "enabled": False})
                elif url.path == "/cam":
                    if name not in CAMERA_NAMES:
                        self._reply({"error": f"unknown camera '{name}'", "cameras": list(CAMERA_NAMES)})
                        return
                    frame = engine.get_frame(name)
                    frames_dir.mkdir(parents=True, exist_ok=True)
                    path = frames_dir / f"{name}_{time.strftime('%H%M%S')}.jpg"
                    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    self._reply({"ok": True, "path": str(path)})
                elif url.path == "/stream":
                    if name not in CAMERA_NAMES:
                        self._reply({"error": f"unknown camera '{name}'"}, 404)
                        return
                    self._stream(name)
                elif runner is not None and (r := handle_http(runner, "GET", url.path, {})):
                    self._reply(r[1], r[0])
                else:
                    self._reply({"error": "unknown endpoint"}, 404)
            except Exception as e:
                self._reply({"error": f"{type(e).__name__}: {e}"}, 500)

        def do_POST(self):
            url = urlparse(self.path)
            try:
                n = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(n) or b"{}")
                if url.path in ("/move", "/trajectory"):
                    with busy:
                        if policy_running():  # checked under the lock: a queued /move can't slip past a start
                            self._reply({"error": "the π policy is driving the arms — POST /policy/stop first."},
                                        409)
                        elif url.path == "/move":
                            self._reply(engine.move(payload.get("targets", {}), payload.get("duration", 2.0)))
                        else:
                            self._reply(engine.run_trajectory(payload.get("points", []), payload.get("hz", 30.0)))
                elif url.path == "/policy" and runner is not None:
                    if not busy.acquire(blocking=False):
                        self._reply({"ok": False, "error": "a move is in progress — try again when it finishes"}, 409)
                        return
                    try:
                        r = handle_http(runner, "POST", url.path, payload)
                    finally:
                        busy.release()
                    self._reply(r[1], r[0])
                elif url.path == "/sim/reset":
                    if policy_running():
                        runner.stop()
                    with busy:
                        engine.reset(randomize=bool(payload.get("randomize")), seed=payload.get("seed"))
                    self._reply({"ok": True, "msg": "sim reset: arms at rest pose, toys re-placed"})
                elif url.path == "/teleop":
                    self._reply({"ok": False, "error": "no leader arms in sim — use /move or /policy"})
                elif url.path == "/cam_restart":
                    self._reply({"ok": True, "msg": "sim cameras never stall"})
                elif url.path == "/stop":
                    self._reply({"ok": True, "msg": "sim server shutting down"})
                    if on_stop:
                        threading.Thread(target=on_stop, daemon=True).start()
                elif runner is not None and (r := handle_http(runner, "POST", url.path, payload)):
                    self._reply(r[1], r[0])
                else:
                    self._reply({"error": "unknown endpoint"}, 404)
            except Exception as e:
                self._reply({"error": f"{type(e).__name__}: {e}"}, 500)

    return Handler


def _tailnet_hosts() -> list[str]:
    ts_bin = shutil.which("tailscale") or str(Path.home() / ".local/bin/tailscale")
    try:
        ts = subprocess.run([ts_bin, "ip", "-4"], capture_output=True, text=True, timeout=5)
        if ts.returncode == 0 and ts.stdout.strip():
            return [ts.stdout.strip().splitlines()[0]]
    except Exception:
        pass
    return []


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--policy-server", default="127.0.0.1:8081",
                    help="lerobot policy_server for π (:8080 belongs to the 1080's real-robot client)")
    ap.add_argument("--no-policy", action="store_true")
    args = ap.parse_args(argv)

    engine = SimEngine()
    engine.start()
    runner = None
    if not args.no_policy:
        runner = PolicyRunner(lambda: policy_observation(engine), engine.send_action, KEYS, policy_features(),
                              server_address=args.policy_server, pause=engine.pause, resume=engine.resume)
    servers: list[ThreadingHTTPServer] = []

    def shutdown():
        time.sleep(0.3)
        if runner is not None:
            runner.stop()
        engine.stop()
        for s in servers:
            s.shutdown()

    handler = make_app(engine, runner, on_stop=shutdown)
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    servers.append(httpd)
    # Bind localhost plus the tailnet address (if up) — never the plain LAN.
    for h in _tailnet_hosts():
        try:
            extra = ThreadingHTTPServer((h, args.port), handler)
            servers.append(extra)
            threading.Thread(target=extra.serve_forever, daemon=True).start()
            print(f"also serving on http://{h}:{args.port} (tailnet)", flush=True)
        except OSError as e:
            print(f"tailnet bind {h}:{args.port} failed: {e}", flush=True)
    print(f"SIM robot server on http://127.0.0.1:{args.port} (policy server {args.policy_server})", flush=True)
    httpd.serve_forever()
    print("sim server stopped", flush=True)


if __name__ == "__main__":
    main()
