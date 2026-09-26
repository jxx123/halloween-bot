#!/usr/bin/env python
"""Robot control web app.

  http://127.0.0.1:8500/        dashboard: live cams, chat with the agent, activity feed
  http://127.0.0.1:8500/face    full-screen robot face (add ?theme=halloween for 🎃)

Endpoints:
  GET  /events            SSE stream of app events (chat, agent thoughts, expressions)
  POST /instruct          {"text": ..., "agent": "claude"|"codex"|"echo"}
  GET  /api/state         proxied robot joint state (robot server :8399)
  GET  /api/agents        available agent backends
  GET|POST /api/policy    π₀-FAST policy status / start {task, seconds, lockstep}
  POST /api/policy/stop   stop the policy run
  POST /api/sim/reset     MuJoCo twin only: arms to rest pose, toys re-placed

Robot server address: HALLOWEEN_ROBOT_API (default http://127.0.0.1:8399) — the real robot
or the MuJoCo twin (python -m halloween_bot.ctl start --sim), same API.

Camera streams come straight from the robot server (:8399/stream?name=...) as MJPEG.
Run with the lerobot env python. The robot server must be running for cams/state.
"""
import json
import os
import queue
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

try:
    from .agents import abort_current, agent_names, get_agent  # package mode (-m)
except ImportError:
    from agents import abort_current, agent_names, get_agent  # script mode

PORT = 8500
ROBOT_API = os.environ.get("HALLOWEEN_ROBOT_API", "http://127.0.0.1:8399")
STATIC = Path(__file__).resolve().parent / "static"
LATEST = Path.home() / "lerobot/outputs/claude_robot/latest"

# scene watcher: keeps latest/overhead.jpg fresh so agents always have a current view,
# and can auto-wake the agent when motion is detected (trick-or-treater mode).
WATCH_FILE = Path(__file__).resolve().parent / "watch_config.json"
watch = {
    "interval": 1.0,         # seconds between overhead captures
    "auto_react": False,     # invoke the agent on motion?
    "motion_threshold": 12.0,  # mean abs pixel diff (0-255) on a 160x90 grayscale
    "react_cooldown": 10.0,  # min seconds between auto-reactions
    "agent": "pumpkin",
}
try:  # persisted overrides survive restarts
    watch.update(json.loads(WATCH_FILE.read_text()))
except (FileNotFoundError, json.JSONDecodeError):
    pass
_last_react = 0.0
host = {"on": False}  # master "robot host mode": face ears + motion reactions

REPO_ROOT = Path(__file__).resolve().parents[2]
js_proc = None  # PS4 teleop child, owned by the webapp


def joystick_alive() -> bool:
    if js_proc is not None and js_proc.poll() is None:
        return True
    r = subprocess.run(["pgrep", "-f", "ps4_teleo[p]"], capture_output=True)
    return r.returncode == 0


def stop_joystick():
    global js_proc
    if js_proc is not None and js_proc.poll() is None:
        js_proc.terminate()
        try:
            js_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            js_proc.kill()
    js_proc = None
    subprocess.run(["pkill", "-f", "ps4_teleo[p]"], capture_output=True)


def start_joystick() -> tuple[bool, str]:
    global js_proc
    if joystick_alive():
        return True, "joystick teleop already running"
    log = open(Path.home() / "lerobot/outputs/claude_robot/ps4_webapp.log", "ab")
    js_proc = subprocess.Popen([sys.executable, "-m", "halloween_bot.ps4_teleop"],
                               cwd=REPO_ROOT, stdout=log, stderr=log)
    time.sleep(2)
    if js_proc.poll() is not None:
        return False, "joystick driver exited at start (controller plugged in?)"
    return True, "joystick teleop started"


def robot_teleop(enable: bool) -> dict:
    req = urllib.request.Request(f"{ROBOT_API}/teleop", data=json.dumps({"enabled": enable}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def teleop_mode() -> str:
    try:
        with urllib.request.urlopen(f"{ROBOT_API}/teleop", timeout=4) as r:
            if json.loads(r.read()).get("enabled"):
                return "leaders"
    except Exception:
        pass
    return "joystick" if joystick_alive() else "off"


def set_teleop_mode(mode: str) -> dict:
    if mode not in ("off", "leaders", "joystick"):
        return {"ok": False, "error": f"unknown mode {mode!r}"}
    if mode == "off":
        stop_joystick()
        try:
            robot_teleop(False)
        except Exception:
            pass
        return {"ok": True, "mode": "off", "msg": "teleop off — arms hold"}
    if mode == "leaders":
        stop_joystick()
        out = robot_teleop(True)
        return {"ok": out.get("ok", False), "mode": teleop_mode(), "msg": out.get("msg", "")}
    # joystick
    try:
        robot_teleop(False)
    except Exception:
        pass
    ok, msg = start_joystick()
    return {"ok": ok, "mode": teleop_mode(), "msg": msg}

TTS_FILE = Path(__file__).resolve().parent / "tts_config.json"
try:
    tts_cfg = json.loads(TTS_FILE.read_text())
except (FileNotFoundError, json.JSONDecodeError):
    tts_cfg = {}


def cartesia_tts(text: str) -> bytes:
    """Text -> mp3 bytes via Cartesia. Raises on any failure (caller falls back)."""
    voice: dict = {"mode": "id", "id": tts_cfg["voice_id"]}
    if tts_cfg.get("controls"):  # emotion/speed prompting (sonic-2 experimental controls)
        voice["__experimental_controls"] = tts_cfg["controls"]
    req = urllib.request.Request(
        "https://api.cartesia.ai/tts/bytes",
        data=json.dumps({
            "model_id": tts_cfg.get("model_id", "sonic-2"),
            "transcript": text,
            "voice": voice,
            "output_format": {"container": "mp3", "bit_rate": 128000, "sample_rate": 44100},
        }).encode(),
        headers={
            "X-API-Key": tts_cfg["api_key"],
            "Cartesia-Version": "2024-11-13",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def robot_call(path: str, payload: dict | None = None, timeout: float = 30) -> tuple[int, dict]:
    """Proxy one JSON call to the robot server; returns (http_code, body)."""
    req = urllib.request.Request(f"{ROBOT_API}{path}")
    if payload is not None:
        req.data = json.dumps(payload).encode()
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        return 502, {"ok": False, "error": f"robot server unreachable: {e}"}


def save_watch():
    try:
        WATCH_FILE.write_text(json.dumps(watch, indent=2))
    except OSError:
        pass

clients: list[queue.Queue] = []
clients_lock = threading.Lock()
history: list[dict] = []  # recent events replayed to new clients
busy = threading.Lock()  # one instruction at a time


def broadcast(event: dict):
    event.setdefault("ts", time.time())
    if event.get("type") != "scene":  # scene ticks are live-only, don't drown chat replay
        history.append(event)
        del history[:-200]
    with clients_lock:
        for q in clients:
            q.put(event)


def scene_watcher():
    global _last_react
    prev = None
    fails = 0
    while True:
        try:
            with urllib.request.urlopen(f"{ROBOT_API}/cam?name=overhead", timeout=15) as r:
                res = json.loads(r.read())
            if not res.get("ok"):
                # C922 video wedges when its mic is opened — self-heal by reconnecting
                fails += 1
                if fails >= 5:
                    fails = 0
                    broadcast({"type": "status", "text": "overhead camera stalled — auto-reconnecting"})
                    req = urllib.request.Request(f"{ROBOT_API}/cam_restart",
                                                 data=json.dumps({"name": "overhead"}).encode(),
                                                 headers={"Content-Type": "application/json"})
                    try:
                        with urllib.request.urlopen(req, timeout=30) as r2:
                            out = json.loads(r2.read())
                        broadcast({"type": "status", "text": out.get("msg", str(out))})
                    except Exception as e:
                        broadcast({"type": "status", "text": f"camera restart failed: {e}"})
            if res.get("ok"):
                fails = 0
                img = cv2.imread(res["path"])
                os.remove(res["path"])  # don't pile up timestamped files every tick
                LATEST.mkdir(parents=True, exist_ok=True)
                tmp = LATEST / "overhead.tmp.jpg"
                cv2.imwrite(str(tmp), img)
                tmp.rename(LATEST / "overhead.jpg")  # atomic: readers never see a half file
                small = cv2.cvtColor(cv2.resize(img, (160, 90)), cv2.COLOR_BGR2GRAY)
                score = float(np.mean(cv2.absdiff(small, prev))) if prev is not None else 0.0
                prev = small
                broadcast({"type": "scene", "motion": round(score, 1)})
                if (watch["auto_react"] and score > watch["motion_threshold"]
                        and not busy.locked() and time.time() - _last_react > watch["react_cooldown"]):
                    _last_react = time.time()
                    text = (f"Motion detected on the overhead camera (score {score:.0f}). "
                            f"Read {LATEST / 'overhead.jpg'} to see the current scene. "
                            "If a person is visible, greet them warmly in one short sentence. "
                            "Otherwise briefly say what changed. Do not move the robot unless "
                            "something clearly requires it.")
                    broadcast({"type": "status", "text": f"auto-react: motion {score:.0f}"})
                    threading.Thread(target=run_instruction, args=(text, watch["agent"]), daemon=True).start()
        except Exception:
            pass  # robot server down or busy — try again next tick
        time.sleep(max(1.0, float(watch["interval"])))


def run_instruction(text: str, agent_name: str):
    global _last_react
    with busy:
        broadcast({"type": "expression", "value": "thinking"})
        agent = get_agent(agent_name)

        def emit(ev):
            if ev["type"] == "say":
                broadcast({"type": "expression", "value": "talking"})
            elif ev["type"] == "action":
                broadcast({"type": "expression", "value": "working"})
            elif ev["type"] == "status" and str(ev.get("text", "")).startswith("error"):
                broadcast({"type": "expression", "value": "sad"})
            broadcast({**ev, "agent": agent.name})

        try:
            agent.run(text, emit)
        except Exception as e:
            broadcast({"type": "status", "text": f"error: {type(e).__name__}: {e}", "agent": agent_name})
            broadcast({"type": "expression", "value": "sad"})
        broadcast({"type": "expression", "value": "idle"})
        # cooldown counts from run END, so the robot's own arm motion during a
        # reaction can't chain-trigger the next one
        _last_react = time.time()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, name, ctype="text/html; charset=utf-8"):
        path = STATIC / name
        if not path.exists():
            self._json({"error": "not found"}, 404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            self._file("dashboard.html")
        elif url.path == "/face":
            self._file("face.html")
        elif url.path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            q: queue.Queue = queue.Queue()
            for ev in history[-50:]:
                q.put(ev)
            with clients_lock:
                clients.append(q)
            try:
                while True:
                    try:
                        ev = q.get(timeout=15)
                        self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with clients_lock:
                    if q in clients:
                        clients.remove(q)
        elif url.path == "/api/state":
            try:
                with urllib.request.urlopen(f"{ROBOT_API}/state", timeout=5) as r:
                    self._json(json.loads(r.read()))
            except Exception as e:
                self._json({"ok": False, "error": str(e)}, 502)
        elif url.path == "/api/agents":
            self._json({"agents": agent_names()})
        elif url.path == "/api/policy":
            code, body = robot_call("/policy", timeout=5)
            self._json(body, code)
        elif url.path == "/api/watch":
            self._json({"ok": True, **watch})
        elif url.path == "/api/host":
            self._json({"ok": True, "on": host["on"]})
        elif url.path == "/api/teleop_mode":
            self._json({"ok": True, "mode": teleop_mode()})
        else:
            self._json({"error": "unknown endpoint"}, 404)

    def do_POST(self):
        url = urlparse(self.path)
        if url.path == "/api/tts":
            n = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
                audio = cartesia_tts(str(payload.get("text", ""))[:1000])
                self.send_response(200)
                self.send_header("Content-Type", "audio/mpeg")
                self.send_header("Content-Length", str(len(audio)))
                self.end_headers()
                self.wfile.write(audio)
            except Exception as e:
                self._json({"error": f"tts failed: {type(e).__name__}: {e}"}, 502)
        elif url.path == "/instruct":
            n = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                self._json({"error": "bad json"}, 400)
                return
            text = (payload.get("text") or "").strip()
            agent = payload.get("agent") or "claude"
            if not text:
                self._json({"error": "empty instruction"}, 400)
                return
            if busy.locked():
                self._json({"ok": False, "error": "agent is busy with a previous instruction"}, 409)
                return
            broadcast({"type": "user", "text": text})
            broadcast({"type": "expression", "value": "listening"})
            threading.Thread(target=run_instruction, args=(text, agent), daemon=True).start()
            self._json({"ok": True})
        elif url.path == "/api/host":
            n = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                self._json({"error": "bad json"}, 400)
                return
            host["on"] = bool(payload.get("enabled"))
            watch["auto_react"] = host["on"]
            save_watch()
            broadcast({"type": "control", "value": "listen_on" if host["on"] else "listen_off"})
            if host["on"]:
                broadcast({"type": "expression", "value": "happy"})
                broadcast({"type": "say", "text": "Pumpkin Bot is awake! Hee hee! Come say hi!"})
                broadcast({"type": "status", "text": "host mode ON: ears + motion reactions active"})
            else:
                if abort_current():
                    broadcast({"type": "status", "text": "in-flight agent run aborted"})
                broadcast({"type": "expression", "value": "idle"})
                broadcast({"type": "status", "text": "host mode OFF: ears + motion reactions stopped"})
            self._json({"ok": True, "on": host["on"]})
        elif url.path == "/api/teleop_mode":
            n = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                self._json({"error": "bad json"}, 400)
                return
            res = set_teleop_mode(str(payload.get("mode", "off")))
            broadcast({"type": "status", "text": f"teleop mode: {res.get('mode')} — {res.get('msg', '')}"})
            self._json(res)
        elif url.path == "/api/teleop":
            n = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
                req = urllib.request.Request(f"{ROBOT_API}/teleop", data=json.dumps(payload).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    res = json.loads(r.read())
                broadcast({"type": "status", "text": res.get("msg", "teleop toggled")})
                self._json(res)
            except Exception as e:
                self._json({"ok": False, "error": str(e)}, 502)
        elif url.path in ("/api/policy", "/api/policy/stop", "/api/sim/reset"):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                self._json({"error": "bad json"}, 400)
                return
            code, body = robot_call(url.path.removeprefix("/api"), payload)
            if body.get("ok"):
                if url.path == "/api/policy":
                    broadcast({"type": "action", "text": f"π policy: {body.get('task')} ({body.get('seconds')} s)"})
                elif url.path == "/api/policy/stop":
                    broadcast({"type": "status", "text": "π policy stopped"})
                else:
                    broadcast({"type": "status", "text": "sim reset: arms at rest, toys re-placed"})
            self._json(body, code)
        elif url.path == "/api/watch":
            n = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                self._json({"error": "bad json"}, 400)
                return
            for k in ("interval", "motion_threshold", "react_cooldown"):
                if k in payload:
                    watch[k] = float(payload[k])
            if "auto_react" in payload:
                watch["auto_react"] = bool(payload["auto_react"])
            if "agent" in payload:
                watch["agent"] = str(payload["agent"])
            save_watch()
            self._json({"ok": True, **watch})
        else:
            self._json({"error": "unknown endpoint"}, 404)


def main():
    threading.Thread(target=scene_watcher, daemon=True).start()
    # Bind localhost plus the tailnet address (if up) — never the plain LAN.
    httpd = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    extra_hosts = []
    try:
        ts_bin = "tailscale"
        import shutil as _sh
        if not _sh.which(ts_bin):
            ts_bin = str(Path.home() / ".local/bin/tailscale")
        ts = subprocess.run([ts_bin, "ip", "-4"], capture_output=True, text=True, timeout=5)
        if ts.returncode == 0 and ts.stdout.strip():
            extra_hosts.append(ts.stdout.strip().splitlines()[0])
    except Exception:
        pass
    for h in extra_hosts:
        try:
            extra = ThreadingHTTPServer((h, PORT), Handler)
            threading.Thread(target=extra.serve_forever, daemon=True).start()
            print(f"also serving on http://{h}:{PORT} (tailnet)", flush=True)
        except OSError as e:
            print(f"tailnet bind {h}:{PORT} failed: {e}", flush=True)
    print(f"robot web app on http://127.0.0.1:{PORT}  (face: /face)", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
