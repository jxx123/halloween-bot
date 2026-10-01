#!/usr/bin/env python
"""Persistent control server bridging the bimanual SO101 followers to a Claude Code agent.

Holds the robot connection (torque on, arms hold position between commands) and
exposes a minimal JSON API on 127.0.0.1:8399:

  GET  /state                  -> normalized joint positions for both arms (+ teleop flag)
  POST /move                   -> {"targets": {"left_gripper.pos": 50, ...}, "duration": 2.0}
                                  smooth linear interpolation to targets; unspecified joints hold
  GET  /cam?name=left_wrist    -> captures a frame, saves JPEG, returns its path
  GET  /stream?name=overhead   -> MJPEG live stream
  POST /teleop                 -> {"enabled": true} mirror leader arms at 50 Hz (cams stay live)
  POST /cam_restart            -> {"name": "overhead"} reconnect a stalled camera
  POST /stop                   -> disconnect robot (torque OFF - arms droop!) and exit

Built on upstream lerobot >= 0.6.1 (`bi_so_follower` / `bi_so_leader`).
Start via:  python -m halloween_bot.ctl start   (refuses to run while teleop/record own the buses)
"""
import json
import os
import subprocess
import sys
import urllib.request
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.bi_so_follower import BiSOFollower, BiSOFollowerConfig
from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig
from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderConfig

from halloween_bot.policy_runner import (
    PI0FAST_CAMERAS,
    PI0FAST_IMAGE_HW,
    PolicyRunner,
    fit_image,
    letterbox,
    POLICY_IMAGE_SIZE,
    handle_http as policy_http,
)

PORT = 8399
CONTROL_HZ = 30
FRAMES_DIR = Path.home() / "lerobot/outputs/claude_robot/frames"

L_FOLLOWER = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6057297-if00"  # orange / left
R_FOLLOWER = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6083852-if00"  # blue / right
L_LEADER = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6055292-if00"  # orange / left
R_LEADER = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6055274-if00"  # blue / right
LEFT_CAM = "/dev/video2"
RIGHT_CAM = "/dev/video0"
# Logitech C922, unique serial -> stable path. The two wrist cams are identical models
# (colliding by-id names), so they stay on raw /dev/video* paths.
OVERHEAD_CAM = "/dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_59D955BF-video-index0"


def refuse_if_bus_busy():
    out = subprocess.run(["pgrep", "-f", "lerobot-(teleoperate|record|rollout)"], capture_output=True, text=True)
    if out.stdout.strip():
        sys.exit(
            "REFUSING TO START: lerobot-teleoperate/record/rollout is running and owns the serial buses.\n"
            "Stop it first: pkill -INT -f 'lerobot-(teleoperate|record|rollout)'"
        )


def make_robot() -> BiSOFollower:
    cfg = BiSOFollowerConfig(
        id="bimanual",
        left_arm_config=SOFollowerConfig(
            port=L_FOLLOWER,
            # per-tick safety cap; must be float (upstream isinstance check rejects int)
            max_relative_target=20.0,
        ),
        right_arm_config=SOFollowerConfig(
            port=R_FOLLOWER,
            max_relative_target=20.0,
        ),
        # top-level cameras keep unprefixed keys; lerobot attaches them to left_arm
        cameras={
            "left_wrist": OpenCVCameraConfig(index_or_path=LEFT_CAM, fps=30, width=640, height=480, fourcc="MJPG"),
            "right_wrist": OpenCVCameraConfig(index_or_path=RIGHT_CAM, fps=30, width=640, height=480, fourcc="MJPG"),
            "overhead": OpenCVCameraConfig(index_or_path=OVERHEAD_CAM, fps=30, width=1280, height=720, fourcc="MJPG"),
        },
    )
    return BiSOFollower(cfg)


robot = None
lock = threading.Lock()
httpd = None

leader = None
teleop_on = threading.Event()
runner = None  # PolicyRunner, created in main() after the robot connects


def cams() -> dict:
    """Top-level cameras live on the left arm in lerobot's BiSOFollower (unprefixed keys)."""
    return robot.left_arm.cameras


def read_positions() -> dict[str, float]:
    left = robot.left_arm.bus.sync_read("Present_Position")
    right = robot.right_arm.bus.sync_read("Present_Position")
    return {f"left_{m}.pos": v for m, v in left.items()} | {f"right_{m}.pos": v for m, v in right.items()}


def clamp(key: str, value: float) -> float:
    lo, hi = (0.0, 100.0) if key.endswith("gripper.pos") else (-100.0, 100.0)
    return max(lo, min(hi, float(value)))


def do_move(targets: dict[str, float], duration: float) -> dict:
    valid_keys = set(robot.action_features)
    unknown = [k for k in targets if k not in valid_keys]
    if unknown:
        return {"error": f"unknown joint keys: {unknown}", "valid_keys": sorted(valid_keys)}
    duration = max(0.5, min(10.0, float(duration)))

    start = read_positions()
    goal = {**start, **{k: clamp(k, v) for k, v in targets.items()}}
    steps = max(1, int(duration * CONTROL_HZ))
    for i in range(1, steps + 1):
        a = i / steps
        action = {k: start[k] + (goal[k] - start[k]) * a for k in goal}
        robot.send_action(action)
        time.sleep(1.0 / CONTROL_HZ)
    final = read_positions()
    return {"ok": True, "requested": {k: goal[k] for k in targets}, "reached": {k: final[k] for k in targets}}


def do_trajectory(points: list[dict], hz: float) -> dict:
    """Stream raw waypoints at a fixed rate (sys-ID collector). Same clamps/caps as /move.

    `present` is sampled BEFORE each send so trace rows align command with the state it
    was issued against. Unspecified joints hold the previous command (not the sagged
    present), so callers should pin the idle arm explicitly in every point.
    """
    valid = set(robot.action_features)
    unknown = sorted({k for p in points for k in p if k not in valid})
    if unknown:
        return {"error": f"unknown joint keys: {unknown}", "valid_keys": sorted(valid)}
    hz = max(1.0, min(100.0, float(hz)))
    hold = read_positions()
    trace = []
    t0 = time.perf_counter()
    for i, p in enumerate(points, start=1):
        present = read_positions()
        cmd = {**hold, **{k: clamp(k, v) for k, v in p.items()}}
        hold = cmd
        robot.send_action(cmd)  # per-tick max_relative_target cap still applies
        trace.append({"t": time.perf_counter() - t0, "present": present, "cmd": cmd})
        time.sleep(max(0.0, i / hz - (time.perf_counter() - t0)))
    trace.append({"t": time.perf_counter() - t0, "present": read_positions(), "cmd": None})
    return {"ok": True, "hz": hz, "trace": trace}


def restart_camera(name: str) -> dict:
    """Reconnect a stalled camera (C922 video can wedge when its mic is opened)."""
    cam = cams().get(name)
    if cam is None:
        return {"error": f"unknown camera '{name}'", "cameras": list(cams())}
    try:
        cam.disconnect()
    except Exception as e:
        print(f"camera {name} disconnect during restart: {e}", flush=True)
    time.sleep(1.0)
    cam.connect()
    return {"ok": True, "msg": f"camera '{name}' reconnected"}


def do_cam(name: str) -> dict:
    if name not in cams():
        return {"error": f"unknown camera '{name}'", "cameras": list(cams())}
    frame = cams()[name].async_read(timeout_ms=2000)  # RGB
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    path = FRAMES_DIR / f"{name}_{time.strftime('%H%M%S')}.jpg"
    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return {"ok": True, "path": str(path)}


def teleop_loop():
    """Mirror leader arms onto followers at ~50 Hz. Cameras/state stay live throughout."""
    while teleop_on.is_set():
        with lock:
            try:
                robot.send_action(leader.get_action())
            except Exception as e:
                print(f"teleop tick failed: {e}", flush=True)
        time.sleep(1 / 50)
    print("teleop loop ended", flush=True)


def set_teleop(enable: bool) -> dict:
    global leader
    if enable:
        if teleop_on.is_set():
            return {"ok": True, "enabled": True, "msg": "already on"}
        if leader is None:
            leader = BiSOLeader(BiSOLeaderConfig(
                id="bimanual",
                left_arm_config=SOLeaderConfig(port=L_LEADER),
                right_arm_config=SOLeaderConfig(port=R_LEADER),
            ))
        if not leader.is_connected:
            leader.connect(calibrate=False)
        teleop_on.set()
        threading.Thread(target=teleop_loop, daemon=True).start()
        return {"ok": True, "enabled": True, "msg": "teleop ON — followers now mirror the leader arms"}
    else:
        teleop_on.clear()
        return {"ok": True, "enabled": False, "msg": "teleop OFF — followers hold their pose"}


def policy_obs() -> dict:
    # Camera reads stay OUTSIDE the robot lock: a wedged C922 (2 s timeout x3) must not
    # stall policy_send / /move. async_read only copies the camera thread's latest frame.
    # letterbox (aspect-preserving pad to 224x224) BEFORE sending: lerobot's async server would
    # otherwise squash 4:3 -> 1:1, feeding the policy images it never trained on.
    frames = {PI0FAST_CAMERAS[n]: letterbox(fit_image(cams()[n].async_read(timeout_ms=2000))) for n in PI0FAST_CAMERAS}
    with lock:
        obs = read_positions()
    return {**obs, **frames}


def policy_send(action: dict) -> None:
    with lock:
        robot.send_action(action)


def sim_holds_policy_server() -> str | None:
    """The 5090 sim shares the pi policy server; refuse to start while its run is live.

    A lerobot policy_server has one observation queue and no client identity — overlapped
    runs would hand the real arms chunks computed from SIM frames. Requires the tunnel
    forward -L 18399:127.0.0.1:8399 (the sim server) next to the 8080 policy forward.
    """
    try:
        with urllib.request.urlopen("http://127.0.0.1:18399/policy", timeout=2) as r:
            if json.loads(r.read()).get("running"):
                return "the 5090 sim is running pi on the shared policy server"
    except Exception:
        pass  # sim down or unreachable = not using the shared server
    return None


def policy_running() -> bool:
    return bool(runner and runner.status().get("running"))


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

    def do_GET(self):
        url = urlparse(self.path)
        try:
            if runner is not None and (r := policy_http(runner, "GET", url.path, {})):
                self._reply(r[1], r[0])
            elif url.path == "/state":
                with lock:
                    self._reply({"ok": True, "positions": read_positions(), "teleop": teleop_on.is_set(),
                                 "policy": runner.status() if runner else None})
            elif url.path == "/teleop":
                self._reply({"ok": True, "enabled": teleop_on.is_set()})
            elif url.path == "/cam":
                name = parse_qs(url.query).get("name", ["left_wrist"])[0]
                with lock:
                    self._reply(do_cam(name))
            elif url.path == "/stream":
                # MJPEG live stream; no global lock — async_read only grabs the camera
                # thread's latest frame, so /move keeps running while browsers watch.
                name = parse_qs(url.query).get("name", ["left_wrist"])[0]
                if name not in cams():
                    self._reply({"error": f"unknown camera '{name}'"}, 404)
                    return
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        frame = cams()[name].async_read(timeout_ms=2000)
                        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                                               [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if ok:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                             + f"Content-Length: {len(jpg)}\r\n\r\n".encode()
                                             + jpg.tobytes() + b"\r\n")
                        time.sleep(1 / 15)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # viewer closed the tab
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
                if teleop_on.is_set():
                    self._reply({"error": "teleop is active — the human has the controls. Turn teleop off first."}, 409)
                    return
                with lock:
                    if policy_running():  # checked under the lock: a queued /move can't slip past a start
                        self._reply({"error": "the pi policy is driving the arms — POST /policy/stop first."}, 409)
                    elif url.path == "/move":
                        self._reply(do_move(payload.get("targets", {}), payload.get("duration", 2.0)))
                    else:
                        self._reply(do_trajectory(payload.get("points", []), payload.get("hz", 30.0)))
            elif url.path == "/policy" and runner is not None:
                if not lock.acquire(blocking=False):
                    self._reply({"ok": False, "error": "a move is in progress — try again when it finishes"}, 409)
                    return
                try:
                    r = policy_http(runner, "POST", url.path, payload)
                finally:
                    lock.release()
                self._reply(r[1], r[0])
            elif url.path == "/teleop":
                if bool(payload.get("enabled")) and policy_running():
                    self._reply({"error": "the pi policy is driving the arms — POST /policy/stop first."}, 409)
                    return
                self._reply(set_teleop(bool(payload.get("enabled"))))
            elif url.path == "/cam_restart":
                with lock:
                    self._reply(restart_camera(payload.get("name", "overhead")))
            elif runner is not None and (r := policy_http(runner, "POST", url.path, payload)):
                self._reply(r[1], r[0])
            elif url.path == "/stop":
                if policy_running():
                    runner.stop()
                self._reply({"ok": True, "msg": "disconnecting (torque off) and shutting down"})
                threading.Thread(target=shutdown, daemon=True).start()
            else:
                self._reply({"error": "unknown endpoint"}, 404)
        except Exception as e:
            self._reply({"error": f"{type(e).__name__}: {e}"}, 500)


def shutdown():
    time.sleep(0.3)
    teleop_on.clear()
    with lock:
        try:
            robot.disconnect()
        finally:
            httpd.shutdown()


def main():
    global robot, httpd
    refuse_if_bus_busy()
    robot = make_robot()
    print("connecting robot (arms will hold their current pose)...", flush=True)
    robot.connect(calibrate=False)
    print("robot connected, torque ON", flush=True)
    global runner
    runner = PolicyRunner(
        policy_obs,
        policy_send,
        list(robot.action_features),
        {**{k: float for k in robot.action_features},
         **{f: (POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE, 3) for f in PI0FAST_CAMERAS.values()}},
        server_address="127.0.0.1:8080",  # local tunnel endpoint -> 5090 policy server
        checkpoint=os.environ.get("POLICY_CHECKPOINT", "delvingdeep/pi0fast-so101-bimanual"),
        policy_type=os.environ.get("POLICY_TYPE", "pi0_fast"),
        blocked_by=sim_holds_policy_server,
    )
    print(f"pi policy runner mounted: POST /policy {{task, seconds}} | "
          f"type={runner.policy_type} ckpt={runner.checkpoint}", flush=True)
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
    print(f"serving on http://127.0.0.1:{PORT}", flush=True)
    httpd.serve_forever()
    print("server stopped", flush=True)


if __name__ == "__main__":
    main()
