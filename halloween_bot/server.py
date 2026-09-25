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
import subprocess
import sys
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
            if url.path == "/state":
                with lock:
                    self._reply({"ok": True, "positions": read_positions(), "teleop": teleop_on.is_set()})
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
            if url.path == "/move":
                if teleop_on.is_set():
                    self._reply({"error": "teleop is active — the human has the controls. Turn teleop off first."}, 409)
                    return
                with lock:
                    self._reply(do_move(payload.get("targets", {}), payload.get("duration", 2.0)))
            elif url.path == "/teleop":
                self._reply(set_teleop(bool(payload.get("enabled"))))
            elif url.path == "/cam_restart":
                with lock:
                    self._reply(restart_camera(payload.get("name", "overhead")))
            elif url.path == "/stop":
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
    httpd = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"serving on http://127.0.0.1:{PORT}", flush=True)
    httpd.serve_forever()
    print("server stopped", flush=True)


if __name__ == "__main__":
    main()
