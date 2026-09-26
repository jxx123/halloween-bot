import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from halloween_bot.sim.calib import KEYS
from halloween_bot.sim.engine import SimEngine
from halloween_bot.sim.server import make_app


class FakeRunner:  # stands in for PolicyRunner
    running = False

    def status(self):
        return {"running": self.running, "phase": "idle"}

    def start(self, *a):
        self.running = True
        return self.status()

    def stop(self):
        self.running = False
        return self.status()


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    eng = SimEngine()
    eng.start()
    runner = FakeRunner()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_app(eng, runner, tmp_path_factory.mktemp("frames")))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}", runner
    httpd.shutdown()
    eng.stop()


def call(url, payload=None):
    req = urllib.request.Request(url)
    if payload is not None:
        req.data = json.dumps(payload).encode()
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_state_matches_real_shape(base):
    url, _ = base
    code, st = call(url + "/state")
    assert code == 200 and st["ok"] and st["sim"] is True and st["teleop"] is False
    assert list(st["positions"]) == KEYS and st["policy"]["phase"] == "idle"


def test_move_and_cam(base):
    url, _ = base
    code, out = call(url + "/move", {"targets": {"left_gripper.pos": 70}, "duration": 1.0})
    assert code == 200 and out["ok"] and abs(out["reached"]["left_gripper.pos"] - 70) < 8
    code, cam = call(url + "/cam?name=overhead")
    assert code == 200 and cam["ok"] and cam["path"].endswith(".jpg")


def test_move_unknown_key_and_bounds(base):
    url, _ = base
    _, out = call(url + "/move", {"targets": {"nope.pos": 1}})
    assert out["error"].startswith("unknown joint keys") and "valid_keys" in out
    _, out = call(url + "/move", {"targets": {"left_gripper.pos": 250}, "duration": 0.5})
    assert out["requested"]["left_gripper.pos"] == 100


def test_unknown_camera_and_teleop(base):
    url, _ = base
    assert "error" in call(url + "/cam?name=nope")[1]
    assert call(url + "/stream?name=nope")[0] == 404
    _, out = call(url + "/teleop", {"enabled": True})
    assert out["ok"] is False and "sim" in out["error"]


def test_move_rejected_while_policy_runs(base):
    url, runner = base
    runner.running = True
    try:
        code, out = call(url + "/move", {"targets": {"left_gripper.pos": 50}})
        assert code == 409 and "policy" in out["error"]
        assert call(url + "/trajectory", {"points": [{}]})[0] == 409
    finally:
        runner.running = False


def test_stream_serves_mjpeg(base):
    url, _ = base
    with urllib.request.urlopen(url + "/stream?name=scene", timeout=10) as r:
        assert r.headers["Content-Type"].startswith("multipart/x-mixed-replace")
        assert r.read(2048).startswith(b"--frame")


def test_reset_and_trajectory(base):
    url, _ = base
    assert call(url + "/sim/reset", {"randomize": True})[1]["ok"]
    code, out = call(url + "/trajectory", {"points": [{"right_wrist_flex.pos": 50}] * 3, "hz": 30})
    assert code == 200 and len(out["trace"]) == 4


def test_policy_routes_mounted(base):
    url, _ = base
    code, out = call(url + "/policy")
    assert code == 200 and out["phase"] == "idle"
