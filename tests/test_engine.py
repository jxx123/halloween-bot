import time

import numpy as np
import pytest

from halloween_bot.sim.calib import KEYS
from halloween_bot.sim.engine import MAX_RELATIVE_TARGET, REST, SimEngine
from halloween_bot.sim.model import PARAMS_FILE, build_model


@pytest.fixture(scope="module")
def eng():
    # command-semantics tests run on Menagerie's stiff servos; the fitted (soft, sagging)
    # servos are covered by test_fitted_servos_sag_like_the_real_arm and tests/test_sysid.py
    return SimEngine(model=build_model(params={}))


def test_reset_is_rest_pose(eng):
    eng.reset()
    pos = eng.read_positions()
    assert list(pos) == KEYS
    for k in KEYS:
        assert pos[k] == pytest.approx(REST[k], abs=0.5)


def test_rest_pose_is_stable(eng):
    eng.reset()
    eng.step(2.0)
    pos = eng.read_positions()
    for k in KEYS:
        assert abs(pos[k] - REST[k]) < 6, (k, pos[k])


def test_send_action_clamps_relative_to_present(eng):
    eng.reset()
    sent = eng.send_action({"right_elbow_flex.pos": 0.0})
    assert sent["right_elbow_flex.pos"] == pytest.approx(REST["right_elbow_flex.pos"] - MAX_RELATIVE_TARGET, abs=0.6)
    with pytest.raises(KeyError):
        eng.send_action({"right_elbow.pos": 1})


def test_move_reaches_target_headless(eng):
    eng.reset()
    out = eng.move({"right_shoulder_lift.pos": -60, "right_elbow_flex.pos": 70, "left_gripper.pos": 80}, 2.0)
    assert out["ok"]
    eng.step(1.0)
    pos = eng.read_positions()
    assert pos["right_shoulder_lift.pos"] == pytest.approx(-60, abs=4)
    assert pos["right_elbow_flex.pos"] == pytest.approx(70, abs=4)
    assert pos["left_gripper.pos"] == pytest.approx(80, abs=4)


@pytest.mark.skipif(not PARAMS_FILE.exists(), reason="no fitted params.json")
def test_fitted_servos_sag_like_the_real_arm():
    """Real elbow holds a few units above command when it comes down to a target (sys-ID report)."""
    e = SimEngine()  # loads params.json
    e.move({"right_shoulder_lift.pos": -60, "right_elbow_flex.pos": 70}, 2.0)
    e.step(1.0)
    over = e.read_positions()["right_elbow_flex.pos"] - 70
    assert 0.5 < over < 9, over


def test_move_unknown_key_returns_real_error_shape(eng):
    out = eng.move({"bogus.pos": 1}, 1.0)
    assert out["error"].startswith("unknown joint keys") and "valid_keys" in out


def test_trajectory_trace_shape(eng):
    eng.reset()
    out = eng.run_trajectory([{"right_wrist_flex.pos": 50}] * 5 + [{"right_wrist_flex.pos": 55}] * 5, hz=30)
    tr = out["trace"]
    assert len(tr) == 11 and tr[-1]["cmd"] is None
    assert tr[0]["cmd"]["right_wrist_flex.pos"] == 50 and set(tr[0]["present"]) == set(KEYS)
    assert tr[5]["t"] == pytest.approx(5 / 30, abs=0.01)


def test_frames_headless_and_threaded():
    e = SimEngine()
    f = e.get_frame("left_wrist")
    assert f.shape == (480, 640, 3) and f.dtype == np.uint8 and f.mean() > 1
    e.start()
    try:
        f2 = e.get_frame("overhead")
        assert f2.shape == (720, 1280, 3)
        e.pause()
        assert e.paused
        time.sleep(0.05)
        t = e.data.time
        time.sleep(0.3)
        assert e.data.time == t
        e.resume()
        time.sleep(0.3)
        assert e.data.time > t
    finally:
        e.stop()
    with pytest.raises(KeyError):
        e.get_frame("nope")
