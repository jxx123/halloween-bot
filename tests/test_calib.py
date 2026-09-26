import math

import pytest

from halloween_bot.sim.calib import KEYS, Calibration, clamp_norm, joint_name, split_key

CAL = Calibration.load()


def test_keys_order_matches_lerobot_bimanual():
    assert KEYS[0] == "left_shoulder_pan.pos" and KEYS[5] == "left_gripper.pos"
    assert KEYS[6] == "right_shoulder_pan.pos" and KEYS[11] == "right_gripper.pos"
    assert split_key("right_wrist_roll.pos") == ("right", "wrist_roll")
    assert joint_name("left_gripper.pos") == "left_gripper"


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("n", [-100, -37.5, 0, 12.25, 100])
def test_round_trip(key, n):
    n = clamp_norm(key, n)
    assert CAL.to_norm(key, CAL.to_rad(key, n)) == pytest.approx(n, abs=1e-9)


def test_body_zero_is_range_middle_and_scale_from_ticks():
    assert CAL.to_rad("left_shoulder_pan.pos", 0) == 0.0
    # left pan range 1255..2900 ticks -> half span 822.5 ticks
    assert CAL.to_rad("left_shoulder_pan.pos", 100) == pytest.approx(822.5 * 2 * math.pi / 4095)


def test_gripper_maps_onto_jaw_range():
    assert CAL.to_rad("right_gripper.pos", 0) == pytest.approx(-0.174533)
    assert CAL.to_rad("right_gripper.pos", 100) == pytest.approx(1.7453292)


def test_clamp_norm_bounds():
    assert clamp_norm("left_gripper.pos", -5) == 0 and clamp_norm("left_gripper.pos", 150) == 100
    assert clamp_norm("left_elbow_flex.pos", -150) == -100


def test_homing_zero_shifts_by_range_middle_minus_2047_per_arm():
    base = Calibration.load(geometry=None)
    cal = Calibration(base.calib, zero={"shoulder_lift": "homing"})
    for side in ("left", "right"):
        c = base.calib[side]["shoulder_lift"]
        shift = ((c["range_min"] + c["range_max"]) / 2 - 2047) * 2 * math.pi / 4095
        k = f"{side}_shoulder_lift.pos"
        assert cal.to_rad(k, 0) == pytest.approx(shift)  # normalized 0 = range middle, now `shift` from model zero
        assert cal.to_norm(k, cal.to_rad(k, -40)) == pytest.approx(-40)
    assert cal.to_rad("right_elbow_flex.pos", 10) == base.to_rad("right_elbow_flex.pos", 10)  # untouched joints
    with pytest.raises(ValueError):
        Calibration(base.calib, zero={"gripper": "homing"})
