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
