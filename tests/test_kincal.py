import json
import math

import numpy as np
import pytest

from halloween_bot.sim.calib import Calibration
from halloween_bot.sim.model import apply_geometry, build_model

TRUE = {"lengths_m": {"forearm_dx": 0.02, "claw_dx": 0.015}, "joint_offsets_deg": {"shoulder_lift": 3.0}}


def test_calibration_offsets_apply_to_both_arms_and_round_trip():
    base = Calibration.load(geometry=None)
    cal = Calibration(base.calib, offsets={"elbow_flex": 0.1})
    for side in ("left", "right"):
        k = f"{side}_elbow_flex.pos"
        assert cal.to_rad(k, 30) == pytest.approx(base.to_rad(k, 30) + 0.1)
        assert cal.to_norm(k, cal.to_rad(k, 30)) == pytest.approx(30)
    assert cal.to_rad("right_shoulder_pan.pos", 10) == base.to_rad("right_shoulder_pan.pos", 10)


def test_apply_geometry_lengthens_links_and_raises_base():
    m0 = build_model(params={}, props=False, geometry={})
    m1 = build_model(params={}, props=False, geometry={})
    apply_geometry(m1, {"lengths_m": {"upper_arm_dx": 0.01, "forearm_dx": 0.02, "claw_dx": 0.03, "base_dz": 0.005}})
    for side in ("left", "right"):
        n = lambda m, b: np.linalg.norm(m.body(f"{side}_{b}").pos)
        assert n(m1, "lower_arm") - n(m0, "lower_arm") == pytest.approx(0.01)
        assert n(m1, "wrist") - n(m0, "wrist") == pytest.approx(0.02)
        assert np.linalg.norm(m1.body(f"{side}_gripper").pos - m0.body(f"{side}_gripper").pos) == pytest.approx(0.03)
        assert m1.body(f"{side}_base").pos[2] - m0.body(f"{side}_base").pos[2] == pytest.approx(0.005)


def _synthetic_rows(kin_true, grid):
    """Scan the shoulder at each (pan, elbow, wf) until the 'true' model's claws touch the table."""
    rows = []
    for pan, elbow, wf in grid:
        pos = {"right_shoulder_pan.pos": pan, "right_elbow_flex.pos": elbow, "right_wrist_flex.pos": wf,
               "right_wrist_roll.pos": -10.0, "right_gripper.pos": 5.0}
        for lift in np.arange(-95.0, 40.0, 0.5):
            pos["right_shoulder_lift.pos"] = float(lift)
            if kin_true.clearance(pos) <= 0.0:
                rows.append({"grid": {"pan": pan, "elbow": elbow, "wf": wf}, "present": dict(pos),
                             "confirm": {"grew": True}})
                break
    return rows


def test_fit_closes_synthetic_contacts(tmp_path):
    from halloween_bot.sim.kincal import Kin, fit, load_rows
    kin_true = Kin(geometry=TRUE)
    grid = [(p, e, w) for p in (-20, -5, 10) for e in (20, 45, 70) for w in (30, 60)]
    raw = _synthetic_rows(kin_true, grid)
    contacts = [r for r in raw if "present" in r]
    assert len(contacts) >= 6
    f = tmp_path / "contacts.json"
    f.write_text(json.dumps(raw))
    rows = load_rows([f])
    kin0 = Kin(geometry={})
    before = np.mean([abs(kin0.clearance(r.present)) for r in rows if r.present])
    geom = fit(rows, anchors=[], max_nfev=60)
    kin_fit = Kin(geometry=geom)
    after = np.mean([abs(kin_fit.clearance(r.present)) for r in rows if r.present])
    assert after < 0.005 and after < before / 4, (before, after)
    assert set(geom) >= {"lengths_m", "joint_offsets_deg", "fit"}


def test_load_rows_reads_v1_schema():
    from halloween_bot.sim.kincal import load_rows
    import os
    p = os.path.expanduser("~/sim_ref/table_contacts.json")
    if not os.path.exists(p):
        pytest.skip("no table_contacts.json on this machine")
    rows = load_rows([p])
    assert len([r for r in rows if r.present]) == 4 and len(rows) == 8


def _v3_rows():
    start = {"right_shoulder_pan.pos": -5, "right_elbow_flex.pos": 90, "right_wrist_flex.pos": 45,
             "right_wrist_roll.pos": -10.0, "right_gripper.pos": 5.0, "right_shoulder_lift.pos": -65}
    contact = {**start, "right_elbow_flex.pos": 75, "right_shoulder_lift.pos": -55}
    cmd = {**contact, "right_elbow_flex.pos": 82.0, "right_shoulder_lift.pos": -65.5}
    return [
        {"grid": {}, "start": start, "no_contact": True, "lift_floor": -100.0, "elbow_ceil": 95.0},
        {"grid": {}, "start": contact, "cmd_at_contact": cmd, "confirmed": True,
         "present_at_trigger": {**cmd, "right_elbow_flex.pos": 82.6}, "present_at_confirm": {**cmd, "right_elbow_flex.pos": 83.2},
         "frame": "contacts_v3/x.jpg"},
        {"grid": {}, "start": contact, "cmd_at_contact": cmd, "confirmed": False,
         "present_at_trigger": {**cmd}, "present_at_confirm": {**cmd}},
    ]


def test_load_rows_v3_schema(tmp_path):
    from halloween_bot.sim.kincal import load_rows
    f = tmp_path / "v3.json"
    f.write_text(json.dumps(_v3_rows()))
    no, yes, failed = load_rows([f])
    assert no.present is None and len(no.path) > 5
    assert max(p["right_elbow_flex.pos"] for p in no.path) <= 95.0  # elbow ceiling respected
    assert min(p["right_shoulder_lift.pos"] for p in no.path) >= -100.0
    assert yes.present["right_elbow_flex.pos"] == 83.2 and yes.weight == 1.0  # the confirmed (pressing) state
    assert yes.photo == (str(tmp_path / "contacts_v3/x.jpg"), yes.trigger)
    assert failed.present is None and failed.path[-1]["right_shoulder_lift.pos"] == -65.5  # trigger pose is free space


def test_apply_geometry_moves_overhead_camera():
    m = build_model(params={}, props=False, geometry={})
    cam = m.camera("overhead").id
    apply_geometry(m, {"overhead_camera": {"pos": [0.5, 0.01, 0.52], "quat": [0.0, 0.0, 0.0, 1.0]}})
    assert np.allclose(m.cam_pos[cam], [0.5, 0.01, 0.52]) and np.allclose(m.cam_quat[cam], [0, 0, 0, 1])


def test_apply_geometry_base_pitch_tilts_the_arm_forward():
    import mujoco
    m0 = build_model(params={}, props=False, geometry={})
    m1 = build_model(params={}, props=False, geometry={})
    apply_geometry(m1, {"base_pitch_deg": 10.0})
    x = np.zeros(3)
    mujoco.mju_rotVecQuat(x, np.array([1.0, 0, 0]), m1.body("right_base").quat)
    x0 = np.zeros(3)
    mujoco.mju_rotVecQuat(x0, np.array([1.0, 0, 0]), m0.body("right_base").quat)
    assert x0[2] == pytest.approx(0, abs=1e-9) and x[2] == pytest.approx(-math.sin(math.radians(10)), abs=1e-6)


def test_per_side_lengths_and_offsets():
    m0 = build_model(params={}, props=False, geometry={})
    m1 = build_model(params={}, props=False, geometry={})
    apply_geometry(m1, {"lengths_m": {"right": {"forearm_dx": 0.01}, "left": {}}})
    grow = lambda side: np.linalg.norm(m1.body(f"{side}_wrist").pos) - np.linalg.norm(m0.body(f"{side}_wrist").pos)
    assert grow("right") == pytest.approx(0.01) and grow("left") == pytest.approx(0.0)
    base = Calibration.load(geometry=None)
    cal = Calibration(base.calib, offsets={"right": {"elbow_flex": 0.1}})
    assert cal.to_rad("right_elbow_flex.pos", 5) == pytest.approx(base.to_rad("right_elbow_flex.pos", 5) + 0.1)
    assert cal.to_rad("left_elbow_flex.pos", 5) == pytest.approx(base.to_rad("left_elbow_flex.pos", 5))


def _fake_measurements(truth: dict) -> dict:
    """Card-format measurements produced by a 'true' geometry (so the fit has a known answer)."""
    from halloween_bot.sim.kincal import POINTS, Kin, STOCK_MM
    kin = Kin(geometry=truth)
    kin.set(truth)
    meas = {}
    for key, stock in (("A1_lift_to_elbow", "upper_arm"), ("A2_elbow_to_wrist", "forearm"), ("A3_wrist_to_claw_tip", "claw")):
        meas[key] = {s: STOCK_MM[stock] + 1000 * truth["lengths_m"][s].get(f"{stock}_dx", 0.0) for s in ("left", "right")}
    meas["A4_table_to_lift_axis"] = {s: STOCK_MM["lift_axis_height"] for s in ("left", "right")}
    for name, elbow in (("P0", 0.0), ("P1", 20.0)):
        state = {f"{s}_{m}.pos": v for s in ("left", "right") for m, v in
                 (("shoulder_pan", 0), ("shoulder_lift", 0), ("elbow_flex", elbow), ("wrist_flex", 0), ("wrist_roll", 0), ("gripper", 5))}
        meas[name] = {"state": state}
        for label, point in POINTS.items():
            meas[name][label] = {s: 1000 * kin.point_height(state, s, point) for s in ("left", "right")}
    return meas


def test_fit_measured_recovers_offsets(tmp_path):
    from halloween_bot.sim.kincal import fit_measured, load_measurements
    truth = {"lengths_m": {"right": {"forearm_dx": 0.005}, "left": {"upper_arm_dx": -0.004}},
             "joint_offsets_deg": {"right": {"shoulder_lift": 10.0, "elbow_flex": -5.0, "wrist_flex": 3.0},
                                   "left": {"shoulder_lift": 6.0, "elbow_flex": 2.0, "wrist_flex": -4.0}}}
    f = tmp_path / "measurements.json"
    f.write_text(json.dumps(_fake_measurements(truth)))
    fixed, poses = load_measurements(f)
    assert fixed["lengths_m"]["right"]["forearm_dx"] == pytest.approx(0.005, abs=1e-6)
    geom = fit_measured(fixed, poses)
    for side in ("left", "right"):
        for motor, deg in truth["joint_offsets_deg"][side].items():
            assert geom["joint_offsets_deg"][side][motor] == pytest.approx(deg, abs=1.0), (side, motor)


def test_shipped_geometry_matches_the_ruler_heights():
    """geometry.json reproduces Jinyu's 2026-09-27 ruler heights (mm above the tabletop) at the reached states."""
    from halloween_bot.sim.kincal import Kin
    from halloween_bot.sim.model import load_geometry
    p0 = {"left_shoulder_pan.pos": -1.19, "left_shoulder_lift.pos": 0.26, "left_elbow_flex.pos": 3.69,
          "left_wrist_flex.pos": 1.05, "left_wrist_roll.pos": -0.26, "right_shoulder_pan.pos": -1.01,
          "right_shoulder_lift.pos": 0.44, "right_elbow_flex.pos": 3.78, "right_wrist_flex.pos": 1.14,
          "right_wrist_roll.pos": -0.31}
    p1 = {"left_shoulder_pan.pos": -0.66, "left_shoulder_lift.pos": 1.23, "left_elbow_flex.pos": 20.22,
          "left_wrist_flex.pos": 1.14, "left_wrist_roll.pos": -0.26, "right_shoulder_pan.pos": -0.57,
          "right_shoulder_lift.pos": 1.05, "right_elbow_flex.pos": 20.22, "right_wrist_flex.pos": 1.14,
          "right_wrist_roll.pos": -0.4}
    measured = [(p0, "right", (231, 238, 241)), (p0, "left", (231, 239, 270)),
                (p1, "right", (230, 198, 150)), (p1, "left", (230, 196, 180))]
    kin = Kin({"joint_offsets_deg": load_geometry()["joint_offsets_deg"]})
    for state, side, mm in measured:
        got = [1000 * kin.point_height(state, side, p) for p in ("elbow_flex", "wrist_flex", "tip")]
        assert np.abs(np.array(got) - mm).max() < 5.0, (side, got, mm)
