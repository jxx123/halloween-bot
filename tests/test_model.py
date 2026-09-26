import mujoco
import numpy as np

from halloween_bot.sim.calib import KEYS, Calibration, joint_name
from halloween_bot.sim.model import CAMERA_NAMES, TOYS, apply_params, build_model


def test_actuators_named_like_lerobot_keys():
    m = build_model(params={})
    assert [m.actuator(i).name for i in range(m.nu)] == [joint_name(k) for k in KEYS]


def test_cameras_and_props_exist():
    m = build_model(params={})
    for cam in CAMERA_NAMES.values():
        assert m.camera(cam).id >= 0
    for toy in TOYS:
        assert m.joint(f"{toy}_free").type == mujoco.mjtJoint.mjJNT_FREE
    assert m.body("basket").id >= 0


def test_joint_ranges_cover_calibrated_ranges():
    cal = Calibration.load()
    m = build_model(params={}, calib=cal)
    for k in KEYS:
        lo, hi = m.joint(joint_name(k)).range
        for n in (0, 100) if k.endswith("gripper.pos") else (-100, 100):
            assert lo - 1e-6 <= cal.to_rad(k, n) <= hi + 1e-6, k


def test_apply_params_sets_gain_bias_damping():
    m = build_model(params={})
    apply_params(m, {"motors": {"elbow_flex": {"kp": 12.0, "kv": 0.5, "damping": 0.3,
                                               "frictionloss": 0.04, "armature": 0.02, "forcerange": 2.0}}})
    for side in ("left", "right"):
        a = m.actuator(f"{side}_elbow_flex")
        j = m.joint(f"{side}_elbow_flex")
        assert a.gainprm[0] == 12.0 and a.biasprm[1] == -12.0 and a.biasprm[2] == -0.5
        assert np.allclose(a.forcerange, [-2.0, 2.0])
        assert m.dof_damping[j.dofadr[0]] == 0.3 and m.dof_armature[j.dofadr[0]] == 0.02
        assert m.dof_frictionloss[j.dofadr[0]] == 0.04


def test_no_props_model_has_no_toys():
    m = build_model(params={}, props=False)
    assert all(m.joint(i).type != mujoco.mjtJoint.mjJNT_FREE for i in range(m.njnt))
