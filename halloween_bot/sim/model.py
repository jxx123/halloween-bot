"""Bimanual SO101 scene: two Menagerie arms on the back edge of a table, mats, a basket, two
plush toys, and the rig's cameras. Composed with MjSpec so joint/actuator names come out as
lerobot keys (prefix left_/right_). Geometry is from the 1080 session's photogrammetry
(±15%), world frame: +x forward toward the toys and the camera, +y robot-left, +z up.
"""
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from .calib import GEOMETRY_FILE, MOTORS, SIDES, Calibration

HERE = Path(__file__).resolve().parent
ARM_XML = HERE / "menagerie_so101" / "so101.xml"
PARAMS_FILE = HERE / "params.json"

BASE_Y = 0.175  # bases 35 cm apart center-to-center
ARM_RGBA = {"left": [1.0, 0.45, 0.1, 1.0], "right": [0.25, 0.6, 0.9, 1.0]}  # orange / blue
MENAGERIE_YELLOW = (1.0, 0.82, 0.12)
# Props sit in the center strip between the arms, clear of both resting grippers
# (rest-pose gripper tips land at x≈0.21, y≈±0.17).
BASKET = {"pos": (0.20, 0.0), "radius": 0.08, "height": 0.12}
TOYS = {
    "chick": {"pos": (0.32, -0.09), "rgba": [1.0, 0.85, 0.25, 1.0]},
    "monkey": {"pos": (0.33, 0.06), "rgba": [0.55, 0.36, 0.22, 1.0]},
}
# name: (pos, look_at, fovy_deg). overhead = the front-elevated C922 (hfov ~70° at 16:9).
FIXED_CAMERAS = {
    # fit to ~/sim_ref/overhead.jpg (rest pose): jaw tips + wrist cams of both arms, ~3% width residual
    "overhead": ((0.44, 0.0, 0.49), (0.13, 0.0, 0.0), 43.3),
    "scene": ((0.55, -0.60, 0.50), (0.12, 0.0, 0.08), 50.0),
}
CAMERA_NAMES = {"overhead": "overhead", "left_wrist": "left_wrist_cam",
                "right_wrist": "right_wrist_cam", "scene": "scene"}
CAMERA_SIZES = {"overhead": (720, 1280), "left_wrist": (480, 640), "right_wrist": (480, 640), "scene": (540, 960)}

_G = mujoco.mjtGeom


def _zquat(angle: float) -> list[float]:
    return [math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]


def _look_at(cam, pos, target):
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    cam.alt.xyaxes = [*right, *up]


def _add_arm(world: mujoco.MjSpec, side: str, y: float, calib: Calibration):
    arm = mujoco.MjSpec.from_file(str(ARM_XML))
    for mat in arm.materials:
        if np.allclose(mat.rgba[:3], MENAGERIE_YELLOW, atol=1e-3):
            mat.rgba = ARM_RGBA[side]
    # widen limits so every calibrated value is reachable (real ranges are a hair wider)
    for motor in MOTORS:
        n_lo, n_hi = (0, 100) if motor == "gripper" else (-100, 100)
        key = f"{side}_{motor}.pos"
        lo, hi = sorted((calib.to_rad(key, n_lo), calib.to_rad(key, n_hi)))
        j = arm.joint(motor)
        j.range = [min(j.range[0], lo), max(j.range[1], hi)]
        a = arm.actuator(motor)
        a.ctrlrange = [min(a.ctrlrange[0], lo), max(a.ctrlrange[1], hi)]
    frame = world.worldbody.add_frame(pos=[0.0, y, 0.0])
    frame.attach_body(arm.body("base"), f"{side}_", "")


def _add_props(wb):
    white = [0.92, 0.9, 0.84, 1.0]
    bx, by = BASKET["pos"]
    r, h = BASKET["radius"], BASKET["height"]
    basket = wb.add_body(name="basket", pos=[bx, by, 0.0])
    basket.add_geom(type=_G.mjGEOM_CYLINDER, size=[r, 0.004, 0], pos=[0, 0, 0.004], rgba=white)
    n = 16
    for k in range(n):
        ang = 2 * math.pi * k / n
        basket.add_geom(type=_G.mjGEOM_BOX, size=[0.006, r * math.sin(math.pi / n) + 0.004, h / 2],
                        pos=[r * math.cos(ang), r * math.sin(ang), h / 2], quat=_zquat(ang), rgba=white)
    for name, toy in TOYS.items():
        x, y = toy["pos"]
        body = wb.add_body(name=name, pos=[x, y, 0.04])
        body.add_freejoint(name=f"{name}_free")
        soft = dict(condim=4, friction=[1.0, 0.01, 0.001], solref=[0.02, 1.0], rgba=toy["rgba"])
        body.add_geom(type=_G.mjGEOM_ELLIPSOID, size=[0.035, 0.03, 0.03], mass=0.02, **soft)
        body.add_geom(type=_G.mjGEOM_SPHERE, size=[0.025, 0, 0], pos=[0, 0, 0.045], mass=0.01, **soft)


def build_spec(calib: Calibration | None = None, props: bool = True) -> mujoco.MjSpec:
    calib = calib or Calibration.load()
    world = mujoco.MjSpec()
    opt = world.option
    opt.timestep = 0.005
    opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    opt.impratio = 10
    opt.iterations, opt.ls_iterations = 10, 20  # Menagerie so101 solver settings
    world.visual.global_.offwidth, world.visual.global_.offheight = 1280, 720
    wb = world.worldbody
    wb.add_geom(name="floor", type=_G.mjGEOM_PLANE, size=[3, 3, 0.1], pos=[0, 0, -0.75],
                rgba=[0.32, 0.32, 0.34, 1], contype=0, conaffinity=0)
    wb.add_geom(name="table", type=_G.mjGEOM_BOX, size=[0.45, 0.70, 0.02], pos=[0.33, 0, -0.02],
                rgba=[0.06, 0.06, 0.06, 1])
    for i, y in enumerate((-0.28, 0.28)):
        wb.add_geom(name=f"mat{i}", type=_G.mjGEOM_BOX, size=[0.20, 0.275, 0.001], pos=[0.22, y, 0.001],
                    rgba=[0.2, 0.2, 0.22, 1], contype=0, conaffinity=0)
    wb.add_light(name="top", pos=[0.3, 0, 1.5], dir=[0, 0, -1], diffuse=[0.7, 0.7, 0.7])
    for side, y in (("left", BASE_Y), ("right", -BASE_Y)):
        _add_arm(world, side, y, calib)
    if props:
        _add_props(wb)
    for name, (pos, target, fovy) in FIXED_CAMERAS.items():
        _look_at(wb.add_camera(name=name, pos=list(pos), fovy=fovy), pos, target)
    return world


def load_params(path: Path = PARAMS_FILE) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def apply_params(model: mujoco.MjModel, params: dict) -> None:
    """params = {"motors": {motor: {kp, kv, damping, frictionloss, armature, forcerange}}}, both arms."""
    for motor, p in (params or {}).get("motors", {}).items():
        for side in SIDES:
            a = model.actuator(f"{side}_{motor}")
            dof = model.joint(f"{side}_{motor}").dofadr[0]
            if "kp" in p:
                a.gainprm[0] = p["kp"]
                a.biasprm[1] = -p["kp"]
            if "kv" in p:
                a.biasprm[2] = -p["kv"]
            if "forcerange" in p:
                a.forcerange[:] = [-p["forcerange"], p["forcerange"]]
            for field, arr in (("damping", model.dof_damping), ("frictionloss", model.dof_frictionloss),
                               ("armature", model.dof_armature)):
                if field in p:
                    arr[dof] = p[field]


def load_geometry(path: Path = GEOMETRY_FILE) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _stretch(body, dx: float):
    """Move a body along its offset from the parent: lengthens that link by dx."""
    v = np.array(body.pos)
    body.pos[:] = v + dx * v / np.linalg.norm(v)


def apply_geometry(model: mujoco.MjModel, geometry: dict) -> None:
    """Kinematic-calibration deltas (kincal.py), both arms. Call once on a freshly compiled model.

    geometry = {"lengths_m": {upper_arm_dx, forearm_dx, claw_dx, base_dz}, "base_pitch_deg": float,
                "overhead_camera": {pos, quat}}:
    the upper arm and forearm grow along their links, the claws slide out along the wrist-roll axis
    (gripper -z), the bases rise off the table, and the overhead camera takes the fitted real pose.
    """
    g = (geometry or {}).get("lengths_m", {})
    for side in SIDES:
        _stretch(model.body(f"{side}_lower_arm"), g.get("upper_arm_dx", 0.0))
        _stretch(model.body(f"{side}_wrist"), g.get("forearm_dx", 0.0))
        grip = model.body(f"{side}_gripper")
        axis = np.zeros(3)
        mujoco.mju_rotVecQuat(axis, np.array([0.0, 0.0, -1.0]), grip.quat)
        grip.pos[:] = grip.pos + g.get("claw_dx", 0.0) * axis
        base = model.body(f"{side}_base")
        base.pos[2] += g.get("base_dz", 0.0)
        pitch = math.radians((geometry or {}).get("base_pitch_deg", 0.0))
        if pitch:  # mount tilted forward (+) about the base's y axis: reach-proportional drop
            dq, q = np.zeros(4), np.zeros(4)
            mujoco.mju_axisAngle2Quat(dq, np.array([0.0, 1.0, 0.0]), pitch)
            mujoco.mju_mulQuat(q, np.array(base.quat), dq)
            base.quat[:] = q
    cam = (geometry or {}).get("overhead_camera")
    if cam:  # the real C922 pose, fit jointly with the arm geometry to the lit photos
        cid = model.camera("overhead").id
        model.cam_pos[cid] = cam["pos"]
        model.cam_quat[cid] = cam["quat"]


def build_model(params: dict | None = None, calib: Calibration | None = None, props: bool = True,
                geometry: dict | None = None) -> mujoco.MjModel:
    model = build_spec(calib, props).compile()
    apply_params(model, load_params() if params is None else params)
    apply_geometry(model, load_geometry() if geometry is None else geometry)
    return model
