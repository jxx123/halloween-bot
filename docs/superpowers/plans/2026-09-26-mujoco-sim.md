# MuJoCo Sim Twin + π₀-FAST Policy Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put a MuJoCo twin of the bimanual SO101 rig behind the robot server's HTTP API on the 5090. π₀-FAST drives it through a shared policy runner, the Claude agent can move it or call π, and the web app shows it.

**Architecture:**
- `halloween_bot/sim/`:
  - composes two Menagerie SO101 arms and a table scene with `MjSpec`;
  - maps lerobot's normalized joint units to radians through the real calibration files;
  - runs physics and EGL rendering in threads (`SimEngine`);
  - serves the same `/state /move /cam /stream` API as `server.py`.
- `halloween_bot/policy_runner.py` speaks lerobot 0.6.1's async gRPC protocol to the `~/pi-serve` policy server. It is robot-agnostic, so the 1080's `server.py` can mount it too.
- Sys-ID fits per-motor actuator parameters to real commanded-vs-present traces.

**Tech Stack:** Python 3.12, MuJoCo 3.14 (MjSpec, EGL), lerobot 0.6.1 (async_inference/transport), grpc, scipy, numpy, OpenCV (headless), pytest; vanilla JS for the dashboard.

**Spec:** `docs/superpowers/specs/2026-09-25-mujoco-sim-design.md`

## Global Constraints

- Joint units: body joints `-100..100`, gripper `0..100`, keys `{left|right}_{shoulder_pan|shoulder_lift|elbow_flex|wrist_flex|wrist_roll|gripper}.pos`. The order is always left then right, motors in that order. This is lerobot's `bi_so_follower` action order and the checkpoint's action order.
- The sim server listens on **:8399** with the same JSON shapes as `server.py`. `/state` adds `"sim": true`.
- Command semantics match the real robot:
  - `/move` interpolates at **30 Hz**, duration clamped to **0.5..10 s**;
  - every send clamps the goal to present ± **20.0** units (`max_relative_target`);
  - values are bounded to the key's range.
- Rest pose (`PARK`, both arms): pan −5, lift −86, elbow 95, wrist_flex 45, wrist_roll −10, gripper 40.
- π₀-FAST camera renames: `overhead→base_0_rgb`, `left_wrist→left_wrist_0_rgb`, `right_wrist→right_wrist_0_rgb`. Checkpoint `delvingdeep/pi0fast-so101-bimanual`, `policy_type="pi0_fast"`, 50-action chunks at 30 fps. Default task: `"Grasp the toy and place it in the basket."`
- The sim uses its **own** policy server on **127.0.0.1:8081**, started from `~/pi-serve/.venv` (which carries the pi0_fast shims). :8080 belongs to the 1080's real-robot client.
- **Never** edit or reinstall `~/pi-serve` site-packages.
- lerobot is pinned `==0.6.1` in the `sim` extra, because observations and actions are pickled across the gRPC link.
- `MUJOCO_GL=egl` is set before `import mujoco` in every sim entry point and in the tests.
- This session never edits `server.py` and never commands the real arms.
- Commit to branch `sim`, ending each message with the session's Co-Authored-By and Claude-Session lines.

## Review Focus

1. **Policy server down or busy on :8081.** `ctl.py policy …` must fail within seconds with "policy server unreachable at 127.0.0.1:8081". It must not hang or leave a half-started run. Test: `test_start_fails_fast_when_unreachable` (Task 4).
2. **Stopping a lockstep run while physics is paused.** It must resume physics, so the sim is never left frozen. Test: `test_stop_during_lockstep_resumes_physics` (Task 4).
3. **`/move` while a policy is running.** Must return 409 with a clear message, the way teleop does on the real server. A second `/policy` start must return 409. Test: `test_move_rejected_while_policy_runs` (Task 5).
4. **Unknown joint keys and out-of-range values.** Unknown keys return the real server's `{"error", "valid_keys"}` shape. Out-of-range values are bounded. Test: `test_move_unknown_key_and_bounds` (Task 5).
5. **Policy server restarted between runs, so the cached instructions are stale.** The runner must notice there are no actions and resend the instructions once, instead of waiting forever. Test: `test_reinstructs_when_no_chunk_arrives` (Task 4).

---

## File Structure

| Path | Responsibility |
|---|---|
| `halloween_bot/sim/menagerie_so101/` | Vendored Menagerie `robotstudio_so101` (so101.xml, assets/, LICENSE). Done. |
| `halloween_bot/sim/calibration/bimanual_{left,right}.json` | Copies of the 1080's so_follower calibration. Done. |
| `halloween_bot/sim/calib.py` | Normalized ↔ radian affine maps per key. |
| `halloween_bot/sim/model.py` | `build_model()`: arms, table, mats, basket, toys, cameras. `apply_params()`: sys-ID parameters. |
| `halloween_bot/sim/engine.py` | `SimEngine`: joint I/O, real-server command semantics, physics and render threads, pause/resume, reset. |
| `halloween_bot/policy_runner.py` | Shared gRPC π runner plus the `handle_http` mount helper. |
| `halloween_bot/sim/server.py` | HTTP server with API parity plus `/sim/reset`, `/trajectory` and `/policy*`. |
| `halloween_bot/sim/sysid.py` | Trace loading (lerobot dataset or trajectory JSON), replay, fit, report. |
| `halloween_bot/sim/sysid_collect.py` | Generates real-robot sequences. Runs on the 1080 against its server's `/trajectory`. |
| `halloween_bot/sim/params.json` | Sys-ID output, loaded by `build_model()`. |
| `halloween_bot/ctl.py` | Adds `start --sim`, `policy`, `policy-stop`, `sim-reset`, and more `cam` names. |
| `halloween_bot/app/webapp.py`, `app/static/dashboard.html` | Sim-aware dashboard and `/api/policy*` / `/api/sim/reset` proxies. |
| `CLAUDE.md` | Machine-agnostic commands, "Two ways to act", "Sim mode". |
| `run_sim.sh` | Brings up the policy server (:8081), sim server (:8399) and webapp (:8500). |
| `tests/` | pytest suite (`conftest.py` sets `MUJOCO_GL`). |

---

### Task 1: Calibration mapping

**Files:**
- Create: `halloween_bot/sim/__init__.py`, `halloween_bot/sim/calib.py`, `tests/conftest.py`, `tests/test_calib.py`

**Interfaces:**
- Produces:
  - `SIDES`, `MOTORS`, `KEYS: list[str]`
  - `split_key(key) -> (side, motor)`
  - `joint_name(key) -> str` (e.g. `"left_gripper"`)
  - `clamp_norm(key, v) -> float`
  - `Calibration.load() -> Calibration`
  - `Calibration.affine(key) -> (a, b)` with `rad = a*n + b`
  - `.to_rad(key, n) -> float`, `.to_norm(key, rad) -> float`

- [ ] **Step 1: Write the failing tests**

```python
# tests/conftest.py
import os
os.environ.setdefault("MUJOCO_GL", "egl")
```

```python
# tests/test_calib.py
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
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_calib.py -q`. Expected: ImportError (no module `halloween_bot.sim.calib`).

- [ ] **Step 3: Implement**

```python
# halloween_bot/sim/__init__.py
"""MuJoCo twin of the bimanual SO101 rig (see docs/superpowers/specs/2026-09-25-mujoco-sim-design.md)."""
```

```python
# halloween_bot/sim/calib.py
"""lerobot normalized joint units <-> MuJoCo joint radians for the bimanual SO101.

lerobot (use_degrees=False) reports body joints as -100..100 across each motor's calibrated
[range_min, range_max] ticks and the gripper as 0..100. The Menagerie model (TheRobotStudio
so101_new_calib) zeroes every body joint at the middle of its range, which is the same zero
lerobot's DEGREES mode uses (mid = (min+max)/2) and the one lerobot's SO101 kinematics feed
straight into so101_new_calib. So a body joint is linear with no offset:
    rad = n/100 * (max-min)/2 ticks * 2π/4095
The gripper's 0..100 (closed..open) maps linearly onto the model's jaw range.
"""
import json
import math
from pathlib import Path

CALIB_DIR = Path(__file__).resolve().parent / "calibration"
SIDES = ("left", "right")
MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
KEYS = [f"{s}_{m}.pos" for s in SIDES for m in MOTORS]  # lerobot bi_so_follower action order
TICKS_PER_RAD = 4095 / (2 * math.pi)
GRIPPER_RANGE = (-0.174533, 1.7453292)  # Menagerie jaw joint range: closed, open


def split_key(key: str) -> tuple[str, str]:
    side, motor = key.removesuffix(".pos").split("_", 1)
    return side, motor


def joint_name(key: str) -> str:
    return key.removesuffix(".pos")


def clamp_norm(key: str, value: float) -> float:
    lo, hi = (0.0, 100.0) if key.endswith("gripper.pos") else (-100.0, 100.0)
    return max(lo, min(hi, float(value)))


class Calibration:
    def __init__(self, calib: dict[str, dict], gripper_range: tuple[float, float] = GRIPPER_RANGE):
        for side in SIDES:
            for motor in MOTORS:
                if calib[side][motor].get("drive_mode", 0):
                    raise ValueError(f"{side}/{motor}: drive_mode=1 calibrations are not supported")
        self.calib = calib
        self._affine = {}
        for key in KEYS:
            side, motor = split_key(key)
            if motor == "gripper":
                lo, hi = gripper_range
                self._affine[key] = ((hi - lo) / 100.0, lo)
            else:
                c = calib[side][motor]
                half_span_rad = (c["range_max"] - c["range_min"]) / 2 / TICKS_PER_RAD
                self._affine[key] = (half_span_rad / 100.0, 0.0)

    @classmethod
    def load(cls, directory: Path = CALIB_DIR) -> "Calibration":
        return cls({s: json.loads((directory / f"bimanual_{s}.json").read_text()) for s in SIDES})

    def affine(self, key: str) -> tuple[float, float]:
        return self._affine[key]

    def to_rad(self, key: str, n: float) -> float:
        a, b = self._affine[key]
        return a * n + b

    def to_norm(self, key: str, rad: float) -> float:
        a, b = self._affine[key]
        return (rad - b) / a
```

- [ ] **Step 4: Run tests.** `.venv/bin/python -m pytest tests/test_calib.py -q` → all pass.
- [ ] **Step 5: Commit** the Menagerie assets, the calibration JSONs, the `pyproject.toml` `sim` extra and headless-OpenCV fix, `calib.py` and the tests.

### Task 2: Scene model

**Files:**
- Create: `halloween_bot/sim/model.py`, `tests/test_model.py`

**Interfaces:**
- Consumes: `Calibration`, `KEYS`, `MOTORS`, `SIDES`, `joint_name`.
- Produces:
  - `build_model(params: dict | None = None, calib: Calibration | None = None, props: bool = True) -> mujoco.MjModel`
  - `apply_params(model, params) -> None`
  - `load_params(path=PARAMS_FILE) -> dict`
  - `CAMERA_NAMES: dict[str, str]` (API name → MJCF camera)
  - `CAMERA_SIZES: dict[str, (h, w)]`
  - `TOYS: dict[str, dict]`
  - `PARAMS_FILE`
- `params` shape: `{"motors": {motor: {"kp", "kv", "damping", "frictionloss", "armature", "forcerange"}}}`. Any subset of fields is allowed, and each value applies to both arms.

- [ ] **Step 1: Failing tests**

```python
# tests/test_model.py
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
    cal = Calibration.load(); m = build_model(params={}, calib=cal)
    for k in KEYS:
        lo, hi = m.joint(joint_name(k)).range
        for n in ((0, 100) if k.endswith("gripper.pos") else (-100, 100)):
            assert lo - 1e-6 <= cal.to_rad(k, n) <= hi + 1e-6, k

def test_apply_params_sets_gain_bias_damping():
    m = build_model(params={})
    apply_params(m, {"motors": {"elbow_flex": {"kp": 12.0, "kv": 0.5, "damping": 0.3,
                                               "frictionloss": 0.04, "armature": 0.02, "forcerange": 2.0}}})
    for side in ("left", "right"):
        a = m.actuator(f"{side}_elbow_flex"); j = m.joint(f"{side}_elbow_flex")
        assert a.gainprm[0] == 12.0 and a.biasprm[1] == -12.0 and a.biasprm[2] == -0.5
        assert np.allclose(a.forcerange, [-2.0, 2.0])
        assert m.dof_damping[j.dofadr[0]] == 0.3 and m.dof_armature[j.dofadr[0]] == 0.02

def test_no_props_model_has_no_toys():
    m = build_model(params={}, props=False)
    assert all(m.joint(i).type != mujoco.mjtJoint.mjJNT_FREE for i in range(m.njnt))
```

- [ ] **Step 2: Run to verify failure** (ImportError).
- [ ] **Step 3: Implement**

```python
# halloween_bot/sim/model.py
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

from .calib import KEYS, MOTORS, SIDES, Calibration, joint_name

HERE = Path(__file__).resolve().parent
ARM_XML = HERE / "menagerie_so101" / "so101.xml"
PARAMS_FILE = HERE / "params.json"

BASE_Y = 0.175  # bases 35 cm apart center-to-center
ARM_RGBA = {"left": [1.0, 0.45, 0.1, 1.0], "right": [0.25, 0.6, 0.9, 1.0]}  # orange / blue
MENAGERIE_YELLOW = (1.0, 0.82, 0.12)
BASKET = {"pos": (0.20, 0.10), "radius": 0.10, "height": 0.12}
TOYS = {
    "chick": {"pos": (0.33, -0.10), "rgba": [1.0, 0.85, 0.25, 1.0]},
    "monkey": {"pos": (0.36, 0.00), "rgba": [0.55, 0.36, 0.22, 1.0]},
}
# name: (pos, look_at, fovy_deg). overhead = the front-elevated C922 (hfov ~70° at 16:9).
FIXED_CAMERAS = {
    "overhead": ((0.62, 0.0, 0.45), (0.08, 0.0, 0.05), 43.3),
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
    right = np.cross(f, [0.0, 0.0, 1.0]); right /= np.linalg.norm(right)
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
        cal_key = f"{side}_{motor}.pos"
        lo, hi = sorted((calib.to_rad(cal_key, n_lo), calib.to_rad(cal_key, n_hi)))
        j = arm.joint(motor)
        j.range = [min(j.range[0], lo), max(j.range[1], hi)]
        a = arm.actuator(motor)
        a.ctrlrange = [min(a.ctrlrange[0], lo), max(a.ctrlrange[1], hi)]
    frame = world.worldbody.add_frame(pos=[0.0, y, 0.0])
    frame.attach_body(arm.body("base"), f"{side}_", "")


def _add_props(wb):
    white = [0.92, 0.9, 0.84, 1.0]
    bx, by = BASKET["pos"]; r, h = BASKET["radius"], BASKET["height"]
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
    for motor, p in (params or {}).get("motors", {}).items():
        for side in SIDES:
            a = model.actuator(f"{side}_{motor}")
            dof = model.joint(f"{side}_{motor}").dofadr[0]
            if "kp" in p:
                a.gainprm[0] = p["kp"]; a.biasprm[1] = -p["kp"]
            if "kv" in p:
                a.biasprm[2] = -p["kv"]
            if "forcerange" in p:
                a.forcerange[:] = [-p["forcerange"], p["forcerange"]]
            for field, arr in (("damping", model.dof_damping), ("frictionloss", model.dof_frictionloss),
                               ("armature", model.dof_armature)):
                if field in p:
                    arr[dof] = p[field]


def build_model(params: dict | None = None, calib: Calibration | None = None, props: bool = True) -> mujoco.MjModel:
    model = build_spec(calib, props).compile()
    apply_params(model, load_params() if params is None else params)
    return model
```

- [ ] **Step 4: Run tests** → pass. Also render the rest pose and check it against `~/sim_ref/overhead.jpg`: blue on image-left, grippers toward the camera.
- [ ] **Step 5: Commit.**

### Task 3: SimEngine

**Files:**
- Create: `halloween_bot/sim/engine.py`, `tests/test_engine.py`

**Interfaces:**
- Consumes: Tasks 1 and 2.
- Produces a `SimEngine(model=None, calib=None, render_hz=15.0)` class with:
  - `.read_positions() -> dict[str, float]`
  - `.send_action(action: dict) -> dict` (raises `KeyError` on unknown keys)
  - `.move(targets: dict, duration: float) -> dict`, the real `do_move` JSON
  - `.run_trajectory(points: list[dict], hz=30.0) -> {"ok", "hz", "trace": [{"t", "present", "cmd"}]}`
  - `.step(seconds)`
  - `.reset(randomize=False, seed=None)`
  - `.start()`, `.stop()`, `.pause()`, `.resume()`, `.paused: bool`
  - `.get_frame(name) -> np.ndarray` (RGB uint8; raises `KeyError` for unknown names)
- Also produces the module constants `MAX_RELATIVE_TARGET=20.0`, `CONTROL_HZ=30`, `REST: dict`.

- [ ] **Step 1: Failing tests**

```python
# tests/test_engine.py
import numpy as np
import pytest
from halloween_bot.sim.calib import KEYS
from halloween_bot.sim.engine import MAX_RELATIVE_TARGET, REST, SimEngine

@pytest.fixture(scope="module")
def eng():
    return SimEngine()

def test_reset_is_rest_pose(eng):
    eng.reset()
    pos = eng.read_positions()
    assert list(pos) == KEYS
    for k in KEYS:
        assert pos[k] == pytest.approx(REST[k], abs=0.5)

def test_rest_pose_is_stable(eng):
    eng.reset(); eng.step(2.0)
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
        e.pause(); assert e.paused
        t = e.data.time; import time; time.sleep(0.3); assert e.data.time == t
        e.resume(); time.sleep(0.3); assert e.data.time > t
    finally:
        e.stop()
    with pytest.raises(KeyError):
        e.get_frame("nope")
```

- [ ] **Step 2: Run to verify failure** (ImportError).
- [ ] **Step 3: Implement**

```python
# halloween_bot/sim/engine.py
"""SimEngine: the MuJoCo twin behind the sim server.

Joint I/O is in lerobot's normalized units, and commands follow the real server's semantics:
  send_action clamps each goal to present ± MAX_RELATIVE_TARGET (lerobot max_relative_target),
  move() is server.py's do_move (30 Hz linear interpolation).
start() runs physics paced to wall clock plus a render thread that owns the EGL context and
renders only the cameras someone asked for in the last few seconds. Without start(), the
engine is headless: move/run_trajectory advance physics directly (tests, sys-ID).
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import threading
import time

import mujoco
import numpy as np

from .calib import KEYS, SIDES, Calibration, clamp_norm, joint_name
from .model import CAMERA_NAMES, CAMERA_SIZES, TOYS, build_model

MAX_RELATIVE_TARGET = 20.0
CONTROL_HZ = 30
PARK = {"shoulder_pan": -5.0, "shoulder_lift": -86.0, "elbow_flex": 95.0, "wrist_flex": 45.0,
        "wrist_roll": -10.0, "gripper": 40.0}
REST = {f"{s}_{m}.pos": v for s in SIDES for m, v in PARK.items()}
FRAME_MAX_AGE = 0.5  # s; older frames are re-rendered before being served
WANT_TTL = 3.0  # s a camera keeps being rendered after its last request


class SimEngine:
    def __init__(self, model: mujoco.MjModel | None = None, calib: Calibration | None = None,
                 render_hz: float = 15.0):
        self.calib = calib or Calibration.load()
        self.model = model or build_model(calib=self.calib)
        self.data = mujoco.MjData(self.model)
        self.lock = threading.RLock()
        self.render_hz = render_hz
        self._qadr = np.array([self.model.joint(joint_name(k)).qposadr[0] for k in KEYS])
        self._act = np.array([self.model.actuator(joint_name(k)).id for k in KEYS])
        aff = [self.calib.affine(k) for k in KEYS]
        self._a = np.array([a for a, _ in aff]); self._b = np.array([b for _, b in aff])
        self._index = {k: i for i, k in enumerate(KEYS)}
        self._running = threading.Event()
        self._paused = threading.Event()
        self._resync = threading.Event()
        self._threads: list[threading.Thread] = []
        self._frames: dict[str, tuple[float, np.ndarray]] = {}
        self._frame_cv = threading.Condition()
        self._wanted: dict[str, float] = {}
        self._headless_renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self.reset()

    # ---- joint I/O -------------------------------------------------------------------
    def _present_array(self) -> np.ndarray:
        return (self.data.qpos[self._qadr] - self._b) / self._a

    def read_positions(self) -> dict[str, float]:
        with self.lock:
            return dict(zip(KEYS, self._present_array().tolist()))

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        unknown = [k for k in action if k not in self._index]
        if unknown:
            raise KeyError(f"unknown joint keys: {unknown}")
        sent = {}
        with self.lock:
            present = self._present_array()
            for k, v in action.items():
                i = self._index[k]
                v = clamp_norm(k, v)
                v = min(max(v, present[i] - MAX_RELATIVE_TARGET), present[i] + MAX_RELATIVE_TARGET)
                self.data.ctrl[self._act[i]] = self._a[i] * v + self._b[i]
                sent[k] = v
        return sent

    def _sleep(self, seconds: float):
        if self._running.is_set():
            time.sleep(seconds)
        else:
            self.step(seconds)

    def _now(self) -> float:
        return self.data.time if not self._running.is_set() else time.perf_counter()

    def move(self, targets: dict[str, float], duration: float) -> dict:
        unknown = [k for k in targets if k not in self._index]
        if unknown:
            return {"error": f"unknown joint keys: {unknown}", "valid_keys": sorted(KEYS)}
        duration = max(0.5, min(10.0, float(duration)))
        start = self.read_positions()
        goal = {**start, **{k: clamp_norm(k, v) for k, v in targets.items()}}
        steps = max(1, int(duration * CONTROL_HZ))
        for i in range(1, steps + 1):
            a = i / steps
            self.send_action({k: start[k] + (goal[k] - start[k]) * a for k in goal})
            self._sleep(1.0 / CONTROL_HZ)
        final = self.read_positions()
        return {"ok": True, "requested": {k: goal[k] for k in targets}, "reached": {k: final[k] for k in targets}}

    def run_trajectory(self, points: list[dict[str, float]], hz: float = 30.0) -> dict:
        unknown = sorted({k for p in points for k in p if k not in self._index})
        if unknown:
            return {"error": f"unknown joint keys: {unknown}", "valid_keys": sorted(KEYS)}
        hz = max(1.0, min(100.0, float(hz)))
        hold = self.read_positions()
        trace = []
        t0 = self._now()
        for p in points:
            present = self.read_positions()
            cmd = {**hold, **{k: clamp_norm(k, v) for k, v in p.items()}}
            hold = cmd
            self.send_action(cmd)
            trace.append({"t": self._now() - t0, "present": present, "cmd": cmd})
            self._sleep(1.0 / hz)
        trace.append({"t": self._now() - t0, "present": self.read_positions(), "cmd": None})
        return {"ok": True, "hz": hz, "trace": trace}

    # ---- physics ---------------------------------------------------------------------
    def step(self, seconds: float):
        n = max(1, round(seconds / self.model.opt.timestep))
        with self.lock:
            mujoco.mj_step(self.model, self.data, nstep=n)

    def reset(self, randomize: bool = False, seed: int | None = None):
        rng = np.random.default_rng(seed)
        with self.lock:
            mujoco.mj_resetData(self.model, self.data)
            rest = np.array([REST[k] for k in KEYS])
            self.data.qpos[self._qadr] = self._a * rest + self._b
            self.data.ctrl[self._act] = self._a * rest + self._b
            for name, toy in TOYS.items():
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_free")
                if jid < 0:
                    continue  # props=False model
                x, y = toy["pos"]
                yaw = 0.0
                if randomize:
                    x += rng.uniform(-0.04, 0.04); y += rng.uniform(-0.05, 0.05); yaw = rng.uniform(-np.pi, np.pi)
                adr = self.model.jnt_qposadr[jid]
                self.data.qpos[adr:adr + 7] = [x, y, 0.04, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            mujoco.mj_forward(self.model, self.data)
        self._resync.set()

    def start(self):
        if self._running.is_set():
            return
        self._running.set()
        self._threads = [threading.Thread(target=self._physics_loop, name="sim-physics", daemon=True),
                         threading.Thread(target=self._render_loop, name="sim-render", daemon=True)]
        for t in self._threads:
            t.start()

    def stop(self):
        self._running.clear()
        for t in self._threads:
            t.join(timeout=3)
        self._threads = []

    def pause(self):
        self._paused.set()

    def resume(self):
        self._resync.set()
        self._paused.clear()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def _physics_loop(self):
        dt = self.model.opt.timestep
        sim0, wall0 = self.data.time, time.perf_counter()
        while self._running.is_set():
            if self._resync.is_set():
                self._resync.clear()
                sim0, wall0 = self.data.time, time.perf_counter()
            if self._paused.is_set():
                time.sleep(0.005)
                continue
            behind = (time.perf_counter() - wall0) - (self.data.time - sim0)
            n = int(behind / dt)
            if n <= 0:
                time.sleep(dt / 2)
                continue
            with self.lock:
                mujoco.mj_step(self.model, self.data, nstep=min(n, 50))
            if n > 50:  # fell >0.25 s behind: drop the debt instead of spiralling
                self._resync.set()

    # ---- cameras ---------------------------------------------------------------------
    def _render(self, renderer: mujoco.Renderer, name: str) -> np.ndarray:
        with self.lock:
            renderer.update_scene(self.data, CAMERA_NAMES[name])
        return renderer.render().copy()

    def _render_loop(self):
        renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        period = 1.0 / self.render_hz
        while self._running.is_set():
            t = time.perf_counter()
            for name in [n for n, ts in list(self._wanted.items()) if t - ts < WANT_TTL]:
                size = CAMERA_SIZES[name]
                if size not in renderers:
                    renderers[size] = mujoco.Renderer(self.model, *size)
                frame = self._render(renderers[size], name)
                with self._frame_cv:
                    self._frames[name] = (time.perf_counter(), frame)
                    self._frame_cv.notify_all()
            time.sleep(max(0.0, period - (time.perf_counter() - t)))
        for r in renderers.values():
            r.close()

    def get_frame(self, name: str, timeout: float = 5.0) -> np.ndarray:
        if name not in CAMERA_NAMES:
            raise KeyError(name)
        if not self._running.is_set():  # headless: render in the caller's thread
            size = CAMERA_SIZES[name]
            if size not in self._headless_renderers:
                self._headless_renderers[size] = mujoco.Renderer(self.model, *size)
            return self._render(self._headless_renderers[size], name)
        self._wanted[name] = time.perf_counter()
        deadline = time.perf_counter() + timeout
        with self._frame_cv:
            while True:
                ts_frame = self._frames.get(name)
                if ts_frame and time.perf_counter() - ts_frame[0] < FRAME_MAX_AGE:
                    return ts_frame[1]
                left = deadline - time.perf_counter()
                if left <= 0:
                    raise TimeoutError(f"camera '{name}' produced no frame within {timeout}s")
                self._frame_cv.wait(left)
```

- [ ] **Step 4: Run tests** → pass. If `test_rest_pose_is_stable` fails because folded links interpenetrate, look at the contact pairs (`data.contact`) and add `<exclude>` pairs between that arm's bodies with `world.add_exclude(bodyname1=…, bodyname2=…)` in `_add_arm`. Don't loosen the test.
- [ ] **Step 5: Commit.**

### Task 4: Shared policy runner

**Files:**
- Create: `halloween_bot/policy_runner.py`, `tests/test_policy_runner.py`

**Interfaces:**
- Consumes: `lerobot.async_inference.helpers.{RemotePolicyConfig, TimedAction, TimedObservation}`, `lerobot.transport.{services_pb2, services_pb2_grpc}`, `lerobot.transport.utils.send_bytes_in_chunks`, `lerobot.utils.feature_utils.hw_to_dataset_features`.
- Produces:
  - `PolicyRunner(get_observation, send_action, action_keys, observation_features, *, server_address="127.0.0.1:8080", checkpoint=DEFAULT_CHECKPOINT, policy_type="pi0_fast", device="cuda", fps=30, actions_per_chunk=50, chunk_size_threshold=0.5, obs_min_interval=0.1, pause=None, resume=None, connect_timeout=5.0, load_timeout=300.0, first_chunk_timeout=20.0, stub_factory=None)`
  - Methods: `.start(task=DEFAULT_TASK, seconds=30.0, lockstep=False) -> dict`, `.stop(timeout=5.0) -> dict`, `.status() -> dict`, `.running: bool`
  - Exceptions: `PolicyBusy`, `PolicyUnavailable`
  - `handle_http(runner, method, path, payload) -> tuple[int, dict] | None`
  - Constants: `DEFAULT_TASK`, `DEFAULT_CHECKPOINT`, `PI0FAST_CAMERAS`
- Status keys: `running, phase (idle|connecting|loading|running|reloading|done|stopped|error), task, lockstep, seconds, elapsed_s, executed, chunks, queue, last_latency_s, error, server`.

- [ ] **Step 1: Failing tests** (a fake stub stands in for the gRPC server)

```python
# tests/test_policy_runner.py
import pickle, threading, time
import grpc, numpy as np, pytest, torch
from lerobot.async_inference.helpers import TimedAction
from lerobot.transport import services_pb2
from halloween_bot.policy_runner import PolicyBusy, PolicyRunner, PolicyUnavailable, handle_http

KEYS = [f"k{i}" for i in range(12)]
FEATURES = {**{k: float for k in KEYS}, "base_0_rgb": (8, 8, 3)}

class Unavailable(grpc.RpcError):
    def code(self): return grpc.StatusCode.UNAVAILABLE

class FakeStub:
    """Answers each must_go observation with a chunk of `chunk` actions valued at the obs timestep."""
    def __init__(self, reachable=True, chunk=10, delay=0.05, instructed_ok=True):
        self.reachable, self.chunk, self.delay = reachable, chunk, delay
        self.instructions = 0; self.obs = []; self.pending = []; self.cv = threading.Condition()
        self.serve = instructed_ok
    def Ready(self, req, timeout=None):
        if not self.reachable: raise Unavailable()
        return services_pb2.Empty()
    def SendPolicyInstructions(self, req, timeout=None):
        self.instructions += 1; self.serve = True
        return services_pb2.Empty()
    def SendObservations(self, it, timeout=None):
        data = b"".join(m.data for m in it); obs = pickle.loads(data)
        self.obs.append(obs)
        if obs.must_go and self.serve:
            with self.cv: self.pending.append(obs); self.cv.notify()
        return services_pb2.Empty()
    def GetActions(self, req, timeout=None):
        with self.cv:
            if not self.pending: self.cv.wait(0.2)
            if not self.pending: return services_pb2.Actions(data=b"")
            obs = self.pending.pop(0)
        time.sleep(self.delay)
        t0 = obs.get_timestep()
        acts = [TimedAction(timestamp=time.time(), timestep=t0 + i, action=torch.full((12,), float(t0 + i)))
                for i in range(self.chunk)]
        return services_pb2.Actions(data=pickle.dumps(acts))

class FakeChannel:
    def close(self): pass

def make(stub, **kw):
    sent = []
    r = PolicyRunner(lambda: {**{k: 0.0 for k in KEYS}, "base_0_rgb": np.zeros((8, 8, 3), np.uint8)},
                     sent.append, KEYS, FEATURES, fps=100, stub_factory=lambda addr: (FakeChannel(), stub), **kw)
    return r, sent

def wait_done(r, t=5):
    end = time.time() + t
    while r.running and time.time() < end: time.sleep(0.02)
    assert not r.running

def test_start_fails_fast_when_unreachable():
    r, _ = make(FakeStub(reachable=False))
    t = time.time()
    with pytest.raises(PolicyUnavailable, match="unreachable"):
        r.start("t", 1)
    assert time.time() - t < 2 and not r.running and r.status()["phase"] == "idle"

def test_runs_executes_actions_in_order_and_finishes():
    stub = FakeStub(); r, sent = make(stub)
    r.start("pick", seconds=0.3)
    wait_done(r)
    st = r.status()
    assert st["phase"] == "done" and st["executed"] >= 10 and st["chunks"] >= 2
    vals = [a["k0"] for a in sent]
    assert vals == sorted(vals)  # timesteps executed monotonically
    assert stub.instructions == 1 and stub.obs[0].observation["task"] == "pick"
    assert set(stub.obs[0].observation) >= set(KEYS) | {"base_0_rgb", "task"}

def test_second_start_is_busy_and_stop_works():
    r, _ = make(FakeStub())
    r.start("t", seconds=30)
    with pytest.raises(PolicyBusy):
        r.start("t", 1)
    st = r.stop()
    assert not st["running"] and st["phase"] == "stopped"

def test_instructions_sent_once_across_runs():
    stub = FakeStub(); r, _ = make(stub)
    r.start("a", 0.1); wait_done(r)
    r.start("b", 0.1); wait_done(r)
    assert stub.instructions == 1

def test_reinstructs_when_no_chunk_arrives():
    stub = FakeStub(); r, _ = make(stub, first_chunk_timeout=0.5)
    r.start("a", 0.1); wait_done(r)
    stub.serve = False  # simulates a restarted server that lost the policy
    r.start("b", 0.2); wait_done(r, 10)
    assert stub.instructions == 2 and r.status()["phase"] == "done"

def test_lockstep_pauses_during_inference_and_resumes():
    calls = []
    stub = FakeStub(delay=0.1)
    r, _ = make(stub, pause=lambda: calls.append("pause"), resume=lambda: calls.append("resume"))
    r.start("t", seconds=0.25, lockstep=True)
    wait_done(r)
    assert calls[0] == "pause" and calls.count("pause") == calls.count("resume") >= 2

def test_stop_during_lockstep_resumes_physics():
    calls = []
    stub = FakeStub(); stub.serve = False  # no actions ever -> stuck waiting while paused
    r, _ = make(stub, pause=lambda: calls.append("pause"), resume=lambda: calls.append("resume"),
                first_chunk_timeout=60)
    r._instructed = True  # skip loading so it goes straight to the paused wait
    r.start("t", seconds=5, lockstep=True)
    time.sleep(0.3)
    r.stop()
    assert calls and calls[-1] == "resume"

def test_lockstep_requires_hooks():
    r, _ = make(FakeStub())
    with pytest.raises(ValueError):
        r.start("t", 1, lockstep=True)

def test_handle_http_routes():
    r, _ = make(FakeStub(reachable=False))
    assert handle_http(r, "GET", "/policy", {})[0] == 200
    assert handle_http(r, "POST", "/policy", {"task": "x"})[0] == 503
    assert handle_http(r, "POST", "/policy/stop", {})[0] == 200
    assert handle_http(r, "GET", "/state", {}) is None
```

- [ ] **Step 2: Run to verify failure** (ImportError).
- [ ] **Step 3: Implement**

```python
# halloween_bot/policy_runner.py
"""Shared π-policy runner: streams robot observations to a lerobot async policy_server over gRPC
and executes the returned action chunks. Robot-agnostic; a server mounts it with two callables:

  get_observation() -> {"left_shoulder_pan.pos": float, ..., "base_0_rgb": HxWx3 uint8, ...}
  send_action({"left_shoulder_pan.pos": float, ...})    # the server's own safety caps apply

and routes HTTP through handle_http():  POST /policy {task, seconds, lockstep}
POST /policy/stop   GET /policy.

Protocol = lerobot 0.6.1 robot_client: Ready -> SendPolicyInstructions -> {SendObservations,
GetActions}*, with the same queue rules (latest chunk wins on overlapping timesteps; send an
observation whenever the queue is at/below chunk_size_threshold; must_go when it ran dry).
Two differences, both for an agent-facing tool:
  * instructions (= a ~47 s policy load on the server) are sent once per runner and re-sent only
    if no chunk arrives within first_chunk_timeout (the server was restarted);
  * observations are rate-limited to obs_min_interval (images are MBs; inference is ~1.6 s).
Lockstep (sim only): pause physics whenever the queue runs dry, until the next chunk lands.
"""
import pickle  # nosec - trusted local policy server, same as lerobot's robot_client
import threading
import time
from collections.abc import Callable

import grpc

from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.feature_utils import hw_to_dataset_features

DEFAULT_CHECKPOINT = "delvingdeep/pi0fast-so101-bimanual"
DEFAULT_TASK = "Grasp the toy and place it in the basket."
PI0FAST_CAMERAS = {"overhead": "base_0_rgb", "left_wrist": "left_wrist_0_rgb", "right_wrist": "right_wrist_0_rgb"}


class PolicyBusy(RuntimeError):
    pass


class PolicyUnavailable(RuntimeError):
    pass


def _default_stub_factory(address: str):
    channel = grpc.insecure_channel(address, grpc_channel_options())
    return channel, services_pb2_grpc.AsyncInferenceStub(channel)


class PolicyRunner:
    def __init__(self, get_observation: Callable[[], dict], send_action: Callable[[dict], object],
                 action_keys: list[str], observation_features: dict, *,
                 server_address: str = "127.0.0.1:8080", checkpoint: str = DEFAULT_CHECKPOINT,
                 policy_type: str = "pi0_fast", device: str = "cuda", fps: float = 30.0,
                 actions_per_chunk: int = 50, chunk_size_threshold: float = 0.5,
                 obs_min_interval: float = 0.1, pause: Callable[[], None] | None = None,
                 resume: Callable[[], None] | None = None, connect_timeout: float = 5.0,
                 load_timeout: float = 300.0, first_chunk_timeout: float = 20.0, stub_factory=None):
        self.get_observation, self.send_action = get_observation, send_action
        self.action_keys = list(action_keys)
        self.lerobot_features = hw_to_dataset_features(observation_features, "observation", use_video=False)
        self.server_address, self.checkpoint, self.policy_type, self.device = (
            server_address, checkpoint, policy_type, device)
        self.fps, self.actions_per_chunk = fps, actions_per_chunk
        self.chunk_size_threshold, self.obs_min_interval = chunk_size_threshold, obs_min_interval
        self.pause, self.resume = pause, resume
        self.connect_timeout, self.load_timeout, self.first_chunk_timeout = (
            connect_timeout, load_timeout, first_chunk_timeout)
        self._stub_factory = stub_factory or _default_stub_factory
        self._instructed = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._chunk_arrived = threading.Event()
        self._thread: threading.Thread | None = None
        self._paused_by_us = False
        self._status = {"running": False, "phase": "idle", "task": None, "lockstep": False, "seconds": 0,
                        "elapsed_s": 0.0, "executed": 0, "chunks": 0, "last_latency_s": None, "error": None,
                        "server": server_address}
        self._reset_queue()

    # ---- public API --------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._status["running"]

    def status(self) -> dict:
        with self._lock:
            return {**self._status, "queue": len(self._queue)}

    def start(self, task: str = DEFAULT_TASK, seconds: float = 30.0, lockstep: bool = False) -> dict:
        if self.running:
            raise PolicyBusy("a policy run is already in progress (POST /policy/stop first)")
        if lockstep and (self.pause is None or self.resume is None):
            raise ValueError("lockstep needs pause/resume hooks (sim only)")
        seconds = max(0.1, min(300.0, float(seconds)))
        channel, stub = self._stub_factory(self.server_address)
        try:
            stub.Ready(services_pb2.Empty(), timeout=self.connect_timeout)
        except grpc.RpcError as e:
            channel.close()
            code = e.code().name if hasattr(e, "code") else type(e).__name__
            raise PolicyUnavailable(f"policy server unreachable at {self.server_address} ({code})") from e
        self._stop.clear()
        self._reset_queue()
        self._set(running=True, phase="connecting", task=task, lockstep=lockstep, seconds=seconds,
                  elapsed_s=0.0, executed=0, chunks=0, last_latency_s=None, error=None)
        self._thread = threading.Thread(target=self._run, args=(channel, stub, task, seconds, lockstep),
                                        name="policy-runner", daemon=True)
        self._thread.start()
        return self.status()

    def stop(self, timeout: float = 5.0) -> dict:
        self._stop.set()
        self._chunk_arrived.set()  # wake a lockstep wait
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout)
        return self.status()

    # ---- internals ---------------------------------------------------------------------
    def _set(self, **kw):
        with self._lock:
            self._status.update(kw)

    def _reset_queue(self):
        with self._lock:
            self._queue: dict[int, object] = {}
            self._latest = -1
            self._chunk_size = -1
            self._must_go = True
            self._last_obs_t = 0.0
            self._must_go_sent_t: float | None = None
        self._chunk_arrived.clear()

    def _instruct(self, stub):
        cfg = RemotePolicyConfig(self.policy_type, self.checkpoint, self.lerobot_features,
                                 self.actions_per_chunk, self.device)
        stub.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(cfg)), timeout=self.load_timeout)
        self._instructed = True

    def _run(self, channel, stub, task, seconds, lockstep):
        receiver = None
        try:
            if not self._instructed:
                self._set(phase="loading")
                self._instruct(stub)
            self._set(phase="running")
            receiver = threading.Thread(target=self._receive_loop, args=(stub,), name="policy-rx", daemon=True)
            receiver.start()
            self._control_loop(stub, task, seconds, lockstep)
            self._set(phase="stopped" if self._stop.is_set() else "done")
        except Exception as e:  # surfaced via status(); the robot just holds its last pose
            self._set(phase="error", error=f"{type(e).__name__}: {e}")
        finally:
            self._stop.set()
            if self._paused_by_us:
                self._paused_by_us = False
                self.resume()
            if receiver is not None:
                receiver.join(timeout=3)
            channel.close()
            self._set(running=False)

    def _receive_loop(self, stub):
        errors = 0
        while not self._stop.is_set():
            try:
                reply = stub.GetActions(services_pb2.Empty(), timeout=30)
                errors = 0
            except grpc.RpcError as e:
                if self._stop.is_set():
                    return
                errors += 1
                self._set(error=f"GetActions failed: {e.code().name if hasattr(e, 'code') else e}")
                if errors >= 10:
                    self._stop.set()
                    return
                time.sleep(0.2)
                continue
            if not reply.data:
                continue
            actions = pickle.loads(reply.data)  # nosec
            with self._lock:
                for a in actions:
                    ts = a.get_timestep()
                    if ts > self._latest:
                        self._queue[ts] = a.get_action()  # latest chunk wins
                self._chunk_size = max(self._chunk_size, len(actions))
                self._status["chunks"] += 1
                if self._must_go_sent_t is not None:
                    self._status["last_latency_s"] = round(time.time() - self._must_go_sent_t, 3)
                    self._must_go_sent_t = None
                self._must_go = True
            self._chunk_arrived.set()

    def _pop(self):
        with self._lock:
            if not self._queue:
                return None
            ts = min(self._queue)
            return ts, self._queue.pop(ts)

    def _send_observation(self, stub, task: str, force_must_go: bool = False):
        now = time.time()
        if not force_must_go and now - self._last_obs_t < self.obs_min_interval:
            return
        raw = dict(self.get_observation())
        raw["task"] = task
        with self._lock:
            must_go = force_must_go or (self._must_go and not self._queue)
            if must_go:
                self._must_go = False
                self._must_go_sent_t = now
            timestep = max(self._latest, 0)
            self._last_obs_t = now
        obs = TimedObservation(timestamp=now, observation=raw, timestep=timestep, must_go=must_go)
        stub.SendObservations(send_bytes_in_chunks(pickle.dumps(obs), services_pb2.Observation, silent=True))

    def _wait_for_chunk(self, stub, task: str):
        """Block (lockstep) until a chunk arrives; re-instruct once if the server seems to have no policy."""
        reinstructed = False
        deadline = time.time() + self.first_chunk_timeout
        while not self._stop.is_set():
            if self._chunk_arrived.wait(0.1):
                self._chunk_arrived.clear()
                with self._lock:
                    if self._queue:
                        return
            if time.time() > deadline:
                if reinstructed:
                    raise TimeoutError(f"no action chunk from the policy server in {self.first_chunk_timeout}s")
                self._reinstruct(stub, task)
                reinstructed = True
                deadline = time.time() + self.first_chunk_timeout

    def _reinstruct(self, stub, task: str):
        self._set(phase="reloading")
        stub.Ready(services_pb2.Empty(), timeout=self.connect_timeout)
        self._instruct(stub)
        self._set(phase="running")
        with self._lock:
            self._must_go = True
        self._send_observation(stub, task, force_must_go=True)

    def _control_loop(self, stub, task: str, seconds: float, lockstep: bool):
        dt = 1.0 / self.fps
        t_first = None
        executed = 0
        waiting_since = time.time()
        reinstructed = False
        while not self._stop.is_set():
            tick = time.perf_counter()
            item = self._pop()
            if item is not None:
                ts, vec = item
                self.send_action(dict(zip(self.action_keys, (float(x) for x in vec.tolist()))))
                with self._lock:
                    self._latest = ts
                executed += 1
                t_first = t_first or time.perf_counter()
                waiting_since = None
            elapsed = executed * dt if lockstep else (time.perf_counter() - t_first if t_first else 0.0)
            self._set(executed=executed, elapsed_s=round(elapsed, 2))
            if elapsed >= seconds:
                return
            with self._lock:
                qsize, chunk = len(self._queue), self._chunk_size
            if lockstep:
                if qsize == 0:
                    self.pause(); self._paused_by_us = True
                    self._send_observation(stub, task, force_must_go=True)
                    self._wait_for_chunk(stub, task)
                    self._paused_by_us = False; self.resume()
                    continue
            else:
                if qsize == 0 and waiting_since is None:
                    waiting_since = time.time()
                if (qsize == 0 and not reinstructed and waiting_since is not None
                        and time.time() - waiting_since > self.first_chunk_timeout):
                    self._reinstruct(stub, task)
                    reinstructed = True
                    waiting_since = time.time()
                if chunk <= 0 or qsize / chunk <= self.chunk_size_threshold:
                    self._send_observation(stub, task)
            time.sleep(max(0.0, dt - (time.perf_counter() - tick)))


def handle_http(runner: PolicyRunner, method: str, path: str, payload: dict) -> tuple[int, dict] | None:
    """Route /policy endpoints; returns (http_code, json) or None if the path isn't ours."""
    if path == "/policy" and method == "GET":
        return 200, {"ok": True, **runner.status()}
    if path == "/policy" and method == "POST":
        try:
            st = runner.start(payload.get("task") or DEFAULT_TASK, payload.get("seconds", 30.0),
                              bool(payload.get("lockstep", False)))
            return 200, {"ok": True, **st}
        except PolicyBusy as e:
            return 409, {"ok": False, "error": str(e)}
        except PolicyUnavailable as e:
            return 503, {"ok": False, "error": str(e)}
        except ValueError as e:
            return 400, {"ok": False, "error": str(e)}
    if path == "/policy/stop" and method == "POST":
        return 200, {"ok": True, **runner.stop()}
    return None
```

- [ ] **Step 4: Run tests** → pass.
- [ ] **Step 5: Commit.**

### Task 5: Sim HTTP server

**Files:**
- Create: `halloween_bot/sim/server.py`, `tests/test_sim_server.py`

**Interfaces:**
- Consumes: `SimEngine`, `PolicyRunner`, `handle_http`, `PI0FAST_CAMERAS`, `KEYS`.
- Produces:
  - `make_app(engine: SimEngine, runner: PolicyRunner | None = None, frames_dir: Path = FRAMES_DIR) -> type[BaseHTTPRequestHandler]`
  - `policy_observation(engine) -> dict`
  - `main(argv=None)`, with CLI flags `--port 8399 --policy-server 127.0.0.1:8081 --no-policy`
- Endpoints: all of `server.py`'s, plus `POST /sim/reset {"randomize"}`, `POST /trajectory {"points", "hz"}`, `GET/POST /policy`, `POST /policy/stop`.
  - `/state` returns `{"ok", "positions", "teleop": false, "sim": true, "paused", "policy": {...}}`.
  - `/move` and `/trajectory` return 409 while a policy runs.

- [ ] **Step 1: Failing tests**

```python
# tests/test_sim_server.py
import json, threading, time, urllib.error, urllib.request
from http.server import ThreadingHTTPServer
import pytest
from halloween_bot.sim.calib import KEYS
from halloween_bot.sim.engine import SimEngine
from halloween_bot.sim.server import make_app

@pytest.fixture(scope="module")
def base(tmp_path_factory):
    eng = SimEngine(); eng.start()
    class Runner:  # stands in for PolicyRunner
        running = False
        def status(self): return {"running": self.running, "phase": "idle"}
        def start(self, *a): self.running = True; return self.status()
        def stop(self): self.running = False; return self.status()
    runner = Runner()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_app(eng, runner, tmp_path_factory.mktemp("frames")))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}", runner
    httpd.shutdown(); eng.stop()

def call(url, payload=None):
    req = urllib.request.Request(url)
    if payload is not None:
        req.data = json.dumps(payload).encode(); req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as r: return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e: return e.code, json.loads(e.read())

def test_state_matches_real_shape(base):
    url, _ = base
    code, st = call(url + "/state")
    assert code == 200 and st["ok"] and st["sim"] is True and st["teleop"] is False
    assert list(st["positions"]) == KEYS

def test_move_and_cam(base):
    url, _ = base
    code, out = call(url + "/move", {"targets": {"left_gripper.pos": 70}, "duration": 1.0})
    assert code == 200 and out["ok"] and abs(out["reached"]["left_gripper.pos"] - 70) < 8
    code, cam = call(url + "/cam?name=overhead")
    assert code == 200 and cam["ok"] and cam["path"].endswith(".jpg")

def test_move_unknown_key_and_bounds(base):
    url, _ = base
    code, out = call(url + "/move", {"targets": {"nope.pos": 1}})
    assert out["error"].startswith("unknown joint keys") and "valid_keys" in out
    code, out = call(url + "/move", {"targets": {"left_gripper.pos": 250}, "duration": 0.5})
    assert out["requested"]["left_gripper.pos"] == 100

def test_unknown_camera_and_teleop(base):
    url, _ = base
    assert "error" in call(url + "/cam?name=nope")[1]
    code, out = call(url + "/teleop", {"enabled": True})
    assert out["ok"] is False and "sim" in out["error"]

def test_move_rejected_while_policy_runs(base):
    url, runner = base
    runner.running = True
    try:
        code, out = call(url + "/move", {"targets": {"left_gripper.pos": 50}})
        assert code == 409 and "policy" in out["error"]
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
```

- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement**

```python
# halloween_bot/sim/server.py
#!/usr/bin/env python
"""Sim robot server: the MuJoCo twin behind the SAME HTTP API as halloween_bot/server.py, so the
web app, ctl.py and the Claude agent work unchanged against it.

  GET  /state  POST /move  GET /cam?name=  GET /stream?name=  POST /teleop  POST /cam_restart
  POST /stop                                   (same JSON shapes as the real server)
  POST /sim/reset {"randomize": bool}          arms to rest pose, toys re-placed
  POST /trajectory {"points": [...], "hz": 30} sys-ID: commanded sequence -> per-tick trace
  GET|POST /policy  POST /policy/stop          π₀-FAST via the shared PolicyRunner

Cameras: overhead, left_wrist, right_wrist (as on the rig) + scene (third-person view).
Start via:  python -m halloween_bot.ctl start --sim
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from ..policy_runner import PI0FAST_CAMERAS, PolicyRunner, handle_http
from .calib import KEYS
from .engine import SimEngine

PORT = 8399
FRAMES_DIR = Path.home() / "lerobot/outputs/claude_robot/frames"
POLICY_IMAGE_HW = (480, 640)  # the real pi0fast client captured all three cams at 640x480


def policy_observation(engine: SimEngine) -> dict:
    """Joint state + the three rig cameras renamed/shaped the way the real pi0fast client sent them."""
    obs = engine.read_positions()
    h, w = POLICY_IMAGE_HW
    for cam, feature in PI0FAST_CAMERAS.items():
        img = engine.get_frame(cam)
        ih, iw = img.shape[:2]
        if iw * h != ih * w:  # C922 16:9 -> 4:3 mode crops the sides
            cw = ih * w // h
            x0 = (iw - cw) // 2
            img = img[:, x0:x0 + cw]
        obs[feature] = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return obs


def policy_features() -> dict:
    h, w = POLICY_IMAGE_HW
    return {**{k: float for k in KEYS}, **{f: (h, w, 3) for f in PI0FAST_CAMERAS.values()}}


def make_app(engine: SimEngine, runner=None, frames_dir: Path = FRAMES_DIR, on_stop=None):
    busy = threading.Lock()  # one motion command at a time, like the real server's lock

    def policy_running() -> bool:
        return runner is not None and runner.running

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _reply(self, obj, code=200):
            body = json.dumps(obj, indent=2, default=float).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _stream(self, name):
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    frame = engine.get_frame(name)
                    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                                           [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         + f"Content-Length: {len(jpg)}\r\n\r\n".encode()
                                         + jpg.tobytes() + b"\r\n")
                    time.sleep(1 / 15)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            url = urlparse(self.path)
            name = parse_qs(url.query).get("name", ["left_wrist"])[0]
            try:
                if url.path == "/state":
                    self._reply({"ok": True, "positions": engine.read_positions(), "teleop": False, "sim": True,
                                 "paused": engine.paused,
                                 "policy": runner.status() if runner is not None else None})
                elif url.path == "/teleop":
                    self._reply({"ok": True, "enabled": False})
                elif url.path == "/cam":
                    try:
                        frame = engine.get_frame(name)
                    except KeyError:
                        self._reply({"error": f"unknown camera '{name}'",
                                     "cameras": ["overhead", "left_wrist", "right_wrist", "scene"]})
                        return
                    frames_dir.mkdir(parents=True, exist_ok=True)
                    path = frames_dir / f"{name}_{time.strftime('%H%M%S')}.jpg"
                    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    self._reply({"ok": True, "path": str(path)})
                elif url.path == "/stream":
                    try:
                        engine.get_frame(name)
                    except KeyError:
                        self._reply({"error": f"unknown camera '{name}'"}, 404)
                        return
                    self._stream(name)
                elif runner is not None and (r := handle_http(runner, "GET", url.path, {})):
                    self._reply(r[1], r[0])
                else:
                    self._reply({"error": "unknown endpoint"}, 404)
            except Exception as e:
                self._reply({"error": f"{type(e).__name__}: {e}"}, 500)

        def do_POST(self):
            url = urlparse(self.path)
            try:
                n = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(n) or b"{}")
                if url.path in ("/move", "/trajectory") and policy_running():
                    self._reply({"error": "the π policy is driving the arms — POST /policy/stop first."}, 409)
                elif url.path == "/move":
                    with busy:
                        self._reply(engine.move(payload.get("targets", {}), payload.get("duration", 2.0)))
                elif url.path == "/trajectory":
                    with busy:
                        self._reply(engine.run_trajectory(payload.get("points", []), payload.get("hz", 30.0)))
                elif url.path == "/sim/reset":
                    if policy_running():
                        runner.stop()
                    with busy:
                        engine.reset(randomize=bool(payload.get("randomize")), seed=payload.get("seed"))
                    self._reply({"ok": True, "msg": "sim reset: arms at rest pose, toys re-placed"})
                elif url.path == "/teleop":
                    self._reply({"ok": False, "error": "no leader arms in sim — use /move or /policy"})
                elif url.path == "/cam_restart":
                    self._reply({"ok": True, "msg": "sim cameras never stall"})
                elif url.path == "/stop":
                    self._reply({"ok": True, "msg": "sim server shutting down"})
                    if on_stop:
                        threading.Thread(target=on_stop, daemon=True).start()
                elif runner is not None and (r := handle_http(runner, "POST", url.path, payload)):
                    self._reply(r[1], r[0])
                else:
                    self._reply({"error": "unknown endpoint"}, 404)
            except Exception as e:
                self._reply({"error": f"{type(e).__name__}: {e}"}, 500)

    return Handler


def _tailnet_hosts() -> list[str]:
    ts_bin = shutil.which("tailscale") or str(Path.home() / ".local/bin/tailscale")
    try:
        ts = subprocess.run([ts_bin, "ip", "-4"], capture_output=True, text=True, timeout=5)
        if ts.returncode == 0 and ts.stdout.strip():
            return [ts.stdout.strip().splitlines()[0]]
    except Exception:
        pass
    return []


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--policy-server", default="127.0.0.1:8081")
    ap.add_argument("--no-policy", action="store_true")
    args = ap.parse_args(argv)

    engine = SimEngine()
    engine.start()
    runner = None
    if not args.no_policy:
        runner = PolicyRunner(lambda: policy_observation(engine), engine.send_action, KEYS, policy_features(),
                              server_address=args.policy_server, pause=engine.pause, resume=engine.resume)
    servers = []

    def shutdown():
        time.sleep(0.3)
        if runner is not None:
            runner.stop()
        engine.stop()
        for s in servers:
            s.shutdown()

    handler = make_app(engine, runner, on_stop=shutdown)
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    servers.append(httpd)
    for h in _tailnet_hosts():
        try:
            extra = ThreadingHTTPServer((h, args.port), handler)
            servers.append(extra)
            threading.Thread(target=extra.serve_forever, daemon=True).start()
            print(f"also serving on http://{h}:{args.port} (tailnet)", flush=True)
        except OSError as e:
            print(f"tailnet bind {h}:{args.port} failed: {e}", flush=True)
    print(f"SIM robot server on http://127.0.0.1:{args.port} (policy server {args.policy_server})", flush=True)
    httpd.serve_forever()
    print("sim server stopped", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests** → pass. `pytest tests -q` should be all green.
- [ ] **Step 5: Commit.**

### Task 6: CLI, launcher, live π smoke test

**Files:**
- Modify: `halloween_bot/ctl.py`
- Create: `run_sim.sh`

**Interfaces:**
- Consumes: the sim server endpoints (Task 5).
- Produces: the CLI commands the agent uses:
  - `ctl start --sim`
  - `ctl policy "<task>" [--seconds S] [--lockstep] [--no-wait]`
  - `ctl policy-stop`
  - `ctl sim-reset [--randomize]`
  - `ctl cam {left_wrist,right_wrist,overhead,scene}`

- [ ] **Step 1: Modify `ctl.py`.** Add these, leaving the real `start` path untouched:

```python
# in cmd_start: choose the server to launch
def cmd_start(args):
    if server_alive():
        print("server already running")
        return
    Path(LOG).parent.mkdir(parents=True, exist_ok=True)
    cmd = [PYTHON, "-m", "halloween_bot.sim.server"] if args.sim else [PYTHON, SERVER]
    log_path = SIM_LOG if args.sim else LOG
    with open(log_path, "ab") as log:
        subprocess.Popen(["setsid", *cmd], stdout=log, stderr=log, start_new_session=True)
    for _ in range(60):  # connect (real) / model build + EGL (sim) take a few seconds
        time.sleep(1)
        if server_alive():
            print("SIM server up (MuJoCo twin on :8399)" if args.sim else "server up, robot connected (torque ON)")
            return
    sys.exit(f"server did not come up — check log: {log_path}")


def cmd_policy(args):
    res = call("/policy", {"task": args.task, "seconds": args.seconds, "lockstep": args.lockstep})
    if not res.get("ok"):
        sys.exit(f"policy start failed: {res.get('error')}")
    print(f"π policy started: task={args.task!r} seconds={args.seconds} lockstep={args.lockstep}", flush=True)
    if args.no_wait:
        return
    last = None
    while True:
        time.sleep(2)
        st = call("/policy")
        line = f"[{st['phase']}] t={st['elapsed_s']}s executed={st['executed']} chunks={st['chunks']}"
        if line != last:
            print(line, flush=True)
            last = line
        if not st["running"]:
            break
    print(json.dumps(st, indent=2))


def cmd_policy_stop(_):
    print(json.dumps(call("/policy/stop", {}), indent=2))


def cmd_sim_reset(args):
    print(json.dumps(call("/sim/reset", {"randomize": args.randomize}), indent=2))
```

`call()` currently `sys.exit`s on HTTP errors, which prints the server's 409/503 JSON; that's fine for the agent. Parser additions:

```python
SIM_LOG = str(Path.home() / "lerobot/outputs/claude_robot/sim_server.log")
DEFAULT_TASK = "Grasp the toy and place it in the basket."
sp = sub.add_parser("start"); sp.add_argument("--sim", action="store_true", help="start the MuJoCo twin instead of the real robot"); sp.set_defaults(fn=cmd_start)
pp = sub.add_parser("policy", help="run the π₀-FAST policy (blocks until it finishes)")
pp.add_argument("task", nargs="?", default=DEFAULT_TASK)
pp.add_argument("--seconds", type=float, default=30.0)
pp.add_argument("--lockstep", action="store_true", help="sim only: pause physics during inference")
pp.add_argument("--no-wait", action="store_true")
pp.set_defaults(fn=cmd_policy)
sub.add_parser("policy-stop").set_defaults(fn=cmd_policy_stop)
rp = sub.add_parser("sim-reset"); rp.add_argument("--randomize", action="store_true"); rp.set_defaults(fn=cmd_sim_reset)
cp.add_argument("name", choices=["left_wrist", "right_wrist", "overhead", "scene"])
```

Also update the module docstring with the new commands.

- [ ] **Step 2: Create `run_sim.sh`**

```bash
#!/usr/bin/env bash
# Bring up the sim stack on the 5090: π₀-FAST policy server (:8081, ~/pi-serve venv with its
# pi0_fast shims — never reinstall that venv), MuJoCo sim server (:8399), web app (:8500).
# :8080 is left to the 1080's real-robot client. Logs: ~/lerobot/outputs/claude_robot/*.log
set -euo pipefail
cd "$(dirname "$0")"
LOG=~/lerobot/outputs/claude_robot
mkdir -p "$LOG"
if ! pgrep -f "policy_server.*--port=8081" >/dev/null; then
  (cd ~/pi-serve && setsid .venv/bin/python -m lerobot.async_inference.policy_server \
      --host=127.0.0.1 --port=8081 >>"$LOG/policy_server_sim.log" 2>&1 &)
  echo "policy server starting on :8081 (first π run loads the model, ~50 s)"
fi
.venv/bin/python -m halloween_bot.ctl start --sim
if ! pgrep -f "halloween_bot.app.webapp" >/dev/null; then
  setsid .venv/bin/python -m halloween_bot.app.webapp >>"$LOG/webapp.log" 2>&1 &
  sleep 2
fi
echo "dashboard: http://127.0.0.1:8500/   face: http://127.0.0.1:8500/face?theme=halloween"
```

- [ ] **Step 3: Live smoke test.**
  - Run `./run_sim.sh`.
  - Run `.venv/bin/python -m halloween_bot.ctl policy --seconds 20`. Expected: phases `loading → running → done`, `chunks ≥ 5`, joints changing in `ctl state`.
  - Repeat with `--lockstep`.
  - Record a scene-cam MP4 of a run and send it to Jinyu.
- [ ] **Step 4: Commit.**

### Task 7: Web app (sim-aware dashboard)

**Files:**
- Modify: `halloween_bot/app/webapp.py`, `halloween_bot/app/static/dashboard.html`

**Interfaces:**
- Consumes: `/state` (`sim`, `policy`), `/policy`, `/policy/stop`, `/sim/reset`.
- Produces: `/api/policy` (GET/POST), `/api/policy/stop`, `/api/sim/reset` proxies; `HALLOWEEN_ROBOT_API` env var.

- [ ] **Step 1: `webapp.py`.**
  - Set `ROBOT_API = os.environ.get("HALLOWEEN_ROBOT_API", "http://127.0.0.1:8399")`.
  - Add a proxy helper:

```python
def robot_call(path: str, payload: dict | None = None, timeout: float = 30) -> tuple[int, dict]:
    req = urllib.request.Request(f"{ROBOT_API}{path}")
    if payload is not None:
        req.data = json.dumps(payload).encode(); req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")
```

  - In `do_GET`, add `elif url.path == "/api/policy": self._json(*reversed(robot_call("/policy")))`. Handle a down robot server with a 502, the way `/api/state` does.
  - In `do_POST`, add `/api/policy` → `robot_call("/policy", payload)`, `/api/policy/stop` and `/api/sim/reset`, each broadcasting a status event (`"π policy: started …"`, `"sim reset"`).
  - Add `import urllib.error`.

- [ ] **Step 2: `dashboard.html`.**
  - A `SIM` badge in the header, hidden unless `state.sim`.
  - A `scene` cam panel, hidden unless sim.
  - A policy bar under the joints panel:
    - task `<input>` defaulting to "Grasp the toy and place it in the basket.";
    - a seconds number input (30) and a lockstep checkbox (sim only);
    - `▶ run π` and `■ stop` buttons;
    - `⟲ reset` (sim only);
    - a status line fed from `state.policy`: phase, elapsed, chunks, latency.
  - Hide the teleop button when sim. Poll through the existing `pollState()`.
- [ ] **Step 3: Verify in the browser** with the claude-in-chrome tools: the SIM badge, 4 streams, policy start/stop and reset all work. Save a screenshot for Jinyu.
- [ ] **Step 4: Commit.**

### Task 8: Agent wiring (CLAUDE.md)

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Rewrite the command section for both machines.** The 5090 uses `PY=.venv/bin/python; $PY -m halloween_bot.ctl …` from the repo root. The 1080 uses `~/miniconda3/envs/lerobot/bin/python ~/lerobot/claude_robot/ctl.py …`. Add a "Two ways to act" section:
  - **Direct joint moves** (`ctl move`) for gestures, waving, pointing, presenting candy, and anything scripted or precise.
  - **The π policy** (`ctl policy "Grasp the toy and place it in the basket." --seconds 30`) for picking up a toy and putting it in the basket, which is what it was trained on. It blocks and prints progress. `ctl policy-stop` aborts. `/move` is refused while it runs.
  - Look before and after either (overhead cam). Prefer direct moves when π isn't making progress after one run.
- [ ] **Step 2: Add a "Sim mode" section.**
  - `ctl state` shows `"sim": true` on the MuJoCo twin.
  - Droop/stop rules don't apply.
  - `ctl sim-reset [--randomize]` restores the scene.
  - `ctl cam scene` gives a third-person view.
  - The latest overhead frame is still at `~/lerobot/outputs/claude_robot/latest/overhead.jpg`.
- [ ] **Step 3: End-to-end check.**
  - With the stack up, send `/instruct` "wave the right arm" (agent `claude`). It must use `ctl move`.
  - Send "put a toy in the basket". It must call `ctl policy`.
  - Check both in the webapp feed.
- [ ] **Step 4: Commit** and tell the 1080 session that CLAUDE.md changed.

### Task 9: System identification

**Files:**
- Create: `halloween_bot/sim/sysid.py`, `halloween_bot/sim/sysid_collect.py`, `tests/test_sysid.py`, `halloween_bot/sim/params.json`, `docs/sysid/report.md` (+ PNGs)

**Interfaces:**
- Consumes: `build_model(props=False)`, `Calibration`, `KEYS`, `MAX_RELATIVE_TARGET`.
- Produces:
  - `Trace(name, present: np.ndarray[T+1, 12], cmd: np.ndarray[T, 12], hz)`
  - `load_dataset_traces(root) -> list[Trace]`
  - `load_trajectory_traces(path) -> list[Trace]`
  - `replay(model, calib, trace) -> np.ndarray[T, 12]` (predicted present[1:])
  - `fit(traces, init=None, motors=MOTORS) -> dict` (params)
  - `rmse(model, calib, traces, holdout=0.2) -> dict[key, float]`
  - CLI: `python -m halloween_bot.sim.sysid fit --dataset ~/so101_bimanual_test_dataset [--traces f.json] --out halloween_bot/sim/params.json --report docs/sysid`

- [ ] **Step 1: Failing tests**

```python
# tests/test_sysid.py
import numpy as np
from halloween_bot.sim.calib import KEYS, Calibration
from halloween_bot.sim.model import build_model
from halloween_bot.sim.sysid import Trace, fit, replay, rmse

CAL = Calibration.load()
TRUE = {"motors": {m: {"kp": 12.0, "kv": 0.4} for m in ("shoulder_pan", "shoulder_lift", "elbow_flex",
                                                           "wrist_flex", "wrist_roll", "gripper")}}

def synthetic_trace(params, T=240):
    m = build_model(params=params, props=False)
    rest = np.array([-5, -86, 95, 45, -10, 40] * 2, float)
    t = np.arange(T) / 30
    cmd = rest + np.outer(np.sin(2 * np.pi * 0.7 * t), [6, 12, -12, 10, 8, 15] * 2)
    tr = Trace("synthetic", np.vstack([rest, np.zeros((T, 12))]), cmd, 30.0)
    tr.present[1:] = replay(m, CAL, tr)
    return tr

def test_replay_shape_and_tracking():
    tr = synthetic_trace(TRUE)
    assert tr.present.shape == (241, 12)
    assert np.abs(tr.present[-1] - tr.cmd[-1]).max() < 20

def test_fit_recovers_better_than_default():
    tr = synthetic_trace(TRUE)
    before = np.mean(list(rmse(build_model(params={}, props=False), CAL, [tr]).values()))
    params = fit([tr], max_nfev=40)
    after = np.mean(list(rmse(build_model(params=params, props=False), CAL, [tr]).values()))
    assert after < before * 0.5
```

- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement `sysid.py`.**
  - `Trace` is a dataclass.
  - `load_dataset_traces` reads `data/chunk-*/file-*.parquet` with pandas, splits by `episode_index`, stacks `observation.state` as `present` and uses `action[:-1]` as `cmd`.
  - `load_trajectory_traces` reads JSON `{"sequences": [{"name", "hz", "trace": [{t, present, cmd}]}]}`. Presents are ordered by `KEYS`, and the final entry has `cmd: None`.
  - `replay`: set qpos and ctrl from `present[0]` and zero qvel, then for each t:
    - goal = clip(cmd[t], present_sim ± 20), bounded per key;
    - set ctrl from the affine calibration;
    - `mj_step(nstep=round(1/(hz*timestep)))`;
    - record the normalized qpos.
  - `fit`:
    - parameter vector = log of `kp, kv, damping, frictionloss, armature` per motor, 30 values shared by both arms;
    - start from the Menagerie `sts3215` class defaults, or `init`;
    - `scipy.optimize.least_squares` with `x_scale="jac"` and bounds of log(1e-4)..log(2e3);
    - residuals = (sim − real) over the first 80% of each trace, per joint, divided by joint RMS;
    - returns `{"motors": {...}, "fit": {"traces": [...], "cost": ...}}`.
  - `rmse` covers the last 20% (holdout), per key.
  - The CLI writes `params.json` and `docs/sysid/report.md` (a table of Menagerie vs fitted holdout RMSE per key) plus an `overlay_<key>.png` per joint using matplotlib (Agg).
- [ ] **Step 4: Implement `sysid_collect.py`** (runs on the 1080 against `POST <server>/trajectory`):
  - args `--arm right|left`, `--server http://127.0.0.1:8399`, `--out traces.json`, `--dry-run` (prints the sequences and duration, sends nothing);
  - holds the base pose `SURVEY = pan −22, lift −63, elbow 80, wf 40, roll −10, grip 30` on the chosen arm;
  - for each motor: steps of +10 / −10 / +20 / −20 around the base pose, held 1.5 s each, then a chirp of amplitude 8 from 0.3 to 2.0 Hz over 8 s, then a return to base;
  - gripper: 30→60→30→10 in free air;
  - the other arm holds its current pose;
  - it never exceeds ±20 units from the base pose;
  - chunks are posted one sequence per request and written to the JSON format `load_trajectory_traces` reads.
- [ ] **Step 5: Run the fit on the teleop dataset.** Write `params.json` and the report, then rerun the pytest suite so the engine tests still pass with the fitted params. Confirm that the fitted body-joint holdout RMSE is lower than Menagerie's, and report both.
- [ ] **Step 6: Commit.** Then send the 1080 session:
  - the `/trajectory` endpoint spec, with the exact code to paste into `server.py`, mirroring `SimEngine.run_trajectory` with `robot.send_action`, `read_positions`, `time.sleep` and the teleop 409 guard;
  - the `sysid_collect.py --dry-run` output;
  - a note that it waits for Jinyu's explicit go.

### Task 10: Camera match + hand-off

**Files:**
- Modify: `halloween_bot/sim/model.py` (`FIXED_CAMERAS`, `BASKET`, `TOYS`)

- [ ] **Step 1: Tune the overhead camera.**
  - Render the rest pose from `overhead` beside `~/sim_ref/overhead.jpg`.
  - Adjust the position, look-at point and fovy until the base positions (px), the arm spacing (~600 px) and the gripper tips match within ~5% of the image width.
  - Record the final numbers in a comment.
- [ ] **Step 2: Check the wrist cams** against the teleop dataset's wrist video frames, where the gripper jaws are visible. Menagerie's `wrist_cam` pose should put the jaws in the same image region. Adjust `fovy` only if it's clearly off.
- [ ] **Step 3: Commit.** Then message the 1080 session with the `PolicyRunner` wiring snippet for `server.py`:

```python
from halloween_bot.policy_runner import PI0FAST_CAMERAS, PolicyRunner, handle_http
def policy_obs():
    with lock:
        obs = read_positions()
    return {**obs, **{PI0FAST_CAMERAS[n]: cams()[n].async_read(timeout_ms=2000) for n in PI0FAST_CAMERAS}}
def policy_send(action):
    with lock:
        robot.send_action(action)
runner = PolicyRunner(policy_obs, policy_send, list(robot.action_features),
                      {**{k: float for k in robot.action_features},
                       **{f: (480, 640, 3) for f in PI0FAST_CAMERAS.values()}},  # set cams to 640x480 like biarm client
                      server_address="127.0.0.1:8080")
# in do_GET/do_POST: `if (r := handle_http(runner, method, url.path, payload)): self._reply(r[1], r[0]); return`
# and guard /move + /teleop with 409 while runner.running
```

Note: the real overhead cam runs at 1280×720. Either resize to 640×480 in `policy_obs`, as the sim does in `policy_observation`, or declare `(720, 1280, 3)` for `base_0_rgb`.
