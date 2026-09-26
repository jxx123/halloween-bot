#!/usr/bin/env python
"""Kinematic calibration of the SO101 twin from real table contacts.

The real arms are WOWROBO builds; their link/claw geometry differs from stock SO101 (Menagerie):
folded poses match within ~1 cm but the error grows with reach (~5 cm at mid-reach). No single
joint offset or claw length explains that, so this fits link-length deltas and joint offsets
jointly (both arms share them):

  lengths_m:         upper_arm_dx, forearm_dx, claw_dx (along the wrist-roll axis), base_dz
  joint_offsets_deg: shoulder_lift, elbow_flex, wrist_flex

Data (the 1080's table-probe runs, JSON list of rows):
  contact row:    {"present": {...}, "cmd_at_contact": {...}, "start": {...}?, "floor": -95?, "confirm": {"grew": bool}?}
  no-contact row: {"error": "no contact before floor", "start": {...}, "floor": ...}
A contact means the closed claws press the table (clearance ≈ -PRESS). Every commanded pose on the
descent before the contact (or down to the floor) means the claws are still above it.
Anchors are free poses (rest, photo overlays): the claws must not be inside the table.

    python -m halloween_bot.sim.kincal fit --contacts ~/sim_ref/table_contacts_v3.json \\
        --anchor ~/sim_ref/state.json --anchor ~/sim_ref/overlay/a/state.json --out halloween_bot/sim/geometry.json
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares, minimize

from .calib import GEOMETRY_FILE, KEYS, Calibration, split_key
from .engine import REST
from .model import apply_geometry, build_model

LENGTHS = ("upper_arm_dx", "forearm_dx", "claw_dx", "base_dz")
OFFSETS = ("shoulder_lift", "elbow_flex", "wrist_flex")
PARAMS = LENGTHS + OFFSETS + ("base_pitch",)
# prior σ: lengths in m, angles in rad. Printed links are rigid (±1 cm); the WOWROBO claws may differ
# more; calibration offsets should be small; the mount tilt is the free, physically plausible knob
# (a forward-tilted base gives exactly the reach-proportional drop the table contacts show).
PRIOR = np.array([0.01, 0.01, 0.02, 0.02, math.radians(5), math.radians(5), math.radians(8), math.radians(15)])
PRESS = 0.003  # m: confirmed contacts press the claws a few mm into the (soft) contact
CONTACT_SIGMA = 0.005
FREE_SIGMA = 0.005
STEP = {"right_shoulder_lift.pos": -1.5, "right_elbow_flex.pos": 1.0}  # the rig's descent coupling
CLAW_BODIES = ("gripper", "moving_jaw_so101_v1")  # not the camera mount: the real camera board differs
SIL_HW = (180, 320)  # silhouette resolution (h, w)
IOU_SIGMA = 0.1  # 0.01 IoU ~ a 3.5 mm contact error: photos steer the shape, contacts own the heights
CAM_PRIOR = np.array([0.05, 0.05, 0.05, math.radians(10), math.radians(10), math.radians(10)])


@dataclass
class Row:
    name: str
    present: dict | None  # state at a contact, None for no-contact rows
    path: list[dict] = field(default_factory=list)  # commanded poses known to be above the table
    weight: float = 1.0
    trigger: dict | None = None  # state when the photo was taken (v3)
    photo: tuple[str, dict] | None = None  # (overhead jpg, state) for the silhouette term


def geometry_from_x(x) -> dict:
    return {"lengths_m": {k: float(v) for k, v in zip(LENGTHS, x[:4])},
            "joint_offsets_deg": {k: math.degrees(float(v)) for k, v in zip(OFFSETS, x[4:7])},
            "base_pitch_deg": math.degrees(float(x[7]))}


def x_from_geometry(geometry: dict) -> np.ndarray:
    g, o = geometry.get("lengths_m", {}), geometry.get("joint_offsets_deg", {})
    return np.array([g.get(k, 0.0) for k in LENGTHS] + [math.radians(o.get(k, 0.0)) for k in OFFSETS]
                    + [math.radians(geometry.get("base_pitch_deg", 0.0))])


class Kin:
    """Kinematics-only twin: geometry deltas + joint offsets -> closed-claw/table clearance."""

    def __init__(self, geometry: dict | None = None, side: str = "right"):
        self.side = side
        raw = Calibration.load(geometry=None)
        # the zero convention is fixed per run (not fitted); fitted offsets are applied in pose()
        self.cal = Calibration(raw.calib, zero=(geometry or {}).get("zero"))
        self.model = build_model(params={}, calib=self.cal, props=False, geometry={})
        self.data = mujoco.MjData(self.model)
        m = self.model
        self._pristine = {name: m.body(name).pos.copy() for s in ("left", "right")
                          for name in (f"{s}_lower_arm", f"{s}_wrist", f"{s}_gripper", f"{s}_base")}
        self._pristine_quat = {f"{s}_base": m.body(f"{s}_base").quat.copy() for s in ("left", "right")}
        self._qadr = {k: m.joint(k.removesuffix(".pos")).qposadr[0] for k in KEYS}
        self._table = m.geom("table").id
        claws = {f"{side}_{b}" for b in CLAW_BODIES}
        self._claw = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in claws
                      and (m.geom_contype[g] or m.geom_conaffinity[g])]
        self._fromto = np.zeros(6)
        self.set(geometry or {})

    def set(self, geometry: dict):
        for name, pos in self._pristine.items():
            self.model.body(name).pos[:] = pos
        for name, quat in self._pristine_quat.items():
            self.model.body(name).quat[:] = quat
        apply_geometry(self.model, geometry)
        self.offsets = {k: math.radians(v) for k, v in geometry.get("joint_offsets_deg", {}).items()}

    def pose(self, pos: dict, full: bool = False):
        joints = {**REST, **pos}
        for k in KEYS:
            self.data.qpos[self._qadr[k]] = self.cal.to_rad(k, joints[k]) + self.offsets.get(split_key(k)[1], 0.0)
        (mujoco.mj_forward if full else mujoco.mj_kinematics)(self.model, self.data)  # full: cameras too

    def clearance(self, pos: dict) -> float:
        self.pose(pos)
        return min(mujoco.mj_geomDistance(self.model, self.data, g, self._table, 0.5, self._fromto)
                   for g in self._claw)


def blue_arm_mask(path: str, hw: tuple[int, int] = SIL_HW) -> np.ndarray:
    """The blue (right) arm in a lit overhead frame: cyan hue on a dark mat, largest blob, left side."""
    img = cv2.resize(cv2.imread(os.path.expanduser(path)), (640, 360))
    m = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), (78, 45, 45), (112, 255, 255))
    m[:, 360:] = 0  # the orange arm and the right half of the scene
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return np.zeros(hw, bool)
    blob = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    return cv2.resize(blob, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST) > 0


class Silhouettes:
    """IoU of the rendered right arm vs the real blue-arm mask, per photo, with a camera-pose delta."""

    def __init__(self, kin: Kin, photos: list[tuple[str, dict]]):
        m = kin.model
        self.kin, self.states = kin, [s for _, s in photos]
        self.real = [blue_arm_mask(p) for p, _ in photos]
        self.cam = m.camera("overhead").id
        self.pos0, self.quat0 = m.cam_pos[self.cam].copy(), m.cam_quat[self.cam].copy()
        self.arm = np.array([m.body(b).name.startswith(f"{kin.side}_") and m.body(b).name != f"{kin.side}_base"
                             for b in range(m.nbody)])
        self.renderer = mujoco.Renderer(m, *SIL_HW)
        self.renderer.enable_segmentation_rendering()

    def set_camera(self, c) -> dict:
        m = self.kin.model
        dq, q = np.zeros(4), np.zeros(4)
        mujoco.mju_euler2Quat(dq, np.asarray(c[3:6], float), "xyz")
        mujoco.mju_mulQuat(q, self.quat0, dq)
        m.cam_pos[self.cam] = self.pos0 + np.asarray(c[:3], float)
        m.cam_quat[self.cam] = q
        return {"pos": m.cam_pos[self.cam].tolist(), "quat": m.cam_quat[self.cam].tolist()}

    def ious(self) -> np.ndarray:
        m, d, out = self.kin.model, self.kin.data, []
        for state, real in zip(self.states, self.real):
            self.kin.pose(state, full=True)
            self.renderer.update_scene(d, "overhead")
            seg = self.renderer.render()[:, :, 0]
            sim = (seg >= 0) & self.arm[m.geom_bodyid[np.clip(seg, 0, m.ngeom - 1)]]
            out.append((sim & real).sum() / max(1, (sim | real).sum()))
        return np.asarray(out)


def _descent(start: dict, lift_floor: float, stop: dict | None = None, inclusive: bool = False,
             elbow_ceil: float = 100.0) -> list[dict]:
    """Commanded poses along the coupling from start down to the floor, or to the stop pose
    (exclusive for a contact, inclusive for a trigger that failed confirmation). The elbow holds at
    its ceiling while the lift keeps descending, as the rig's probe does."""
    pose, out = dict(start), []
    stop_lift = None if stop is None else stop["right_shoulder_lift.pos"]
    for _ in range(300):
        lift = pose["right_shoulder_lift.pos"]
        if lift < lift_floor - 1e-6:
            break
        if stop_lift is not None and (lift < stop_lift - 1e-6 or (not inclusive and lift <= stop_lift + 1e-6)):
            break
        out.append(dict(pose))
        pose["right_shoulder_lift.pos"] += STEP["right_shoulder_lift.pos"]
        pose["right_elbow_flex.pos"] = min(elbow_ceil, pose["right_elbow_flex.pos"] + STEP["right_elbow_flex.pos"])
    return out


def load_rows(paths) -> list[Row]:
    rows = []
    for path in paths:
        path = Path(os.path.expanduser(str(path)))
        for i, r in enumerate(json.loads(path.read_text())):
            name = f"{path.stem}[{i}]"
            start = r.get("start")
            floor = float(r.get("lift_floor", r.get("floor", -95.0)))
            ceil = float(r.get("elbow_ceil", 100.0))
            if "present_at_trigger" in r:  # v3: confirm pass recorded
                photo = (str(path.parent / r["frame"]), r["present_at_trigger"]) if r.get("frame") else None
                if r.get("confirmed"):
                    rows.append(Row(name, r["present_at_confirm"], _descent(start, floor, r["cmd_at_contact"], False, ceil),
                                    1.0, r["present_at_trigger"], photo))
                else:  # the deviation shrank on the extra step: no contact, even at the trigger pose
                    rows.append(Row(name, None, _descent(start, floor, r["cmd_at_contact"], True, ceil),
                                    1.0, r["present_at_trigger"], photo))
            elif "present" in r:
                weight = 1.0
                confirm = r.get("confirm")
                if confirm is not None and not confirm.get("grew", False):
                    weight = 0.2  # deviation didn't grow on the extra step: probably not a contact
                elif confirm is None and r.get("trigger_joint", "").endswith("shoulder_lift.pos") and r.get("deviation", 0) < 0:
                    weight = 0.3  # unconfirmed negative lift trigger: likely stick-slip, not a press
                cmd = r.get("cmd_at_contact", r["present"])
                rows.append(Row(name, r["present"], _descent(start, floor, cmd, False, ceil) if start else [], weight))
            elif start:
                rows.append(Row(name, None, _descent(start, floor, None, False, ceil)))
            else:
                rows.append(Row(name, None, []))  # no start recorded: carries no constraint
    return rows


def residuals(x, kin: Kin, rows: list[Row], anchors: list[dict]) -> np.ndarray:
    kin.set(geometry_from_x(x))
    r = []
    for row in rows:
        if row.present is not None:
            r.append(row.weight * (kin.clearance(row.present) + PRESS) / CONTACT_SIGMA)
        for pose in row.path[::3]:
            r.append(max(0.0, -kin.clearance(pose) - 0.002) / FREE_SIGMA)
    for pose in anchors:
        r.append(max(0.0, -kin.clearance(pose) - PRESS) / FREE_SIGMA)
    r.extend(np.asarray(x) / PRIOR)
    return np.asarray(r)


def fit(rows: list[Row], anchors: list[dict], x0=None, max_nfev: int = 200, zero: dict | None = None) -> dict:
    kin = Kin(geometry={"zero": zero or {}})
    x0 = np.zeros(len(PARAMS)) if x0 is None else np.asarray(x0, float)
    sol = least_squares(residuals, x0, args=(kin, rows, anchors), x_scale=PRIOR, diff_step=1e-3,
                        max_nfev=max_nfev)
    geom = geometry_from_x(sol.x)
    geom["zero"] = dict(zero or {})
    kin0 = Kin(geometry={})
    kin.set(geom)
    geom["fit"] = {
        "cost_start": float(0.5 * np.sum(residuals(np.zeros(len(PARAMS)), kin0, rows, anchors) ** 2)),
        "cost": float(sol.cost),
        "rows": [{"name": r.name, "weight": r.weight,
                  "clearance_cm_before": None if r.present is None else round(100 * kin0.clearance(r.present), 2),
                  "clearance_cm_after": None if r.present is None else round(100 * kin.clearance(r.present), 2),
                  "path_min_cm_after": round(100 * min(kin.clearance(p) for p in r.path), 2) if r.path else None}
                 for r in rows],
    }
    return geom


def fit_joint(rows: list[Row], anchors: list[dict], photos: list[tuple[str, dict]], maxfev: int = 4000,
              zero: dict | None = None) -> dict:
    """Geometry + overhead-camera pose against contacts, free paths, anchors AND silhouettes (Powell:
    IoU is piecewise constant, so no gradients). Stage 1 fits the camera alone on the photos."""
    kin = Kin(geometry={"zero": zero or {}})
    sil = Silhouettes(kin, photos)
    n = len(PARAMS)

    def total(z):
        kin.set(geometry_from_x(z[:n]))
        sil.set_camera(z[n:])
        r = residuals(z[:n], kin, rows, anchors)
        return float(np.sum(r ** 2) + np.sum(((1 - sil.ious()) / IOU_SIGMA) ** 2) + np.sum((z[n:] / CAM_PRIOR) ** 2))

    def cam_only(c):
        sil.set_camera(c)
        return float(np.sum(((1 - sil.ious()) / IOU_SIGMA) ** 2) + np.sum((c / CAM_PRIOR) ** 2))

    kin.set({})  # (zero convention stays: it lives in kin.cal)
    iou_start = sil.ious()
    cam = minimize(cam_only, np.zeros(6), method="Powell", options={"maxfev": maxfev // 3, "xtol": 1e-4, "ftol": 1e-5}).x
    sol = minimize(total, np.concatenate([np.zeros(n), cam]), method="Powell",
                   options={"maxfev": maxfev, "xtol": 1e-4, "ftol": 1e-5})
    geom = geometry_from_x(sol.x[:n])
    geom["zero"] = dict(zero or {})
    kin.set(geom)
    geom["overhead_camera"] = sil.set_camera(sol.x[n:])
    kin0 = Kin(geometry={})
    geom["fit"] = {
        "cost": float(sol.fun), "nfev": int(sol.nfev),
        "iou_start": [round(float(v), 3) for v in iou_start], "iou": [round(float(v), 3) for v in sil.ious()],
        "camera_delta": {"pos_m": [round(float(v), 4) for v in sol.x[n:n + 3]],
                         "rot_deg": [round(math.degrees(float(v)), 2) for v in sol.x[n + 3:]]},
        "rows": [{"name": r.name, "weight": r.weight,
                  "clearance_cm_before": None if r.present is None else round(100 * kin0.clearance(r.present), 2),
                  "clearance_cm_after": None if r.present is None else round(100 * kin.clearance(r.present), 2),
                  "path_min_cm_after": round(100 * min(kin.clearance(p) for p in r.path), 2) if r.path else None}
                 for r in rows],
    }
    return geom


def _load_state(path) -> dict:
    d = json.loads(Path(os.path.expanduser(str(path))).read_text())
    return {k: v for k, v in d.get("positions", d).items() if k.startswith("right")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    fp = sub.add_parser("fit")
    fp.add_argument("--contacts", action="append", required=True, help="table-probe JSON (repeatable)")
    fp.add_argument("--anchor", action="append", default=[], help="state.json of a free pose (repeatable)")
    fp.add_argument("--photo", action="append", default=[],
                    help="DIR with overhead.jpg + state.json (repeatable); v3 rows bring their own frames")
    fp.add_argument("--no-silhouettes", action="store_true", help="contacts only (least squares)")
    fp.add_argument("--zero", default="", help='zero conventions, e.g. "shoulder_lift=homing,elbow_flex=homing"')
    fp.add_argument("--holdout-every", type=int, default=4, help="hold out every Nth contact row for checking")
    fp.add_argument("--out", default=str(GEOMETRY_FILE))
    fp.add_argument("--max-nfev", type=int, default=200)
    a = ap.parse_args(argv)

    rows = load_rows(a.contacts)
    anchors = [_load_state(p) for p in a.anchor]
    contacts = [r for r in rows if r.present is not None]
    held = {r.name for r in contacts[a.holdout_every - 1::a.holdout_every]} if a.holdout_every > 0 else set()
    train = [r for r in rows if r.name not in held]
    print(f"{len(contacts)} contacts ({len(held)} held out), {sum(r.present is None and bool(r.path) for r in rows)} "
          f"no-contact paths, {len(anchors)} anchors", flush=True)
    zero = dict(kv.split("=") for kv in a.zero.split(",") if kv)
    if a.no_silhouettes:
        geom = fit(train, anchors, max_nfev=a.max_nfev, zero=zero)
    else:
        photos = [r.photo for r in train if r.photo] + [
            (str(Path(os.path.expanduser(p)) / "overhead.jpg"), _load_state(Path(os.path.expanduser(p)) / "state.json"))
            for p in a.photo]
        print(f"{len(photos)} photos in the silhouette term", flush=True)
        geom = fit_joint(train, anchors, photos, zero=zero)
    kin0, kin = Kin(geometry={}), Kin(geometry=geom)
    kin.set(geom)
    geom["fit"]["holdout"] = [{"name": r.name, "clearance_cm_before": round(100 * kin0.clearance(r.present), 2),
                               "clearance_cm_after": round(100 * kin.clearance(r.present), 2)}
                              for r in contacts if r.name in held]
    Path(a.out).write_text(json.dumps(geom, indent=2) + "\n")
    print(json.dumps({"lengths_m": geom["lengths_m"], "joint_offsets_deg": geom["joint_offsets_deg"],
                      "base_pitch_deg": geom.get("base_pitch_deg"), "cost": round(geom["fit"]["cost"], 2),
                      "holdout": geom["fit"]["holdout"]}, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
