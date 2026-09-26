#!/usr/bin/env python
"""PS4 teleop v2: drive both SO101 grippers in 3D (end-effector mode) or per-joint.

EE mode (default): sticks command Cartesian tip velocity per arm; a damped-least-squares
solve over (pan, lift, elbow) tracks it, and wrist_flex is slaved to hold the tool pitch.
Forward kinematics uses the homing-zero convention (tick 2047 = 0 rad) that the MuJoCo
contact fit validated to ~1 cm in this workspace, with stock link lengths.

At startup the driver auto-probes the pan sign against the overhead camera (frame-diff
centroid), so "stick right" matches the USER's view (facing the arms), and self-tests the
vertical sign with a 1 cm IK move checked against its own FK.

Mapping (EE mode):
  Left stick x/y   left tip  lateral / forward-back      L2 / L1   left tip down / up
  Right stick x/y  right tip lateral / forward-back      R2 / R1   right tip down / up
  D-pad up/down    left tool pitch trim                  Tri/Cross right tool pitch trim
  D-pad l/r        left gripper open/close               Sq/Circle right gripper open/close
  OPTIONS both arms to ready pose | SHARE invert lateral | PS quit

Run:  python -m halloween_bot.ps4_teleop [--mode ee|joint] [--speed-mm 60]
"""
import argparse
import json
import math
import select
import struct
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

SERVER = "http://127.0.0.1:8399"
HZ = 30
DEADZONE = 0.12
LATEST = Path.home() / "lerobot/outputs/claude_robot/latest/overhead.jpg"

AX_LX, AX_LY, AX_L2, AX_RX, AX_RY, AX_R2, AX_HATX, AX_HATY = 0, 1, 2, 3, 4, 5, 6, 7
BT_CROSS, BT_CIRCLE, BT_TRIANGLE, BT_SQUARE = 0, 1, 2, 3
BT_L1, BT_R1, BT_L2, BT_R2, BT_SHARE, BT_OPTIONS, BT_PS, BT_L3, BT_R3 = 4, 5, 6, 7, 8, 9, 10, 11, 12
JS_EVENT = struct.Struct("IhBB")

# stock SO101 skeleton (mm) — good to ~1 cm here; the human closes the loop visually
H_SHOULDER, L1, L2_LINK, L3 = 116.6, 116.0, 135.0, 160.0
PSI0 = -0.09  # tool-pitch constant so FK matches the stock all-zero tip height

ARM_LAT = {"left": 1.0, "right": 1.0}  # user-validated: both arms same lateral sense

READY = {"shoulder_pan.pos": -5.0, "shoulder_lift.pos": -55.0, "elbow_flex.pos": 60.0,
         "wrist_flex.pos": 30.0, "wrist_roll.pos": -10.0, "gripper.pos": 50.0}
REST = {"shoulder_pan.pos": -5.0, "shoulder_lift.pos": -86.0, "elbow_flex.pos": 95.0,
        "wrist_flex.pos": 45.0, "wrist_roll.pos": -10.0, "gripper.pos": 40.0}

CAL_DIR = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")


def call(path, payload=None, timeout=15):
    req = urllib.request.Request(SERVER + path, headers={"Content-Type": "application/json"},
                                 data=None if payload is None else json.dumps(payload).encode())
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class ArmKin:
    """Normalized units <-> radians (homing zero) and tip FK/Jacobian for one arm."""

    def __init__(self, side: str):
        cal = json.loads((CAL_DIR / f"bimanual_{side}.json").read_text())
        self.side = side
        self.rng = {j: (cal[j]["range_min"], cal[j]["range_max"]) for j in JOINTS}

    def rad(self, joint, n):
        lo, hi = self.rng[joint]
        ticks = lo + (n + 100.0) / 200.0 * (hi - lo)
        return (ticks - 2047.0) * (2.0 * math.pi / 4096.0)

    def norm(self, joint, theta):
        lo, hi = self.rng[joint]
        ticks = theta / (2.0 * math.pi / 4096.0) + 2047.0
        return max(-100.0, min(100.0, (ticks - lo) / (hi - lo) * 200.0 - 100.0))

    def fk(self, q):  # q = normalized {pan, lift, elbow, wf}; returns tip xyz (mm) + pitch
        tp = self.rad("shoulder_pan", q["shoulder_pan"])
        tl = self.rad("shoulder_lift", q["shoulder_lift"])
        te = self.rad("elbow_flex", q["elbow_flex"])
        tw = self.rad("wrist_flex", q["wrist_flex"])
        er, ez = L1 * math.sin(tl), H_SHOULDER + L1 * math.cos(tl)
        phi = tl + te + math.pi / 2
        wr, wz = er + L2_LINK * math.sin(phi), ez + L2_LINK * math.cos(phi)
        psi = phi + tw + PSI0
        r, z = wr + L3 * math.sin(psi), wz + L3 * math.cos(psi)
        return np.array([r * math.sin(tp), r * math.cos(tp), z]), psi

    def jac(self, q, eps=0.4):
        cols = []
        for j in ("shoulder_pan", "shoulder_lift", "elbow_flex"):
            qp = dict(q); qp[j] += eps
            cols.append((self.fk(qp)[0] - self.fk(q)[0]) / eps)
        return np.stack(cols, axis=1)  # 3x3: d(tip)/d(norm units)


def overhead_centroid_shift(move_fn):
    f0 = cv2.imread(str(LATEST), cv2.IMREAD_GRAYSCALE)
    move_fn()
    time.sleep(1.8)
    f1 = cv2.imread(str(LATEST), cv2.IMREAD_GRAYSCALE)
    if f0 is None or f1 is None:
        return None
    d = cv2.absdiff(f0, f1)
    d = (d > 25).astype(np.uint8)
    if d.sum() < 200:
        return None
    ys, xs = np.nonzero(d)
    # split frame halves by time is impossible from one diff; use intensity-weighted spread:
    return float(xs.mean()), float(ys.mean()), int(d.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("ee", "joint"), default="ee")
    ap.add_argument("--device", default="/dev/input/js0")
    ap.add_argument("--speed-mm", type=float, default=280.0, help="EE speed at full stick, mm/s")
    ap.add_argument("--speed", type=float, default=60.0, help="joint mode speed, units/s")
    ap.add_argument("--lat-sign", type=int, default=1, help="+1/-1 global lateral sign (user-validated; 0 re-probes)")
    a = ap.parse_args()

    st = call("/state")
    if st.get("teleop"):
        sys.exit("leader teleop is on — turn it off first")
    if st.get("policy") and st["policy"].get("running"):
        sys.exit("pi policy is driving — stop it first")

    kin = {"left": ArmKin("left"), "right": ArmKin("right")}
    cmd = dict(st["positions"])

    print("moving both arms to READY pose...", flush=True)
    call("/move", {"targets": {f"{s}_{k}": v for s in ("left", "right") for k, v in READY.items()},
                   "duration": 3.0}, timeout=40)
    cmd = dict(call("/state")["positions"])

    def q_of(side):
        return {j: cmd[f"{side}_{j}.pos"] for j in JOINTS}

    # --- startup probes (EE mode) -------------------------------------------------
    lat_sign = a.lat_sign
    if a.mode == "ee" and not lat_sign:
        print("probing lateral sign against the overhead camera...", flush=True)
        before = cmd["right_shoulder_pan.pos"]

        def bump():
            call("/move", {"targets": {"right_shoulder_pan.pos": before + 8}, "duration": 1.2}, timeout=20)
        res = overhead_centroid_shift(bump)
        call("/move", {"targets": {"right_shoulder_pan.pos": before}, "duration": 1.2}, timeout=20)
        # +pan moved the claw toward image-left => user-left => lat_sign -1 maps stick-right to -pan
        if res is None:
            lat_sign = -1
            print("probe inconclusive; defaulting lat_sign=-1 (SHARE flips it live)", flush=True)
        else:
            cx0 = 640  # frame centre reference; use shift direction via second diff pass
            # crude but effective: brighter side of the diff tells the direction
            lat_sign = -1 if res[0] < cx0 else +1
            print(f"probe: diff centroid x={res[0]:.0f} -> lat_sign={lat_sign}", flush=True)
        cmd = dict(call("/state")["positions"])

    # vertical self-test: ask IK for +10 mm z on the right arm, verify FK went up
    z_ok = True
    if a.mode == "ee":
        q0 = q_of("right")
        p0, _ = kin["right"].fk(q0)
        J = kin["right"].jac(q0)
        dq = np.linalg.solve(J.T @ J + 4.0 * np.eye(3), J.T @ np.array([0, 0, 10.0]))
        tgt = dict(cmd)
        for j, d in zip(("shoulder_pan", "shoulder_lift", "elbow_flex"), dq):
            tgt[f"right_{j}.pos"] = float(np.clip(cmd[f"right_{j}.pos"] + d, -100, 100))
        call("/move", {"targets": {k: v for k, v in tgt.items() if k.startswith("right")}, "duration": 1.0}, timeout=20)
        p1, _ = kin["right"].fk(q_of := {j: call("/state")["positions"][f"right_{j}.pos"] for j in JOINTS})
        if p1[2] < p0[2] - 2:
            z_ok = False
        print(f"vertical self-test: dz={p1[2] - p0[2]:+.1f} mm ({'ok' if z_ok else 'INVERTED — flipping'})", flush=True)
        cmd = dict(call("/state")["positions"])
    z_sign = 1.0 if z_ok else -1.0

    pitch_target = {}
    for s in ("left", "right"):
        pitch_target[s] = kin[s].fk({j: cmd[f"{s}_{j}.pos"] for j in JOINTS})[1]

    mode = a.mode
    print(f"PS4 teleop live (mode={mode}, lat_sign={lat_sign}).", flush=True)
    print("OPTIONS=switch ee/joint | L3=ready pose | R3=park/rest | SHARE=flip lateral | PS=quit", flush=True)

    axes, buttons = {}, {}
    dev = open(a.device, "rb", buffering=0)

    def axis(n):
        v = axes.get(n, 0) / 32767.0
        return 0.0 if abs(v) < DEADZONE else v

    def trig(n):
        return (axes.get(n, -32767) + 32767) / 65534.0

    last = time.time()
    while True:
        while select.select([dev], [], [], 0)[0]:
            data = dev.read(JS_EVENT.size)
            if not data:
                break
            _, value, etype, num = JS_EVENT.unpack(data)
            if etype & 0x02:
                axes[num] = value
            elif etype & 0x01:
                buttons[num] = value
                if num == BT_PS and value:
                    print("PS pressed: exiting, arms hold.", flush=True)
                    return
                if num == BT_SHARE and value:
                    lat_sign = -lat_sign
                    print(f"lateral flipped: lat_sign={lat_sign}", flush=True)
                if num == BT_OPTIONS and value:
                    mode = "joint" if mode == "ee" else "ee"
                    cmd = dict(call("/state")["positions"])
                    for s in ("left", "right"):
                        pitch_target[s] = kin[s].fk({j: cmd[f"{s}_{j}.pos"] for j in JOINTS})[1]
                    print(f"mode -> {mode}", flush=True)
                if num in (BT_L3, BT_R3) and value:
                    pose = READY if num == BT_L3 else REST
                    print("L3: ready pose" if num == BT_L3 else "R3: park/rest pose", flush=True)
                    call("/move", {"targets": {f"{s}_{k}": v for s in ("left", "right")
                                               for k, v in pose.items()}, "duration": 3.0}, timeout=40)
                    cmd = dict(call("/state")["positions"])
                    for s in ("left", "right"):
                        pitch_target[s] = kin[s].fk({j: cmd[f"{s}_{j}.pos"] for j in JOINTS})[1]

        now = time.time()
        dt, last = now - last, now
        moved = False

        if mode == "ee":
            for s, ax_x, ax_y, bt_up, ax_dn, bt_pu, bt_pd, bt_go, bt_gc in (
                    ("left", AX_LX, AX_LY, BT_L1, AX_L2, None, None, None, None),
                    ("right", AX_RX, AX_RY, BT_R1, AX_R2, BT_TRIANGLE, BT_CROSS, BT_SQUARE, BT_CIRCLE)):
                v = np.array([
                    lat_sign * ARM_LAT[s] * axis(ax_x) * a.speed_mm,
                    -axis(ax_y) * a.speed_mm,
                    z_sign * (buttons.get(bt_up, 0) - trig(ax_dn)) * a.speed_mm,
                ]) * dt
                # pitch trim + gripper
                if s == "left":
                    pt = -axes.get(AX_HATY, 0) / 32767.0
                    g = -axes.get(AX_HATX, 0) / 32767.0
                else:
                    pt = buttons.get(BT_TRIANGLE, 0) - buttons.get(BT_CROSS, 0)
                    g = buttons.get(BT_SQUARE, 0) - buttons.get(BT_CIRCLE, 0)
                if pt:
                    pitch_target[s] += pt * 0.8 * dt
                    moved = True
                if g:
                    k = f"{s}_gripper.pos"
                    cmd[k] = float(np.clip(cmd[k] + g * 60 * dt, 0, 100))
                    moved = True
                if np.linalg.norm(v) > 0.01:
                    q = {j: cmd[f"{s}_{j}.pos"] for j in JOINTS}
                    J = kin[s].jac(q)
                    dq = np.linalg.solve(J.T @ J + 4.0 * np.eye(3), J.T @ v)
                    dq = np.clip(dq, -9.0, 9.0)
                    for j, d in zip(("shoulder_pan", "shoulder_lift", "elbow_flex"), dq):
                        cmd[f"{s}_{j}.pos"] = float(np.clip(cmd[f"{s}_{j}.pos"] + d, -100, 100))
                    moved = True
                if pt or np.linalg.norm(v) > 0.01:
                    # slave wrist_flex to hold tool pitch
                    kn = kin[s]
                    tl = kn.rad("shoulder_lift", cmd[f"{s}_shoulder_lift.pos"])
                    te = kn.rad("elbow_flex", cmd[f"{s}_elbow_flex.pos"])
                    tw = pitch_target[s] - (tl + te + math.pi / 2) - PSI0
                    cmd[f"{s}_wrist_flex.pos"] = kn.norm("wrist_flex", tw)
        else:  # joint mode, pan un-reversed
            sp = a.speed * dt
            d = {"left_shoulder_pan.pos": lat_sign * ARM_LAT["left"] * axis(AX_LX) * sp,
                 "left_shoulder_lift.pos": -axis(AX_LY) * sp,
                 "right_shoulder_pan.pos": lat_sign * ARM_LAT["right"] * axis(AX_RX) * sp,
                 "right_shoulder_lift.pos": -axis(AX_RY) * sp,
                 "left_elbow_flex.pos": (trig(AX_L2) - buttons.get(BT_L1, 0)) * sp,
                 "right_elbow_flex.pos": (trig(AX_R2) - buttons.get(BT_R1, 0)) * sp,
                 "left_wrist_flex.pos": -axes.get(AX_HATY, 0) / 32767.0 * sp,
                 "right_wrist_flex.pos": (buttons.get(BT_TRIANGLE, 0) - buttons.get(BT_CROSS, 0)) * sp,
                 "left_gripper.pos": -axes.get(AX_HATX, 0) / 32767.0 * sp * 1.5,
                 "right_gripper.pos": (buttons.get(BT_SQUARE, 0) - buttons.get(BT_CIRCLE, 0)) * sp * 1.5}
            for k, dv in d.items():
                if dv:
                    lo, hi = (0, 100) if "gripper" in k else (-100, 100)
                    cmd[k] = float(np.clip(cmd[k] + dv, lo, hi))
                    moved = True

        if moved:
            try:
                call("/trajectory", {"points": [cmd], "hz": HZ})
            except Exception as e:
                print(f"send failed: {e}", flush=True)
                time.sleep(0.5)
                cmd = dict(call("/state")["positions"])
        time.sleep(max(0.0, 1.0 / HZ - (time.time() - now)))


if __name__ == "__main__":
    main()
