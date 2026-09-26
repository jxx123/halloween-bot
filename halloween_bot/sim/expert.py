"""Scripted pick-and-place expert for the MuJoCo twin: the demonstrator for sim data collection.

Uses privileged sim state (toy/basket poses) to plan gripper-tip waypoints, solves IK for the arm
nearest the toy, and executes through the real command path (SimEngine.send_action: 30 Hz, ±20-unit
per-tick clamp, fitted servo dynamics), so recorded actions look like what a policy would send.

Plan (claws pointing down):  rest -> above toy -> down around toy -> close -> lift -> above basket
-> lower -> open -> up -> rest.  The other arm holds its rest pose.
"""
import numpy as np
import mujoco
from scipy.optimize import least_squares

from .calib import KEYS, SIDES, joint_name
from .engine import CONTROL_HZ, REST, SimEngine
from .model import BASKET, TOYS

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")  # position + claw pitch
OPEN, CLOSED = 60.0, 0.0  # gripper: 60 ≈ 9 cm between jaw tips (a wider-open moving jaw hangs into the table); 0 squeezes
ROLL = -10.0  # wrist roll held at the rig's neutral
HOVER_DZ = 0.08  # above the toy's head before descending
CARRY_Z = 0.23  # travel height (basket rim 0.12 + a hanging toy)
RELEASE_Z = 0.19
DOWN_WEIGHT = 0.03  # m per unit of claw-axis error: position first, pointing-down second
GRASP_GAP = OPEN  # jaw midpoint of the OPEN gripper = grasp point: both jaws clear a ~7 cm toy on the way down


class Expert:
    def __init__(self, engine: SimEngine, hz: float = CONTROL_HZ):
        self.eng, self.m, self.hz = engine, engine.model, hz
        self.ik_data = mujoco.MjData(self.m)
        self._qadr = {k: self.m.joint(joint_name(k)).qposadr[0] for k in KEYS}
        self._grasp_local = {side: self._grasp_offset(side) for side in SIDES}

    def _grasp_offset(self, side: str) -> np.ndarray:
        """Midpoint between the jaw tips at GRASP_GAP, in the gripper body frame: aim THIS at the toy
        (the gripperframe site is the fixed jaw's tip, i.e. one side of the grasp)."""
        d = self.ik_data
        d.qpos[:] = self.eng.data.qpos
        k = f"{side}_gripper.pos"
        d.qpos[self._qadr[k]] = self.eng.calib.to_rad(k, GRASP_GAP)
        mujoco.mj_kinematics(self.m, d)
        mid = 0.5 * (d.geom_xpos[self.m.geom(f"{side}_fixed_jaw_sph_tip1").id]
                     + d.geom_xpos[self.m.geom(f"{side}_moving_jaw_sph_tip1").id])
        body = self.m.body(f"{side}_gripper").id
        return d.xmat[body].reshape(3, 3).T @ (mid - d.xpos[body])

    # ---- kinematics -------------------------------------------------------------------
    def _tip_axis(self, data, side):
        body = self.m.body(f"{side}_gripper").id
        tip = data.xpos[body] + data.xmat[body].reshape(3, 3) @ self._grasp_local[side]  # grasp point
        roll = data.xanchor[self.m.joint(f"{side}_wrist_roll").id]
        axis = tip - roll
        return tip, axis / np.linalg.norm(axis)

    def tip(self, side: str) -> np.ndarray:
        return self._tip_axis(self.eng.data, side)[0]

    def tip_after(self, q: dict, side: str):
        """Tip position and claw axis if the arm were at normalized joints q (kinematics only)."""
        d = self.ik_data
        d.qpos[:] = self.eng.data.qpos
        for k, v in q.items():
            d.qpos[self._qadr[k]] = self.eng.calib.to_rad(k, v)
        mujoco.mj_kinematics(self.m, d)
        return self._tip_axis(d, side)

    def ik(self, side: str, target: np.ndarray, seed: dict | None = None) -> dict:
        """Normalized pan/lift/elbow/wrist_flex putting the claw tip at target, claws as close to down as possible."""
        keys = [f"{side}_{j}.pos" for j in ARM_JOINTS]
        cal, d = self.eng.calib, self.ik_data
        seed = seed or self.eng.read_positions()
        x0 = np.array([cal.to_rad(k, seed[k]) for k in keys])
        lo = np.array([self.m.jnt_range[self.m.joint(joint_name(k)).id][0] for k in keys])
        hi = np.array([self.m.jnt_range[self.m.joint(joint_name(k)).id][1] for k in keys])
        adr = [self._qadr[k] for k in keys]
        base = self.eng.data.qpos.copy()

        def res(x):
            d.qpos[:] = base
            d.qpos[adr] = x
            mujoco.mj_kinematics(self.m, d)
            tip, axis = self._tip_axis(d, side)
            return np.concatenate([tip - target, DOWN_WEIGHT * (axis - np.array([0.0, 0.0, -1.0]))])

        best = None
        for start in (x0, np.clip(x0 + np.array([0, 0.4, -0.4, 0.2]), lo, hi)):
            sol = least_squares(res, np.clip(start, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), xtol=1e-10, ftol=1e-10)
            if best is None or sol.cost < best.cost:
                best = sol
        return {k: float(np.clip(cal.to_norm(k, v), -100, 100)) for k, v in zip(keys, best.x)}

    # ---- plan + execute ---------------------------------------------------------------
    def toy_positions(self) -> dict[str, np.ndarray]:
        """Grasp targets: each toy's HEAD (the sphere geom), upright or lying down, like a hand grabs a plush."""
        out = {}
        for name in TOYS:
            body = self.m.body(name).id
            head = [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == body
                    and self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE][0]
            out[name] = self.eng.data.geom_xpos[head].copy()
        return out

    def settle(self, seconds: float = 1.0):
        """Let freshly reset toys drop and roll to rest before planning (not recorded)."""
        for _ in range(int(seconds * self.hz)):
            self.eng.send_action(dict(REST))
            self.eng.step(1.0 / self.hz)

    def run(self, rng: np.random.Generator, on_tick=None, toy: str | None = None, settle: float = 1.0) -> dict:
        """One pick-and-place. on_tick(action: dict) is called once per control tick, BEFORE the action
        executes (the recorder snapshots the observation there)."""
        if settle:
            self.settle(settle)
        toys = self.toy_positions()
        toy = toy or str(rng.choice(sorted(toys)))
        p_toy = toys[toy]
        side = "right" if p_toy[1] < 0 else "left"
        bx, by = BASKET["pos"]
        hold = dict(REST)
        frames = 0

        def send(targets: dict, grip: float):
            nonlocal frames
            action = {**hold, **targets, f"{side}_gripper.pos": grip, f"{side}_wrist_roll.pos": ROLL}
            if on_tick is not None:
                on_tick(action)  # recorder: obs is read BEFORE this action executes
            self.eng.send_action(action)
            self.eng.step(1.0 / self.hz)
            frames += 1
            hold.update(action)

        def move(to: dict, grip: float, seconds: float, settle: float = 1.0, tol: float = 3.0):
            start = {k: hold[k] for k in to}
            n = max(1, int(seconds * self.hz))
            for i in range(1, n + 1):
                a = 0.5 - 0.5 * np.cos(np.pi * i / n)  # smooth start/stop
                send({k: start[k] + (to[k] - start[k]) * a for k in to}, grip)
            # the fitted servos are soft and lag: hold the goal until the arm catches up (bounded)
            for _ in range(int(settle * self.hz)):
                present = self.eng.read_positions()
                if all(abs(present[k] - to[k]) < tol for k in to):
                    break
                send({}, grip)

        def line(p_from, p_to, grip, seconds, pieces=4):
            q = None
            for s in np.linspace(0, 1, pieces + 1)[1:]:
                q = self.ik(side, p_from + (p_to - p_from) * s, seed=q or hold)
                move(q, grip, seconds / pieces)
            return q

        above_toy = p_toy + np.array([0.0, 0.0, HOVER_DZ])
        over_basket = np.array([bx, by, CARRY_Z])
        in_basket = np.array([bx, by, RELEASE_Z])

        move(self.ik(side, above_toy), OPEN, 2.0)
        p_toy = self.toy_positions()[toy]  # closed loop: re-read before descending
        if np.linalg.norm(p_toy + [0, 0, HOVER_DZ] - above_toy) > 0.01:
            above_toy = p_toy + np.array([0.0, 0.0, HOVER_DZ])
            move(self.ik(side, above_toy, seed=hold), OPEN, 0.6)
        at_toy = p_toy.copy()
        lifted = np.array([p_toy[0], p_toy[1], CARRY_Z])
        line(above_toy, at_toy, OPEN, 1.2)
        move({}, CLOSED, 0.8)  # squeeze
        line(at_toy, lifted, CLOSED, 1.2)
        grasped = self.eng.data.xpos[self.m.body(toy).id][2] > 0.08
        move(self.ik(side, over_basket, seed=hold), CLOSED, 1.8)
        line(over_basket, in_basket, CLOSED, 0.8, pieces=2)
        move({}, OPEN, 0.6)
        line(in_basket, over_basket, OPEN, 0.6, pieces=2)
        move({k: REST[k] for k in REST if k.startswith(side) and "gripper" not in k and "roll" not in k}, REST[f"{side}_gripper.pos"], 2.0)
        for _ in range(int(0.5 * self.hz)):  # let the toy settle, still recording
            send({}, REST[f"{side}_gripper.pos"])
        p_end = self.eng.data.xpos[self.m.body(toy).id]
        in_basket_now = np.hypot(p_end[0] - bx, p_end[1] - by) < BASKET["radius"] - 0.01 and p_end[2] < BASKET["height"]
        return {"success": bool(in_basket_now), "toy": toy, "arm": side, "grasped": bool(grasped), "frames": frames,
                "toy_end": [round(float(v), 3) for v in p_end]}
