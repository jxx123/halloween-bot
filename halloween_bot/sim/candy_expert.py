"""Scripted demonstrator for the candy scene: the three data-collection tasks.

  pick_place   one arm picks a candy from the bowl and puts it on the plate on its side
  handover     arm A picks a long candy, presents it claws-forward, arm B grabs the free end, A lets go,
               B puts it on the plate on B's side
  give_human   a hand reaches in; the arm picks a candy and drops it into the palm; the hand takes it away

Same execution path as the toy Expert: SimEngine.send_action at 30 Hz with the ±20 per-tick clamp and
the fitted servos. Privileged state (candy poses, plate, hand) plans the motion; the recorder only
sees the cameras and joint states. Wrist roll is planned too: the jaws close across a long candy's
short side.
"""
import math

import mujoco
import numpy as np

from . import candy as C
from .calib import SIDES
from .engine import REST, SimEngine
from .expert import ARM_JOINTS, Expert

OPEN = 40.0  # ≈ 62 mm between the jaw tips: clears the widest candy (42 mm) by 1 cm a side
MM_PER_UNIT, GAP0_MM = 1.43, 4.1  # jaw-tip gap ≈ 4.1 mm + 1.43 mm per gripper unit
SQUEEZE_MM = 20.0  # close this far past the candy's width: the fitted gripper servo is soft (~1 N per 9 mm)
FIXED_CLEAR = 0.006  # m between the fixed jaw tip (its innermost point) and the candy's side on the way down
FLOOR_CLEAR = 0.0025  # m every jaw part keeps off the bowl floor at the bottom of the descent (the open moving
# jaw hangs lowest; pressed on the floor, friction pins it open against the weak gripper servo)
MOVING_CLEAR = 0.016  # m the open moving jaw's tip clears the candy's far side by (its face leans ~7 mm inward above the tip)
MIN_ROOM = 0.004  # m both jaw paths must clear other candy and the bowl wall for a candy to be chosen
HOVER = 0.07  # above the grasp point before descending
CARRY_Z = 0.12  # travel height: clears the bowl rim (37 mm) with a candy hanging below the jaws
HANDOVER = np.array([0.25, 0.0, 0.15])  # where arm A holds a candy out for B (y shifted toward A)
HANDOVER_Y = 0.02
HANDOVER_TILT_DEG = 25.0  # each gripper leans toward its own side so their bodies clear
TILT_WEIGHT = 0.06  # IK weight of that lean: a soft preference (position first; 0.3 cost ~15 mm of reach)
FWD_WEIGHT = 0.12  # IK weight of B's claws-forward reach (it must stay roughly level to slide in along the candy)
BASE_XY = {"left": np.array([0.0, 0.175]), "right": np.array([0.0, -0.175])}
TASKS = ("pick_place", "handover", "give_human")
# which arm picks: "bowl" = side of the jittered bowl centre + a coin flip within 1 cm (sim_candy_v0/v1: ~15% of
# demos then teach two answers for one picture); "midline" = side of the table's fixed centre line, the right arm
# for anything within MIDLINE_TIE of it (a consistent rule, like a person's)
ARM_RULES = ("bowl", "midline")
MIDLINE_TIE = 0.015
LONG = [n for n, c in C.CANDIES.items() if c["handover"] is not None]  # long enough for two grippers


def instruction(task: str, candy: str) -> str:
    words = C.CANDIES[candy]["words"]
    return {"pick_place": f"Pick up the {words} and put it on the plate.",
            "handover": f"Pick up the {words}, hand it to the other arm, and put it on the plate.",
            "give_human": f"Give the {words} to the person."}[task]


class NoCandy(RuntimeError):
    """This bowl layout has no candy the jaws can fit around for the task: reset and try another."""


def other(side: str) -> str:
    return "left" if side == "right" else "right"


class CandyExpert(Expert):
    grasp_gap = OPEN

    def __init__(self, engine: SimEngine, hz: float = 30.0, arm_rule: str = "bowl"):
        super().__init__(engine, hz)
        if arm_rule not in ARM_RULES:
            raise ValueError(f"arm_rule must be one of {ARM_RULES}")
        self.arm_rule = arm_rule
        self.instruction = ""
        self._tips = {s: (self.m.geom(f"{s}_fixed_jaw_sph_tip1").id, self.m.geom(f"{s}_moving_jaw_sph_tip1").id)
                      for s in SIDES}
        self.opening = {s: OPEN for s in SIDES}  # per-arm approach opening (set per candy)
        self._jaw_geoms = {s: [g for g in range(self.m.ngeom)
                               if self.m.body(self.m.geom_bodyid[g]).name in (f"{s}_gripper", f"{s}_moving_jaw_so101_v1")
                               and (self.m.geom_contype[g] or self.m.geom_conaffinity[g])] for s in SIDES}
        self._floor = self.m.geom("bowl_floor").id
        self._mount = {s: self.m.geom(f"{s}_camera_box2").id for s in SIDES}
        self._begin(None)

    def _jaw_tips_local(self, side: str, opening: float):
        d = self.ik_data
        d.qpos[:] = self.eng.data.qpos
        d.qpos[self._qadr[f"{side}_gripper.pos"]] = self.eng.calib.to_rad(f"{side}_gripper.pos", opening)
        mujoco.mj_kinematics(self.m, d)
        b = self.m.body(f"{side}_gripper").id
        R, x = d.xmat[b].reshape(3, 3), d.xpos[b]
        f, mv = self._tips[side]
        return R.T @ (d.geom_xpos[f] - x), R.T @ (d.geom_xpos[mv] - x)

    @staticmethod
    def extent(name: str, d, u) -> float:
        """Half-extent of the candy along the jaw direction d (long axis u), for a box-like footprint."""
        info = C.CANDIES[name]
        if u is None:
            return info["hw"]
        c = abs(float(np.dot(d[:2], u[:2])) / (np.linalg.norm(d[:2]) * np.linalg.norm(u[:2]) + 1e-9))
        return c * info["half_len"] + math.sqrt(max(0.0, 1 - c * c)) * info["hw"]

    def set_grasp(self, side: str, name: str, hw: float | None = None):
        """Aim the fixed jaw just beside the candy (its half-width + FIXED_CLEAR from the candy centre), so the
        moving jaw sweeps it in instead of shoving it across the bowl."""
        hw = C.CANDIES[name]["hw"] if hw is None else hw
        self.opening[side] = self.open_for(hw)
        f, mv = self._jaw_tips_local(side, self.opening[side])
        self._grasp_local[side] = f + (hw + FIXED_CLEAR) * (mv - f) / np.linalg.norm(mv - f)

    @staticmethod
    def open_for(hw: float) -> float:
        """Just wide enough: the fixed jaw FIXED_CLEAR from one side, the moving jaw MOVING_CLEAR past the other."""
        return float(np.clip((1000 * (2 * hw + FIXED_CLEAR + MOVING_CLEAR) - GAP0_MM) / MM_PER_UNIT, 15.0, OPEN))

    @staticmethod
    def close_to(name: str, hw: float | None = None) -> float:
        hw = C.CANDIES[name]["hw"] if hw is None else hw
        return max(0.0, (2000 * hw - SQUEEZE_MM - GAP0_MM) / MM_PER_UNIT)

    # ---- low-level motion (records through on_tick) -------------------------------------
    def _begin(self, on_tick):
        self.on_tick, self.hold, self.frames, self.anims = on_tick, dict(REST), 0, []

    def send(self, targets: dict):
        action = {**self.hold, **targets}
        self.anims = [a for a in self.anims if next(a, StopIteration) is not StopIteration]
        if self.on_tick is not None:
            self.on_tick(action)  # the recorder snapshots the observation BEFORE this action executes
        self.eng.send_action(action)
        self.eng.step(1.0 / self.hz)
        self.frames += 1
        self.hold.update(action)

    def move(self, to: dict, seconds: float, settle: float = 1.0, tol: float = 3.0):
        start = {k: self.hold[k] for k in to}
        n = max(1, int(seconds * self.hz))
        for i in range(1, n + 1):
            a = 0.5 - 0.5 * math.cos(math.pi * i / n)
            self.send({k: start[k] + (to[k] - start[k]) * a for k in to})
        track = [k for k in to if "gripper" not in k]  # a gripper closing on candy stalls short of its goal
        for _ in range(int(settle * self.hz)):
            present = self.eng.read_positions()
            if all(abs(present[k] - to[k]) < tol for k in track):
                break
            self.send({})

    def wait(self, seconds: float):
        for _ in range(int(seconds * self.hz)):
            self.send({})

    def grip(self, side: str, value: float, seconds: float = 0.6):
        self.move({f"{side}_gripper.pos": value}, seconds, settle=0.0)

    def pose(self, side: str, target, axis=(0.0, 0.0, -1.0), weight: float = 0.03, roll: float | None = None,
             seed: dict | None = None) -> dict:
        roll = self.hold[f"{side}_wrist_roll.pos"] if roll is None else roll
        q = self.ik(side, np.asarray(target, float), seed=seed or self.hold, axis=axis, axis_weight=weight, roll=roll)
        return {**q, f"{side}_wrist_roll.pos": roll}

    def line(self, side: str, p_from, p_to, seconds: float, pieces: int = 4, **kw):
        q = None
        for s in np.linspace(0, 1, pieces + 1)[1:]:
            q = self.pose(side, np.asarray(p_from) + (np.asarray(p_to) - np.asarray(p_from)) * s, seed=q, **kw)
            self.move(q, seconds / pieces)
        return q

    def reach(self, side: str, target, tol: float = 0.0015, rounds: int = 4, **kw):
        """Servo the grasp point onto target: the fitted servos sag under gravity (several mm at the claw), so
        re-aim by the measured error, like a teleoperator watching the wrist camera."""
        target = np.asarray(target, float)
        aim = target.copy()
        for _ in range(rounds):
            err = self.tip(side) - target
            if np.linalg.norm(err) < tol:
                break
            aim = aim - err
            self.move(self.pose(side, aim, **kw), 0.25, settle=0.5, tol=0.5)

    def rest(self, side: str, seconds: float = 1.8):
        self.move({f"{side}_{j}.pos": REST[f"{side}_{j}.pos"] for j in (*ARM_JOINTS, "wrist_roll", "gripper")}, seconds)

    # ---- wrist roll -----------------------------------------------------------------------
    def floor_clearance(self, q: dict, side: str, closed: float = 0.0) -> float:
        """Smallest distance between the jaws and the bowl floor at arm pose q, over the whole close from
        self.opening down to `closed`: the moving jaw swings like a pendulum and dips lowest mid-close."""
        d = self.ik_data
        d.qpos[:] = self.eng.data.qpos
        d.mocap_pos[:], d.mocap_quat[:] = self.eng.data.mocap_pos, self.eng.data.mocap_quat
        for k, v in q.items():
            d.qpos[self._qadr[k]] = self.eng.calib.to_rad(k, v)
        ft, best = np.zeros(6), np.inf
        g_key = f"{side}_gripper.pos"
        for g_val in np.linspace(self.opening[side], closed, 6):
            d.qpos[self._qadr[g_key]] = self.eng.calib.to_rad(g_key, g_val)
            mujoco.mj_kinematics(self.m, d)
            best = min(best, min(mujoco.mj_geomDistance(self.m, d, g, self._floor, 0.05, ft) for g in self._jaw_geoms[side]))
        return float(best)

    def jaw_dir(self, q: dict, side: str) -> np.ndarray:
        """Unit vector from the fixed jaw tip to the moving jaw tip (the closing direction) at arm pose q."""
        d = self.ik_data
        d.qpos[:] = self.eng.data.qpos
        for k, v in {**q, f"{side}_gripper.pos": self.opening[side]}.items():
            d.qpos[self._qadr[k]] = self.eng.calib.to_rad(k, v)
        mujoco.mj_kinematics(self.m, d)
        f, mv = self._tips[side]
        v = d.geom_xpos[mv] - d.geom_xpos[f]
        return v / np.linalg.norm(v)

    def roll_for(self, side: str, q: dict, score) -> float:
        """Normalized wrist_roll maximizing score(jaw_dir) at arm pose q; ties go to the roll nearest now."""
        now = self.hold[f"{side}_wrist_roll.pos"]
        grid = np.arange(-98.0, 98.1, 1.0)
        vals = np.array([score(self.jaw_dir({**q, f"{side}_wrist_roll.pos": r}, side)) for r in grid])
        good = grid[vals >= vals.max() - 0.01]
        return float(good[np.argmin(np.abs(good - now))])

    def room(self, center, d, hw: float, gap_m: float, avoid: str | None) -> float:
        """Clearance (m, capped at 2 cm) between both jaws' paths and the other candies in the bowl and its
        wall, for jaws closing along horizontal d across a candy at center."""
        if avoid is None:
            return 0.0
        d = np.array([d[0], d[1], 0.0]) / (np.linalg.norm(d[:2]) + 1e-9)
        pts = [center - (hw + FIXED_CLEAR) * d] + [center + t * d for t in np.linspace(hw, gap_m - hw - FIXED_CLEAR, 4)]
        m, data = self.m, self.eng.data
        best = 0.02
        for other in self.eng.layout.get("candies", []):
            if other == avoid or not C.in_bowl(m, data, other):
                continue
            o, ou = C.candy_pose(m, data, other)
            info = C.CANDIES[other]
            for p in pts:  # distance to the other candy's long-axis segment, minus its half-width
                t = np.clip((p - o)[:2] @ ou[:2], -info["half_len"], info["half_len"])
                best = min(best, np.linalg.norm((p - o)[:2] - t * ou[:2]) - info["hw"])
        bowl = np.asarray(self.eng.layout.get("bowl", center[:2]))
        for p in pts:
            best = min(best, C.BOWL["floor_r"] - np.linalg.norm(p[:2] - bowl))
        return float(best)

    def best_room(self, name: str) -> float:
        """Most room the jaws can get around this candy over the usable closing directions (no IK needed)."""
        p, u = self.candy_target(name)
        center = np.array([p[0], p[1], 0.0])
        if u is None:
            dirs = [np.array([math.cos(a), math.sin(a), 0.0]) for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)]
        else:
            perp = np.array([-u[1], u[0], 0.0]) / (np.linalg.norm(u[:2]) + 1e-9)
            dirs = [perp, -perp]
        hw = C.CANDIES[name]["hw"]
        gap = (GAP0_MM + MM_PER_UNIT * self.open_for(hw)) / 1000
        return max(self.room(center, d, hw, gap, name) for d in dirs)

    def aligned_pose(self, side: str, target, u, name: str | None = None, hw: float | None = None, **kw) -> dict:
        """Claws-down pose at target, jaws closing across the horizontal direction u (None: any direction),
        on whichever side leaves both jaws the most room from neighbouring candy (when name is given)."""
        q = self.pose(side, target, **kw)
        if u is not None:
            u = np.array([u[0], u[1], 0.0]) / (np.linalg.norm(u[:2]) + 1e-9)
        hw = C.CANDIES[name]["hw"] if (name and hw is None) else (hw or 0.0)
        gap = (GAP0_MM + MM_PER_UNIT * self.opening[side]) / 1000
        center = np.array([target[0], target[1], 0.0])

        def score(d):  # perpendicular first (a few degrees of yaw error widens a bar a lot), then room
            perp = 0.0 if u is None else -8.0 * abs(d @ u)
            return perp - 0.2 * abs(d[2]) + 2.0 * self.room(center, d, hw, gap, name)

        for _ in range(2):  # the grasp point sits off the roll axis: re-solve at the new roll
            roll = self.roll_for(side, q, score)
            q = self.pose(side, target, roll=roll, **kw)
        return q

    # ---- skills -----------------------------------------------------------------------------
    def candy_target(self, name: str, local=None):
        info = C.CANDIES[name]
        p = C.candy_point(self.m, self.eng.data, name, info["grasp"] if local is None else local)
        _, u = C.candy_pose(self.m, self.eng.data, name)
        return p, (None if name in C.ROUND else u)

    def pick(self, side: str, name: str, local=None) -> bool:
        """Pick `name` from the bowl at its local grasp point (or `local`) and lift it to carry height."""
        self.set_grasp(side, name)
        p, u = self.candy_target(name, local)
        above = p + [0, 0, HOVER]
        q = self.aligned_pose(side, above, u, name)
        self.set_grasp(side, name, self.extent(name, self.jaw_dir(q, side), u))
        q = self.pose(side, above, roll=q[f"{side}_wrist_roll.pos"])
        self.move({**q, f"{side}_gripper.pos": self.opening[side]}, 2.0)
        p2, u2 = self.candy_target(name, local)  # closed loop: candy may have rolled while the arm came over
        if np.linalg.norm(p2 - p) > 0.008:
            p, u = p2, u2
            above = p + [0, 0, HOVER]
            self.move(self.aligned_pose(side, above, u, name), 0.7)
        # round candy: jaws at/above the equator press it down; below it they pop it up and out
        grasp = p + [0, 0, 0.002 if (name in C.ROUND or name == "lollipop") else -0.001]
        roll = self.hold[f"{side}_wrist_roll.pos"]
        for _ in range(3):  # raise the grasp point until every jaw part clears the bowl floor
            gap = self.floor_clearance(self.pose(side, grasp, roll=roll), side, self.close_to(name))
            if gap >= FLOOR_CLEAR - 1e-4:
                break
            grasp[2] += FLOOR_CLEAR - gap
        self.line(side, above, grasp, 1.1)
        self.reach(side, grasp)
        self.grip(side, self.close_to(name), 0.6)
        self.wait(0.25)
        self.line(side, grasp, [grasp[0], grasp[1], CARRY_Z], 0.9, pieces=3)
        held = C.candy_point(self.m, self.eng.data, name, C.CANDIES[name]["grasp"] if local is None else local)
        return bool(held[2] > 0.06 and np.linalg.norm(held - self.tip(side)) < 0.03)

    def put_down(self, side: str, spot, name: str, drop: float = 0.02, axis=(0.0, 0.0, -1.0), weight: float = 0.03):
        """Carry over spot (the surface point), lower, open, back off."""
        spot = np.asarray(spot, float)
        kw = dict(axis=axis, weight=weight)
        over = np.array([spot[0], spot[1], max(CARRY_Z, spot[2] + 0.08)])
        self.move(self.pose(side, over, **kw), 1.6)
        low = spot + [0, 0, C.CANDIES[name]["rest_z"] + drop]
        self.line(side, over, low, 0.9, pieces=3, **kw)
        self.grip(side, OPEN, 0.5)
        self.line(side, low, low + [0, 0, 0.07], 0.6, pieces=2, **kw)

    # ---- episodes ---------------------------------------------------------------------------
    def settle(self, seconds: float = 1.5):
        for _ in range(int(seconds * self.hz)):
            self.eng.send_action(dict(REST))
            self.eng.step(1.0 / self.hz)

    def setup(self, task: str, rng: np.random.Generator) -> dict:
        """Choose candy/arm, place the plate or hand, set the instruction. Not recorded."""
        self.settle()
        m, d = self.m, self.eng.data
        in_bowl = [n for n in self.eng.layout.get("candies", []) if C.in_bowl(m, d, n)]
        pool = [n for n in in_bowl if n in LONG] if task == "handover" else in_bowl
        pool = [n for n in pool if self.best_room(n) >= MIN_ROOM]  # the jaws must fit around it
        if not pool:
            raise NoCandy(f"no pickable candy for {task} in the bowl: {in_bowl}")
        name = str(rng.choice(sorted(pool)))
        p, _ = C.candy_pose(m, d, name)
        if self.arm_rule == "midline":  # the table's fixed centre line; near it always the right arm (no coin flips)
            side = "left" if p[1] > MIDLINE_TIE else "right"
        else:  # v0/v1 data: the candy's side of the (jittered) bowl centre, a coin flip within 1 cm of it
            bowl_y = self.eng.layout["bowl"][1]
            side = ("right" if p[1] < bowl_y else "left") if abs(p[1] - bowl_y) > 0.01 else str(rng.choice(SIDES))
        ep = {"task": task, "candy": name, "arm": side}
        if task in ("pick_place", "handover"):
            placer = side if task == "pick_place" else other(side)
            s = 1.0 if placer == "left" else -1.0
            plate = [rng.uniform(0.12, 0.30), s * rng.uniform(0.27, 0.33), 0.0]
            C.set_mocap(m, d, "plate", plate)
            ep.update(plate=plate, placer=placer)
        else:
            yaw = rng.uniform(-math.radians(45), math.radians(45))
            # out past the bowl rim: the fingers (8 cm toward the robot) must not hang over the candy
            target = np.array([rng.uniform(0.36, 0.40), rng.uniform(-0.10, 0.10), rng.uniform(0.10, 0.15)])
            out = np.array([math.cos(yaw), math.sin(yaw), 0.0])  # forearm direction: toward the person
            ep.update(hand=target.tolist(), hand_yaw=yaw, hand_from=(target + 0.16 * out + [0, 0, 0.06]).tolist())
            C.set_hand(m, d, ep["hand_from"], yaw, teleport=True)
        mujoco.mj_forward(m, d)
        self.instruction = ep["instruction"] = instruction(task, name)
        return ep

    def _glide(self, a, b, yaw: float, seconds: float):
        """Move the hand's target from a to b (smooth start/stop), one control tick per step."""
        n = max(1, int(seconds * self.hz))
        for i in range(1, n + 1):
            s = 0.5 - 0.5 * math.cos(math.pi * i / n)
            C.set_hand(self.m, self.eng.data, np.asarray(a) + (np.asarray(b) - np.asarray(a)) * s, yaw)
            yield

    def run(self, task: str, rng: np.random.Generator, on_tick=None, ep: dict | None = None) -> dict:
        ep = ep or self.setup(task, rng)
        self._begin(on_tick)
        name, side = ep["candy"], ep["arm"]
        m, d = self.m, self.eng.data
        held = True
        give_x = C.CANDIES[name]["handover"][0] if task == "handover" else None
        held = self.pick(side, name, None if give_x is None else (give_x, 0.0, 0.0))
        if task == "give_human":  # the person reaches in once the robot has lifted a candy
            self.anims.append(self._glide(ep["hand_from"], ep["hand"], ep["hand_yaw"], 1.4))
        if task == "pick_place":
            self.put_down(side, ep["plate"], name)
            self.rest(side)
        elif task == "handover":
            held &= self.handover(side, name)
            self.put_down(other(side), ep["plate"], name)
            self.rest(other(side))
        elif task == "give_human":
            # wait over the spot until the hand has arrived, then aim at where the palm actually is
            self.move(self.pose(side, np.asarray(ep["hand"]) + [0, 0, 0.10]), 1.6)
            while self.anims:
                self.send({})
            palm, _ = C.hand_frame(m, d)
            self.put_down(side, palm, name, drop=0.015)
            away = np.asarray(ep["hand"]) + (np.asarray(ep["hand_from"]) - palm) * 1.1
            self.anims.append(self._glide(palm, away, ep["hand_yaw"], 2.2))
            self.rest(side)
        self.wait(0.5)
        if task == "give_human":
            success = C.in_hand(m, d, name)
        else:
            success = C.on_plate(m, d, name)
        if task == "handover":  # a real in-air transfer, not B fishing a dropped candy out of the bowl
            success = success and held
        return {"success": bool(success), "task": task, "candy": name, "arm": side, "grasped": bool(held),
                "frames": self.frames, "instruction": ep["instruction"],
                "candy_end": [round(float(v), 3) for v in d.xpos[m.body(name).id]]}

    def mount_side(self, side: str, direction) -> float:
        """How far (-1..1) the wrist-camera mount points along `direction` from the grasp point, in ik_data's
        current pose (call right after jaw_dir/kinematics)."""
        d = self.ik_data
        v = (d.geom_xpos[self._mount[side]] - self._tip_axis(d, side)[0])[:2]
        return float(v @ np.asarray(direction)[:2]) / (np.linalg.norm(v) + 1e-9)

    def spin_for(self, side: str, q: dict, name: str, local, toward, mount_away=None) -> float:
        """Wrist roll (claws down, holding `name`) that lines the candy up with the horizontal direction `toward`,
        its `local` point on the `toward` side of the grasp (and the camera mount toward `mount_away`)."""
        g = self.m.body(f"{side}_gripper").id
        R0, x0 = self.eng.data.xmat[g].reshape(3, 3).copy(), self.eng.data.xpos[g].copy()
        rel = R0.T @ (C.candy_point(self.m, self.eng.data, name, local) - x0)  # the point, in the gripper frame
        best, best_r = -np.inf, self.hold[f"{side}_wrist_roll.pos"]
        for r in np.arange(-98.0, 98.1, 1.0):
            qr = {**q, f"{side}_wrist_roll.pos": r}
            dvec = self.jaw_dir(qr, side)  # (leaves ik_data at qr)
            d = self.ik_data
            pt = d.xpos[g] + d.xmat[g].reshape(3, 3) @ rel
            grasp_pt = self._tip_axis(d, side)[0]
            v = (pt - grasp_pt)[:2]
            score = float(v @ np.asarray(toward)[:2]) / (np.linalg.norm(v) + 1e-9) - 0.2 * abs(dvec[2]) - 0.001 * abs(r - best_r)
            if mount_away is not None:
                score += 0.3 * self.mount_side(side, mount_away)
            if score > best:
                best, best_r_new = score, r
        return float(best_r_new)

    def handover(self, a: str, name: str) -> bool:
        """A (holding `name` claws-down near one end, as it lifted it) brings it between the arms, long axis
        pointing at B, claws leaning toward A's side; B comes down claws-down on the other end, leaning toward
        B's side. Each wrist-camera mount faces away from the other arm. B closes, A lets go and backs off.
        (Claws-forward presentations drop candy: the pivoting jaw squeezes it out of a tip grip.)"""
        b = other(a)
        s_a = 1.0 if a == "left" else -1.0
        tilt = math.radians(HANDOVER_TILT_DEG)
        ax_a = np.array([0.0, -s_a * math.sin(tilt), -math.cos(tilt)])  # A's wrist leans toward A's side
        ax_b = np.array([0.0, s_a * math.sin(tilt), -math.cos(tilt)])  # B's toward B's side
        _, take_x, take_hw = C.CANDIES[name]["handover"]
        take = (take_x, 0.0, 0.0)
        toward_b = np.array([0.0, -s_a, 0.0])
        # 1) claws still down: spin so the candy lies along y with B's part toward B (a pure spin about vertical)
        arm_now = {k: self.hold[k] for k in self.hold if k.startswith(a) and "gripper" not in k}
        roll = self.spin_for(a, arm_now, name, take, toward=toward_b, mount_away=-toward_b)
        self.move({f"{a}_wrist_roll.pos": roll}, 0.9)
        # 2) to the handover point, leaning toward A's side
        h = HANDOVER + [0, s_a * HANDOVER_Y, 0]
        kw_a = dict(axis=ax_a, weight=TILT_WEIGHT)
        self.move(self.pose(a, h, roll=roll, **kw_a), 1.8)
        self.reach(a, h, **kw_a)
        self.wait(0.3)
        # 3) B onto its part, from above, leaning toward B's side, jaws across the candy, mount away from A
        e = C.candy_point(self.m, self.eng.data, name, take)
        _, u = C.candy_pose(self.m, self.eng.data, name)
        self.set_grasp(b, name, take_hw)
        kw = dict(axis=ax_b, weight=TILT_WEIGHT)
        above = e + [0, 0, 0.06]
        qb = self.pose(b, above, **kw)
        roll_b = self.roll_for(b, qb, lambda dvec: -abs(dvec @ u) + 0.3 * self.mount_side(b, toward_b))
        # up on B's own side first, then across high, then down: a joint-space swing from rest sweeps the candy
        start = self.tip(b)
        high = np.array([start[0], start[1], above[2] + 0.04])
        self.line(b, start, high, 1.0, pieces=3, **kw)
        self.move({**self.pose(b, above + [0, 0, 0.04], roll=roll_b, **kw), f"{b}_gripper.pos": self.opening[b]}, 1.4)
        self.line(b, above + [0, 0, 0.04], e, 1.1, pieces=4, roll=roll_b, **kw)
        self.reach(b, e, **kw)
        self.grip(b, self.close_to(name, take_hw), 0.6)
        self.wait(0.25)
        # 4) A lets go and backs off up and toward its side
        self.grip(a, self.open_for(C.CANDIES[name]["hw"]), 0.5)
        tip_a = self.tip(a)
        self.line(a, tip_a, tip_a + [0, s_a * 0.03, 0.06], 0.6, pieces=2, **kw_a)
        self.rest(a, 1.5)
        c = C.candy_point(self.m, self.eng.data, name, (0, 0, 0))
        return bool(c[2] > 0.08 and np.linalg.norm(c - self.tip(b)) < 0.07)


def attempt(eng: SimEngine, task: str, seed: int, on_tick=None, arm_rule: str = "bowl") -> dict | None:
    """One seeded episode from reset: deterministic, so a headless run predicts a rendered one exactly.
    None when the layout has no candy the task can use."""
    rng = np.random.default_rng(seed)
    include = str(rng.choice(LONG)) if task == "handover" else None
    eng.reset(randomize=True, seed=seed, include=include)
    ex = CandyExpert(eng, arm_rule=arm_rule)
    try:
        ep = ex.setup(task, rng)
    except NoCandy:
        return None
    if on_tick is not None:
        on_tick.expert = ex  # the recorder reads ex.instruction for the frame's task string
    return ex.run(task, rng, on_tick=on_tick, ep=ep)

