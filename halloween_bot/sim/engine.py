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
        self._a = np.array([a for a, _ in aff])
        self._b = np.array([b for _, b in aff])
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
        return time.perf_counter() if self._running.is_set() else self.data.time

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
        """Sys-ID: send each point (merged over the last command) at hz; trace present-before-send + cmd."""
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
                    x += rng.uniform(-0.04, 0.04)
                    y += rng.uniform(-0.05, 0.05)
                    yaw = rng.uniform(-np.pi, np.pi)
                adr = self.model.jnt_qposadr[jid]
                self.data.qpos[adr:adr + 7] = [x, y, 0.04, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            mujoco.mj_forward(self.model, self.data)
            self._resync.set()  # inside the lock: the physics loop can't step on a stale clock anchor

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
            if self._paused.is_set():  # before resync: resume() re-arms it for the first live step
                time.sleep(0.005)
                continue
            if self._resync.is_set():
                self._resync.clear()
                sim0, wall0 = self.data.time, time.perf_counter()
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
        """Latest RGB frame of an API camera (overhead, left_wrist, right_wrist, scene)."""
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
