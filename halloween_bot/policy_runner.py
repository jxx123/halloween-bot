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

import cv2
import grpc
import numpy as np

from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.feature_utils import hw_to_dataset_features

DEFAULT_CHECKPOINT = "delvingdeep/pi0fast-so101-bimanual"
DEFAULT_TASK = "Grasp the toy and place it in the basket."
# robot camera name -> the checkpoint's image feature name
PI0FAST_CAMERAS = {"overhead": "base_0_rgb", "left_wrist": "left_wrist_0_rgb", "right_wrist": "right_wrist_0_rgb"}
PI0FAST_IMAGE_HW = (480, 640)  # the real pi0fast client captured all three cams at 640x480


def fit_image(img: np.ndarray, hw: tuple[int, int] = PI0FAST_IMAGE_HW) -> np.ndarray:
    """Center-crop to hw's aspect (the C922's 4:3 mode crops the sides of its 16:9 view), then resize."""
    h, w = hw
    ih, iw = img.shape[:2]
    if iw * h > ih * w:
        cw = ih * w // h
        img = img[:, (iw - cw) // 2:(iw - cw) // 2 + cw]
    elif iw * h < ih * w:
        ch = iw * h // w
        img = img[(ih - ch) // 2:(ih - ch) // 2 + ch]
    return img if img.shape[:2] == hw else cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


class PolicyBusy(RuntimeError):
    pass


class PolicyUnavailable(RuntimeError):
    pass


def _default_stub_factory(address: str):
    channel = grpc.insecure_channel(address, grpc_channel_options())
    return channel, services_pb2_grpc.AsyncInferenceStub(channel)


def _rpc_code(e: Exception) -> str:
    try:
        return e.code().name
    except Exception:
        return type(e).__name__


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
        self.server_address, self.checkpoint = server_address, checkpoint
        self.policy_type, self.device = policy_type, device
        self.fps, self.actions_per_chunk = fps, actions_per_chunk
        self.chunk_size_threshold, self.obs_min_interval = chunk_size_threshold, obs_min_interval
        self.pause, self.resume = pause, resume
        self.connect_timeout, self.load_timeout = connect_timeout, load_timeout
        self.first_chunk_timeout = first_chunk_timeout
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
            raise PolicyUnavailable(f"policy server unreachable at {self.server_address} ({_rpc_code(e)})") from e
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
                self._set(error=f"GetActions failed: {_rpc_code(e)}")
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

    def _reinstruct(self, stub, task: str):
        self._set(phase="reloading")
        stub.Ready(services_pb2.Empty(), timeout=self.connect_timeout)
        self._instruct(stub)
        self._set(phase="running")
        with self._lock:
            self._must_go = True
        self._send_observation(stub, task, force_must_go=True)

    def _wait_for_chunk(self, stub, task: str):
        """Lockstep: block until a chunk lands; re-instruct once if the server seems to have no policy."""
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
                    self.pause()
                    self._paused_by_us = True
                    self._send_observation(stub, task, force_must_go=True)
                    self._wait_for_chunk(stub, task)
                    self._paused_by_us = False
                    self.resume()
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
