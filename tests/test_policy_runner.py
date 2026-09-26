import pickle
import threading
import time

import grpc
import numpy as np
import pytest
import torch
from lerobot.async_inference.helpers import TimedAction
from lerobot.transport import services_pb2

from halloween_bot.policy_runner import PolicyBusy, PolicyRunner, PolicyUnavailable, handle_http

KEYS = [f"k{i}" for i in range(12)]
FEATURES = {**{k: float for k in KEYS}, "base_0_rgb": (8, 8, 3)}


class Unavailable(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.UNAVAILABLE


class FakeStub:
    """Answers each must_go observation with a chunk of `chunk` actions valued at their timestep."""

    def __init__(self, reachable=True, chunk=10, delay=0.05, instruct_fixes=True, ready_delay=0.0,
                 garbage=False):
        self.reachable, self.chunk, self.delay = reachable, chunk, delay
        self.instruct_fixes, self.ready_delay, self.garbage = instruct_fixes, ready_delay, garbage
        self.channel = None
        self.instructions = 0
        self.obs = []
        self.pending = []
        self.cv = threading.Condition()
        self.serve = True

    def Ready(self, req, timeout=None):
        time.sleep(self.ready_delay)
        if not self.reachable:
            raise Unavailable()
        return services_pb2.Empty()

    def SendPolicyInstructions(self, req, timeout=None):
        self.instructions += 1
        if self.channel is not None and self.channel.block_instructions:  # a slow model load
            if not self.channel.closed.wait(10):
                raise AssertionError("instructions never cancelled")
            raise Cancelled()
        if self.instruct_fixes:
            self.serve = True
        return services_pb2.Empty()

    def SendObservations(self, it, timeout=None):
        obs = pickle.loads(b"".join(m.data for m in it))
        self.obs.append(obs)
        if obs.must_go and self.serve:
            with self.cv:
                self.pending.append(obs)
                self.cv.notify()
        return services_pb2.Empty()

    def GetActions(self, req, timeout=None):
        with self.cv:
            if not self.pending:
                self.cv.wait(0.2)
            if not self.pending:
                return services_pb2.Actions(data=b"")
            obs = self.pending.pop(0)
        time.sleep(self.delay)
        if self.garbage:
            return services_pb2.Actions(data=b"not a pickle")
        t0 = obs.get_timestep()
        acts = [TimedAction(timestamp=time.time(), timestep=t0 + i, action=torch.full((12,), float(t0 + i)))
                for i in range(self.chunk)]
        return services_pb2.Actions(data=pickle.dumps(acts))


class Cancelled(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.CANCELLED


class FakeChannel:
    def __init__(self, block_instructions=False):
        self.block_instructions = block_instructions
        self.closed = threading.Event()

    def close(self):
        self.closed.set()


def make(stub, block_instructions=False, **kw):
    sent = []

    def factory(addr):
        stub.channel = FakeChannel(block_instructions)
        return stub.channel, stub

    r = PolicyRunner(lambda: {**{k: 0.0 for k in KEYS}, "base_0_rgb": np.zeros((8, 8, 3), np.uint8)},
                     sent.append, KEYS, FEATURES, fps=100, stub_factory=factory, **kw)
    return r, sent


def wait_done(r, t=5):
    end = time.time() + t
    while r.running and time.time() < end:
        time.sleep(0.02)
    assert not r.running


def test_start_fails_fast_when_unreachable():
    r, _ = make(FakeStub(reachable=False))
    t = time.time()
    with pytest.raises(PolicyUnavailable, match="unreachable"):
        r.start("t", 1)
    assert time.time() - t < 2 and not r.running and r.status()["phase"] == "idle"


def test_runs_executes_actions_in_order_and_finishes():
    stub = FakeStub()
    r, sent = make(stub)
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
    stub = FakeStub()
    r, _ = make(stub)
    r.start("a", 0.1)
    wait_done(r)
    r.start("b", 0.1)
    wait_done(r)
    assert stub.instructions == 1


def test_reinstructs_when_no_chunk_arrives():
    stub = FakeStub()
    r, _ = make(stub, first_chunk_timeout=0.5)
    r.start("a", 0.1)
    wait_done(r)
    stub.serve = False  # a restarted server that lost the policy
    r.start("b", 0.2)
    wait_done(r, 10)
    assert stub.instructions == 2 and r.status()["phase"] == "done"


def test_lockstep_pauses_during_inference_and_resumes():
    calls = []
    r, _ = make(FakeStub(delay=0.1), pause=lambda: calls.append("pause"), resume=lambda: calls.append("resume"))
    r.start("t", seconds=0.25, lockstep=True)
    wait_done(r)
    assert calls[0] == "pause" and calls.count("pause") == calls.count("resume") >= 2


def test_stop_during_lockstep_resumes_physics():
    calls = []
    stub = FakeStub()
    stub.serve = False  # never answers -> stuck waiting while paused
    r, _ = make(stub, pause=lambda: calls.append("pause"), resume=lambda: calls.append("resume"),
                first_chunk_timeout=60)
    r._instructed = True  # skip loading: straight to the paused wait
    r.start("t", seconds=5, lockstep=True)
    time.sleep(0.3)
    r.stop()
    assert calls and calls[-1] == "resume" and calls.count("pause") == calls.count("resume")


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


def test_fit_image_crops_16_9_to_4_3_then_resizes():
    from halloween_bot.policy_runner import fit_image
    img = np.zeros((720, 1280, 3), np.uint8)
    img[:, 160:1120] = 255  # the central 4:3 region
    out = fit_image(img)
    assert out.shape == (480, 640, 3) and out.min() == 255
    assert fit_image(np.zeros((480, 640, 3), np.uint8)).shape == (480, 640, 3)


def test_realtime_run_errors_out_when_no_chunk_ever_arrives():
    stub = FakeStub(instruct_fixes=False)
    stub.serve = False
    r, _ = make(stub, first_chunk_timeout=0.3)
    r.start("t", seconds=0.5)
    wait_done(r, 5)
    st = r.status()
    assert st["phase"] == "error" and "no action chunk" in st["error"]


def test_receiver_crash_ends_the_run_with_an_error():
    r, _ = make(FakeStub(garbage=True))
    r.start("t", seconds=5)
    wait_done(r, 5)
    assert r.status()["phase"] == "error"


def test_concurrent_starts_run_exactly_once():
    r, _ = make(FakeStub(ready_delay=0.2))
    outcomes = []

    def go():
        try:
            r.start("t", seconds=0.3)
            outcomes.append("started")
        except PolicyBusy:
            outcomes.append("busy")

    threads = [threading.Thread(target=go) for _ in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(outcomes) == ["busy", "started"]
    wait_done(r)


def test_stop_cancels_a_blocking_model_load():
    r, _ = make(FakeStub(), block_instructions=True)
    r.start("t", seconds=30)
    time.sleep(0.2)
    assert r.status()["phase"] == "loading"
    t = time.time()
    st = r.stop()
    assert time.time() - t < 2 and not st["running"]


def test_blocked_by_refuses_start():
    r, _ = make(FakeStub(), blocked_by=lambda: "the real robot is using the policy server")
    with pytest.raises(PolicyBusy, match="real robot"):
        r.start("t", 1)
    assert not r.running


def test_run_yields_when_another_client_appears():
    other = {"reason": None}
    r, _ = make(FakeStub(), blocked_by=lambda: other["reason"])
    r.start("t", seconds=30)
    time.sleep(0.3)
    other["reason"] = "the real robot connected"
    wait_done(r, 3)
    st = r.status()
    assert st["phase"] == "error" and "real robot connected" in st["error"]
