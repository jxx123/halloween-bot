"""RTC policy server: chunk labelling and the prefix taken from the previous chunk (no model needed)."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("lerobot.async_inference.policy_server")
pytest.importorskip("lerobot.policies.rtc.configuration_rtc")

spec = importlib.util.spec_from_file_location("rtc_policy_server", Path(__file__).parents[1] / "tools" / "rtc_policy_server.py")
rtc_mod = importlib.util.module_from_spec(spec)
sys.modules["rtc_policy_server"] = rtc_mod
spec.loader.exec_module(rtc_mod)


class FakePolicy:
    def __init__(self):
        self.calls = []
        self.n = 0

    def predict_action_chunk(self, obs, **kw):
        self.calls.append(kw)
        self.n += 1
        return torch.arange(50, dtype=torch.float32).reshape(1, 50, 1).repeat(1, 1, 2) + 1000 * self.n


def make_server(monkeypatch, enabled=True):
    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    monkeypatch.setattr(rtc_mod, "raw_observation_to_observation", lambda *a, **k: {})
    s = rtc_mod.RTCPolicyServer(PolicyServerConfig(host="127.0.0.1", port=18099, fps=30),
                                RTCConfig(enabled=enabled, execution_horizon=10), net_margin_steps=1)
    s.policy, s.preprocessor, s.postprocessor = FakePolicy(), (lambda o: o), (lambda a: a)
    s.lerobot_features, s.actions_per_chunk = {}, 50
    monkeypatch.setattr(type(s), "policy_image_features", property(lambda self: {}))
    return s


def obs(ts):
    return SimpleNamespace(get_observation=lambda: {}, get_timestep=lambda: ts, get_timestamp=lambda: 0.0)


def test_chunks_start_at_the_next_step_and_carry_the_unexecuted_tail(monkeypatch):
    s = make_server(monkeypatch)
    first = s._predict_action_chunk(obs(0))
    assert [a.get_timestep() for a in first][:3] == [1, 2, 3]  # labelled from the next step
    assert s.policy.calls[0] == {}  # nothing to continue yet
    s._latency.append(0.1)  # 3 steps at 30 fps
    s._predict_action_chunk(obs(20))  # the robot has run steps 1..20 of the first chunk
    kw = s.policy.calls[1]
    prefix = kw["prev_chunk_left_over"]
    assert prefix.shape == (10, 2)  # execution_horizon rows
    assert prefix[0, 0].item() == 1000 + 20  # first chunk's action for step 21 (index 20)
    assert kw["inference_delay"] == 3 + 1  # ceil(0.1 s / (1/30 s)) + the network margin


def test_rtc_off_serves_plain_chunks(monkeypatch):
    s = make_server(monkeypatch, enabled=False)
    s._predict_action_chunk(obs(0))
    s._predict_action_chunk(obs(20))
    assert s.policy.calls[1] == {}
