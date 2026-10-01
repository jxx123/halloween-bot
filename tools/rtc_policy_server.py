#!/usr/bin/env python
"""lerobot's async PolicyServer with real-time chunking (RTC) for flow-matching policies (pi05, pi0).

Without RTC the robot keeps executing the old chunk while the next one is inferred, and the new chunk is
planned from an observation that is already stale when it lands: a jump or a stall at every chunk
boundary. RTC (Black et al. 2025; lerobot policies/rtc) conditions the new chunk on the old chunk's
unexecuted tail, freezing the first `inference_delay` steps (they run during inference) and softly
guiding the next `execution_horizon`, so chunks join smoothly.

lerobot 0.6.1 has RTC only for a robot in the same process (rollout/inference/rtc.py); its async server
ignores it. This keeps the async protocol unchanged, so the same clients work (halloween_bot PolicyRunner,
lerobot robot_client):
  - an observation's timestep is the robot's last executed step (both clients send that);
  - the new chunk is labelled from the NEXT step (i0 = timestep + 1); the client keeps only steps it has not
    run yet ("latest chunk wins" for the future), which drops the steps that ran during inference;
  - the prefix = the previous chunk's normalized actions from i0 on, padded/truncated to execution_horizon;
  - the delay (steps) = ceil(recent max compute latency / dt) + --net-margin-steps.

Run it from the ~/pi-train venv (pristine lerobot 0.6.1: pi05 needs no shims), not ~/pi-serve:
    ~/pi-train/.venv/bin/python tools/rtc_policy_server.py --port 8083
"""
import argparse
import logging
import math
import os
import time
from collections import deque
from concurrent import futures

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import grpc
import torch

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.helpers import raw_observation_to_observation
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.transport import services_pb2_grpc


def fit_length(prev: torch.Tensor, steps: int) -> torch.Tensor:
    """Pad (zeros) or truncate a [T, A] prefix to `steps` rows (lerobot's _normalize_prev_actions_length)."""
    if prev.shape[0] >= steps:
        return prev[:steps]
    out = torch.zeros((steps, prev.shape[1]), dtype=prev.dtype, device=prev.device)
    out[: prev.shape[0]] = prev
    return out


class RTCPolicyServer(PolicyServer):
    def __init__(self, config: PolicyServerConfig, rtc: RTCConfig, net_margin_steps: int = 1):
        super().__init__(config)
        self.rtc, self.net_margin_steps = rtc, net_margin_steps
        self._prev: tuple[int, torch.Tensor] | None = None  # (first timestep, normalized actions [T, A])
        self._latency = deque(maxlen=10)
        self.stats = {"chunks": 0, "rtc_chunks": 0, "last_delay": None}

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        reply = super().SendPolicyInstructions(request, context)
        if self.rtc.enabled and hasattr(self.policy, "init_rtc_processor"):
            self.policy.config.rtc_config = self.rtc
            self.policy.init_rtc_processor()
            self.logger.info(f"RTC on: execution_horizon={self.rtc.execution_horizon}, "
                             f"max_guidance_weight={self.rtc.max_guidance_weight}")
        elif self.rtc.enabled:
            self.logger.warning(f"{type(self.policy).__name__} has no RTC support: serving plain chunks")
        self._prev = None
        self._latency.clear()
        return reply

    def _predict_action_chunk(self, observation_t):
        t_start = time.perf_counter()
        obs = raw_observation_to_observation(observation_t.get_observation(), self.lerobot_features,
                                             self.policy_image_features)
        obs = self.preprocessor(obs)
        i0 = observation_t.get_timestep() + 1  # the next step the robot will take
        prefix, delay = None, 0
        if self._prev is not None:
            t0, prev = self._prev
            k = i0 - t0
            if 0 <= k < prev.shape[0]:
                prefix = fit_length(prev[k:], self.rtc.execution_horizon)
                dt = self.config.environment_dt
                delay = (math.ceil(max(self._latency) / dt) if self._latency else 0) + self.net_margin_steps
                delay = min(delay, self.rtc.execution_horizon)
        if not self.rtc.enabled:
            prefix, delay = None, 0
        kwargs = {"inference_delay": delay, "prev_chunk_left_over": prefix} if prefix is not None else {}
        with torch.no_grad():
            chunk = self.policy.predict_action_chunk(obs, **kwargs)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)
        chunk = chunk[:, : self.actions_per_chunk, :]
        self._prev = (i0, chunk[0].detach().clone())  # normalized: what the next prefix is made of
        actions = torch.stack([self.postprocessor(chunk[:, i, :]) for i in range(chunk.shape[1])], dim=1)
        actions = actions.squeeze(0).detach().cpu()
        self._latency.append(time.perf_counter() - t_start)
        self.stats["chunks"] += 1
        self.stats["rtc_chunks"] += prefix is not None
        self.stats["last_delay"] = delay
        self.logger.info(f"chunk from step {i0}: rtc={prefix is not None} delay={delay} "
                         f"compute={self._latency[-1] * 1000:.0f}ms")
        return self._time_action_chunk(observation_t.get_timestamp(), list(actions), i0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8083)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--execution-horizon", type=int, default=10)
    ap.add_argument("--max-guidance-weight", type=float, default=10.0)
    ap.add_argument("--net-margin-steps", type=int, default=1, help="extra delay steps for the network round trip")
    ap.add_argument("--no-rtc", action="store_true", help="serve plain chunks (same timing, for A/B tests)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    cfg = PolicyServerConfig(host=a.host, port=a.port, fps=a.fps)
    rtc = RTCConfig(enabled=not a.no_rtc, execution_horizon=a.execution_horizon,
                    max_guidance_weight=a.max_guidance_weight)
    server_impl = RTCPolicyServer(cfg, rtc, a.net_margin_steps)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(server_impl, server)
    server.add_insecure_port(f"{a.host}:{a.port}")
    server.start()
    server_impl.logger.info(f"RTC PolicyServer on {a.host}:{a.port} (rtc={'off' if a.no_rtc else 'on'})")
    server.wait_for_termination()


if __name__ == "__main__":
    main()
