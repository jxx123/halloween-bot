import os

os.environ.setdefault("MUJOCO_GL", "egl")
import json
import urllib.request

import numpy as np
import pandas as pd
import pytest

from halloween_bot.sim import sysid_collect
from halloween_bot.sim.calib import KEYS, Calibration
from halloween_bot.sim.engine import SimEngine
from halloween_bot.sim.model import build_model
from halloween_bot.sim.sysid import Trace, fit, load_dataset_traces, load_trajectory_traces, replay, rmse

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


def test_load_trajectory_traces_round_trips_engine_trace(tmp_path):
    soft = {"motors": {"elbow_flex": {"kp": 5.0, "kv": 0.2}}}  # unsaturated at 20 units, so the clamp matters
    model = build_model(params=soft, props=False)
    eng = SimEngine(model=model, calib=CAL)
    t = np.arange(30) / 30
    points = [{"right_elbow_flex.pos": 55.0}] * 10 + [  # a 40-unit jump: the ±20 per-send clamp is active
        {"right_elbow_flex.pos": 55 + 15 * np.sin(np.pi * s), "left_wrist_flex.pos": 45 + 10 * s} for s in t]
    out = eng.run_trajectory(points, hz=40)  # 40 Hz = exactly 5 physics steps per tick
    path = tmp_path / "traces.json"
    path.write_text(json.dumps({"sequences": [{"name": "elbow", "hz": out["hz"], "trace": out["trace"]}]}))

    (tr,) = load_trajectory_traces(path)
    assert tr.name == "elbow" and tr.hz == 40
    assert tr.present.shape == (41, 12) and tr.cmd.shape == (40, 12)
    assert tr.present[3] == pytest.approx([out["trace"][3]["present"][k] for k in KEYS])
    assert tr.cmd[7] == pytest.approx([out["trace"][7]["cmd"][k] for k in KEYS])
    # replay with the same model reproduces what the engine measured, tick for tick
    assert np.abs(replay(model, CAL, tr) - tr.present[1:]).max() < 1e-3


def test_load_dataset_traces_pairs_action_t_with_state_t_plus_1(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text(json.dumps({"fps": 30}))
    (tmp_path / "data" / "chunk-000").mkdir(parents=True)
    ep = [0] * 5 + [1] * 4
    rows = [{"action": np.full(12, 100.0 + i, np.float32), "observation.state": np.full(12, float(i), np.float32),
             "episode_index": e, "frame_index": i} for i, e in enumerate(ep)]
    pd.DataFrame(rows).to_parquet(tmp_path / "data" / "chunk-000" / "file-000.parquet")

    a, b = load_dataset_traces(tmp_path)
    assert a.hz == 30 and a.present.shape == (5, 12) and a.cmd.shape == (4, 12)
    assert b.present.shape == (4, 12) and b.cmd.shape == (3, 12)
    assert a.present[:, 0].tolist() == [0, 1, 2, 3, 4] and a.cmd[:, 0].tolist() == [100, 101, 102, 103]
    assert b.present[:, 0].tolist() == [5, 6, 7, 8] and b.cmd[:, 0].tolist() == [105, 106, 107]


@pytest.mark.parametrize("arm", ["left", "right"])
def test_collect_sequences_stay_within_20_of_base(arm):
    seqs = sysid_collect.build_sequences(arm)
    base = sysid_collect.base_pose(arm)
    assert {s["name"] for s in seqs} >= {f"{arm}_{m}" for m in sysid_collect.MOTORS}
    for s in seqs:
        assert s["hz"] == 30 and s["points"][0] == base and s["points"][-1] == base
        for p in s["points"]:
            assert set(p) == set(base)  # only the chosen arm; the other arm holds
            assert max(abs(p[k] - base[k]) for k in p) <= 20.0
            # keep 5 units off the calibrated ends (mechanical stops: a stalled servo trips overload)
            assert all(5 <= v <= 95 if k.endswith("gripper.pos") else -95 <= v <= 95 for k, v in p.items())


def test_collect_dry_run_makes_no_network_calls(monkeypatch, capsys):
    def no_network(*a, **k):
        raise AssertionError("dry run touched the network")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    assert sysid_collect.main(["--arm", "right", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "right_elbow_flex" in out and "total" in out
