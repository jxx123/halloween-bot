import numpy as np

from halloween_bot.sim.calib import KEYS


def test_record_writes_a_loadable_lerobot_dataset(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from halloween_bot.sim.record import CAMERAS, record

    root = tmp_path / "ds"
    stats = record(1, root, repo_id="local/sim_test", seed0=0, image_hw=(96, 128))
    assert stats["saved"] == 1 and stats["attempts"] >= 1
    ds = LeRobotDataset("local/sim_test", root=root)
    assert ds.num_episodes == 1 and 200 < ds.num_frames < 700
    item = ds[10]
    assert item["observation.state"].shape == (12,) and item["action"].shape == (12,)
    for cam in CAMERAS:
        assert item[f"observation.images.{cam}"].shape == (3, 96, 128)
    assert item["task"] == "Grasp the toy and place it in the basket."
    assert ds.meta.features["action"]["names"] == KEYS
    # actions lead states: the arm follows its commands
    a = np.stack([ds[i]["action"].numpy() for i in range(0, ds.num_frames, 20)])
    s = np.stack([ds[i]["observation.state"].numpy() for i in range(0, ds.num_frames, 20)])
    assert np.abs(a - s).mean() < 15
