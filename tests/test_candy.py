import json

import mujoco
import numpy as np
import pytest

from halloween_bot.sim import candy as C
from halloween_bot.sim.candy_expert import LONG, TASKS, CandyExpert, attempt, instruction
from halloween_bot.sim.engine import REST, SimEngine


@pytest.fixture(scope="module")
def eng():
    return SimEngine(scene="candy")


def settle(eng, seconds=1.5):
    for _ in range(int(seconds * 30)):
        eng.send_action(dict(REST))
        eng.step(1 / 30)


def test_toy_scene_is_still_the_default():
    m = SimEngine().model
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "basket") >= 0
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "candy_bowl") < 0


def test_reset_drops_a_random_subset_into_the_bowl_and_parks_the_rest(eng):
    eng.reset(randomize=True, seed=4)
    chosen = eng.layout["candies"]
    assert 3 <= len(chosen) <= 6 and len(set(chosen)) == len(chosen)
    settle(eng)
    m, d = eng.model, eng.data
    assert sum(C.in_bowl(m, d, n) for n in chosen) >= len(chosen) - 1  # a rolling candy may escape now and then
    for n in set(C.CANDIES) - set(chosen):
        assert d.xpos[m.body(n).id][2] < -0.2  # parked on the hidden shelf under the table


def test_reset_can_force_a_candy_into_the_bowl(eng):
    eng.reset(randomize=True, seed=11, include="licorice")
    assert "licorice" in eng.layout["candies"]


def test_hand_follows_its_target_and_carries_what_is_in_the_palm(eng):
    eng.reset(randomize=True, seed=0)
    m, d = eng.model, eng.data
    C.set_hand(m, d, [0.34, 0.0, 0.12], teleport=True)
    adr = m.joint("gumball_free").qposadr[0]
    d.qpos[adr:adr + 3] = [0.34, 0.0, 0.12 + 0.015]
    mujoco.mj_forward(m, d)
    settle(eng, 0.5)
    assert C.in_hand(m, d, "gumball")
    for i in range(1, 46):  # glide 12 cm away over 1.5 s
        C.set_hand(m, d, [0.34 + 0.12 * i / 45, 0.0, 0.12 + 0.03 * i / 45])
        eng.send_action(dict(REST))
        eng.step(1 / 30)
    settle(eng, 0.5)
    pos, _ = C.hand_frame(m, d)
    assert np.linalg.norm(pos - [0.46, 0.0, 0.15]) < 0.01
    assert C.in_hand(m, d, "gumball")  # friction carried it: the welded hand has real velocity


def test_instructions_name_the_candy():
    assert instruction("pick_place", "candy_corn") == "Pick up the candy corn and put it on the plate."
    assert instruction("give_human", "lollipop") == "Give the lollipop to the person."
    assert "hand it to the other arm" in instruction("handover", "licorice")
    assert set(LONG) <= set(C.CANDIES) and TASKS == ("pick_place", "handover", "give_human")


def test_gripper_opening_and_squeeze_scale_with_the_candy():
    assert CandyExpert.open_for(0.021) > CandyExpert.open_for(0.007)
    assert CandyExpert.close_to("peanut_cup") > CandyExpert.close_to("licorice") == 0.0


@pytest.mark.parametrize("task,seed", [("pick_place", 1), ("give_human", 7)])
def test_expert_completes_a_known_episode(eng, task, seed):
    res = attempt(eng, task, seed)
    assert res is not None and res["success"], res
    assert res["grasped"] and 200 < res["frames"] < 1200


def test_episodes_are_deterministic_per_seed(eng):
    a = attempt(eng, "pick_place", 3)
    qa = eng.data.qpos.copy()
    b = attempt(eng, "pick_place", 3)
    assert a == b and np.array_equal(qa, eng.data.qpos)


def test_record_candy_writes_a_loadable_multitask_dataset(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from halloween_bot.sim.record import record

    root = tmp_path / "ds"
    stats = record(2, root, repo_id="local/candy_test", seed0=0, image_hw=(96, 128), log=lambda *_: None,
                   scene="candy", tasks=("pick_place", "give_human"))
    assert stats["saved"] == {"pick_place": 1, "give_human": 1}
    ds = LeRobotDataset("local/candy_test", root=root)
    assert ds.num_episodes == 2
    tasks = set(ds.meta.tasks.index)
    assert any(t.startswith("Pick up the") for t in tasks) and any(t.startswith("Give the") for t in tasks)
    meta = json.loads((root / "meta" / "sim_record.json").read_text())
    assert {r["task"] for r in meta["results"]} == {"pick_place", "give_human"}


def test_eval_world_brings_the_hand_in_once_the_candy_is_lifted(eng):
    from halloween_bot.sim.eval_policy import World

    eng.reset(randomize=True, seed=7)
    ex = CandyExpert(eng)
    ep = ex.setup("give_human", np.random.default_rng(7))
    world = World(eng, "give_human", ep)
    m, d = eng.model, eng.data
    for _ in range(5):
        world.tick()
    assert world.hand_path is None  # candy still in the bowl: the person waits
    adr = m.joint(f"{ep['candy']}_free").qposadr[0]
    d.qpos[adr + 2] = 0.12  # the robot has lifted it
    mujoco.mj_forward(m, d)
    for _ in range(60):
        world.tick()
    target = d.mocap_pos[int(m.body_mocapid[m.body("hand_target").id])]
    assert np.linalg.norm(target - ep["hand"]) < 1e-6  # reached in all the way
    assert not world.success()
