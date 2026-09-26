import numpy as np
import pytest

from halloween_bot.sim.calib import KEYS
from halloween_bot.sim.engine import SimEngine
from halloween_bot.sim.expert import Expert


@pytest.fixture(scope="module")
def eng():
    return SimEngine()


@pytest.mark.parametrize("target", [(0.28, -0.11, 0.075), (0.25, -0.05, 0.12), (0.20, -0.10, 0.22)])
def test_ik_reaches_target_pointing_down(eng, target):
    eng.reset()
    ex = Expert(eng)
    q = ex.ik("right", np.array(target))
    assert set(q) == {k for k in KEYS if k.startswith("right") and "gripper" not in k and "roll" not in k}
    tip, axis = ex.tip_after(q, "right")
    assert np.linalg.norm(tip - np.array(target)) < 0.006
    assert axis[2] < -0.85  # claws within ~30° of straight down


def test_expert_puts_a_toy_in_the_basket(eng):
    eng.reset(randomize=True, seed=3)
    ex = Expert(eng)
    result = ex.run(np.random.default_rng(3))
    assert result["success"], result
    assert 200 < result["frames"] < 600
