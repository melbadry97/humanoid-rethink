import mujoco
import numpy as np
import pytest

from hrethink.genome import SEEDS, Genome, mutate
from hrethink.mjcf import build_xml
from hrethink.sim import n_params, optimise_controller, rollout, score_design


@pytest.mark.parametrize("name", list(SEEDS))
@pytest.mark.parametrize("terrain", ["flat", "rough"])
def test_seed_bodies_compile(name, terrain):
    xml, joints = build_xml(SEEDS[name], terrain)
    m = mujoco.MjModel.from_xml_string(xml)
    assert m.nu == len(joints)


def test_random_mutants_compile():
    import random

    rng, g = random.Random(0), Genome()
    for _ in range(50):
        g = mutate(g, rng, rate=0.6)
        mujoco.MjModel.from_xml_string(build_xml(g)[0])


def test_from_loose_clips_out_of_range():
    g = Genome.from_loose({"thigh_len": 9.0, "max_torque": -5, "foot": "wheel", "bogus": 1})
    assert g.thigh_len == 0.55 and g.max_torque == 20.0 and g.foot == "wheel"


def test_standing_still_does_not_fall():
    for g in SEEDS.values():
        if g.foot == "point" and g.leg_pairs == 1:
            continue  # point-foot bipeds need active balance
        r = rollout(g, np.zeros(n_params(build_xml(g)[1])), duration=2.0)  # zero params = hold pose
        assert not r.fell


def test_tiny_optimisation_and_scoring():
    g = SEEDS["centaur"]
    res = optimise_controller(g, generations=2, popsize=4, duration=1.0, workers=1)
    metrics = score_design(g, res.params, duration=1.0)
    assert set(metrics) >= {"score", "speed_flat", "speed_rough", "push_survived", "cost_of_transport"}
