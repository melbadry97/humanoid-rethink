"""Controller, rollouts, inner-loop controller optimisation, and task-suite scoring.

Inner loop: every proposed body gets its own open-loop CPG gait with simple
IMU feedback, tuned by a small evolution strategy. This is cheap (seconds on
a laptop CPU) and fair across very different bodies, which is what the outer
morphology search needs. Swap in PPO (MuJoCo Playground / Isaac Lab) once a
body is worth refining.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from multiprocessing import Pool

import mujoco
import numpy as np

from .genome import Genome
from .mjcf import build_xml, standing_height

CTRL_EVERY = 5  # physics steps per control step (100 Hz control at dt=2 ms)
GLOBAL = ["freq", "lr_phase", "pair_phase", "k_pitch", "k_rate", "k_roll"]


def joint_types(joints: list[str]) -> list[str]:
    return sorted({j.split("_")[0] for j in joints})


def n_params(joints: list[str]) -> int:
    return len(GLOBAL) + 3 * len(joint_types(joints))


class CPG:
    """Shared-parameter central pattern generator.

    Joints of the same type share amplitude/offset/phase; left/right and
    front/rear copies are phase-shifted by learned offsets. Params live in
    [-1, 1] and are mapped to physical ranges here.
    """

    def __init__(self, params: np.ndarray, joints: list[str]):
        p = np.clip(params, -1, 1)
        self.joints = joints
        self.freq = 1.75 + 1.25 * p[0]
        self.lr_phase = math.pi * (1 + p[1])
        self.pair_phase = math.pi * (1 + p[2])
        self.k_pitch, self.k_rate, self.k_roll = 3 * p[3], 0.5 * p[4], 3 * p[5]
        types = joint_types(joints)
        per = p[len(GLOBAL):].reshape(len(types), 3)
        self.type_params = {t: per[i] for i, t in enumerate(types)}

    def __call__(self, t: float, pitch: float, pitch_rate: float, roll: float) -> np.ndarray:
        ramp = min(1.0, t / 1.0)  # fade the gait in so every body starts from a stable stand
        out = np.zeros(len(self.joints))
        for i, j in enumerate(self.joints):
            jt, _, tag = j.partition("_")
            p_amp, off, ph = self.type_params[jt]
            amp = max(0.0, p_amp)  # params <= 0 mean "hold still", so the ES starts from standing
            side = -1.0 if tag.endswith("r") else 1.0
            phase = math.pi * ph + (self.lr_phase if side < 0 else 0.0)
            if tag.startswith("r") and len(tag) == 2:  # rear pair of a 4-legged body
                phase += self.pair_phase
            s = ramp * math.sin(2 * math.pi * self.freq * t + phase)
            if jt == "wheel":
                # wheels balance like a Segway: drive under the falling trunk
                out[i] = 25.0 * ramp * (0.3 * amp * s + off) + 10.0 * (self.k_pitch * pitch + self.k_rate * pitch_rate)
                continue
            target = 0.6 * amp * s + 0.8 * off * ramp
            if jt == "hiproll":
                target = side * 0.4 * off * ramp + 0.2 * amp * s + side * self.k_roll * roll
            elif jt in ("hippitch", "ankle"):
                target += self.k_pitch * pitch + self.k_rate * pitch_rate
            out[i] = target
        return out


@dataclass
class Rollout:
    distance: float
    lateral: float
    alive_frac: float
    energy: float
    mass: float
    duration: float
    fell: bool

    @property
    def speed(self) -> float:
        return self.distance / self.duration

    @property
    def cot(self) -> float:
        """Cost of transport: mechanical energy / (weight * distance)."""
        return self.energy / (self.mass * 9.81 * max(self.distance, 0.05))


_MODEL_CACHE: dict[str, mujoco.MjModel] = {}


def _model(xml: str) -> mujoco.MjModel:
    m = _MODEL_CACHE.get(xml)
    if m is None:
        if len(_MODEL_CACHE) > 16:
            _MODEL_CACHE.clear()
        m = _MODEL_CACHE[xml] = mujoco.MjModel.from_xml_string(xml)
    return m


def rollout(g: Genome, params: np.ndarray, duration: float = 6.0, terrain: str = "flat",
            seed: int = 0, push_at: float | None = None, push_force: float = 0.0) -> Rollout:
    xml, joints = build_xml(g, terrain, seed)
    m = _model(xml)
    d = mujoco.MjData(m)
    ctrl = CPG(params, joints)
    pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    mass = float(m.body_subtreemass[pelvis])
    z0 = standing_height(g)
    steps = int(duration / m.opt.timestep)
    energy, alive_steps, fell = 0.0, 0, False
    push_steps = range(int(push_at / m.opt.timestep), int((push_at + 0.1) / m.opt.timestep)) if push_at else range(0)
    for k in range(steps):
        if k % CTRL_EVERY == 0:
            q = d.qpos[3:7]
            # pitch/roll of the pelvis from its quaternion (w, x, y, z)
            pitch = math.asin(max(-1.0, min(1.0, 2 * (q[0] * q[2] - q[3] * q[1]))))
            roll = math.atan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
            d.ctrl[:] = ctrl(d.time, pitch, d.qvel[4], roll)
        d.xfrc_applied[pelvis, 1] = push_force * mass if k in push_steps else 0.0
        mujoco.mj_step(m, d)
        energy += float(np.abs(d.actuator_force * d.actuator_velocity).sum()) * m.opt.timestep
        up_z = d.xmat[pelvis][8]
        if d.qpos[2] < 0.45 * z0 or up_z < 0.35 or not np.isfinite(d.qpos).all():
            fell = True
            break
        alive_steps += 1
    return Rollout(distance=float(d.qpos[0]), lateral=float(d.qpos[1]), alive_frac=alive_steps / steps,
                   energy=energy, mass=mass, duration=duration, fell=fell)


def train_fitness(r: Rollout) -> float:
    # progress only counts if the robot is still up; lunging forward and falling is not a gait
    return r.distance * r.alive_frac ** 2 + 1.0 * r.alive_frac - 0.5 * abs(r.lateral) - 0.01 * r.energy / r.mass


def _eval_one(args) -> float:
    g_json, params, duration = args
    g = Genome.model_validate_json(g_json)
    return train_fitness(rollout(g, params, duration))


@dataclass
class TrainResult:
    params: np.ndarray
    fitness: float
    history: list[float] = field(default_factory=list)


def optimise_controller(g: Genome, generations: int = 20, popsize: int = 24, duration: float = 8.0,
                        seed: int = 0, workers: int | None = None) -> TrainResult:
    """(mu/mu, lambda)-ES with elitism over CPG parameters."""
    _, joints = build_xml(g)
    rng = np.random.default_rng(seed)
    dim = n_params(joints)
    mean, sigma = np.zeros(dim), 0.5
    best_p, best_f, hist = mean.copy(), -np.inf, []
    mu = popsize // 4
    workers = workers or min(4, os.cpu_count() or 1)
    g_json = g.model_dump_json()
    with Pool(workers) as pool:
        for _ in range(generations):
            pop = np.clip(mean + sigma * rng.standard_normal((popsize, dim)), -1, 1)
            pop[0] = best_p  # elitism
            fits = np.array(pool.map(_eval_one, [(g_json, p, duration) for p in pop]))
            order = np.argsort(-fits)
            if fits[order[0]] > best_f:
                best_f, best_p = float(fits[order[0]]), pop[order[0]].copy()
            mean = pop[order[:mu]].mean(axis=0)
            sigma = max(0.08, sigma * 0.93)
            hist.append(best_f)
    return TrainResult(best_p, best_f, hist)


def score_design(g: Genome, params: np.ndarray, duration: float = 8.0) -> dict:
    """Task suite reflecting 'modern needs': flat speed, rubble, push recovery, efficiency."""
    flat = rollout(g, params, duration, "flat")
    rough = [rollout(g, params, duration, "rough", seed=s) for s in (1, 2)]
    push = rollout(g, params, duration, "flat", push_at=3.0, push_force=4.0)
    speed_flat = flat.speed * flat.alive_frac
    speed_rough = float(np.mean([r.speed * r.alive_frac for r in rough]))
    push_ok = 0.0 if push.fell else 1.0
    cot = flat.cot if flat.distance > 0.1 else 10.0
    score = 0.5 * (speed_flat + speed_rough) * (0.5 + 0.5 * push_ok) - 0.02 * min(cot, 10.0)
    return {
        "score": round(score, 4),
        "speed_flat": round(speed_flat, 3),
        "speed_rough": round(speed_rough, 3),
        "push_survived": bool(push_ok),
        "fell_flat": flat.fell,
        "cost_of_transport": round(cot, 3),
        "mass_kg": round(flat.mass, 2),
        "lateral_drift_m": round(flat.lateral, 3),
    }
