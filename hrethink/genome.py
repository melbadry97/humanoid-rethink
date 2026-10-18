"""Parametric body plan ("genome") for humanoid-ish robots.

The genome is deliberately small and human-readable so an LLM can reason
about it, propose edits, and explain why. Every field has a physical meaning
and a bounded range; `Genome.clamped()` keeps proposals inside them.
"""
from __future__ import annotations

import random
from typing import Literal

from pydantic import BaseModel, Field


class Genome(BaseModel):
    # Trunk
    pelvis_len: float = Field(0.25, ge=0.15, le=0.8, description="pelvis length along walking direction (m); long pelvis + 2 leg pairs = centaur")
    pelvis_width: float = Field(0.26, ge=0.15, le=0.45, description="hip-to-hip width (m)")
    chest_height: float = Field(0.35, ge=0.0, le=0.6, description="upper body height above pelvis (m); 0 = no upper body")
    spine: bool = Field(True, description="actuated pitch joint between pelvis and chest (cheetah-like)")
    spine_stiffness: float = Field(20.0, ge=0.0, le=200.0, description="passive spring on the spine (Nm/rad)")

    # Legs
    leg_pairs: Literal[1, 2] = Field(1, description="1 = biped, 2 = four legs under the pelvis (upper body stays humanoid)")
    thigh_len: float = Field(0.32, ge=0.12, le=0.55)
    shin_len: float = Field(0.32, ge=0.12, le=0.55)
    knee_dir: Literal["forward", "backward"] = Field("forward", description="forward = human knee, backward = bird/digitigrade")
    foot: Literal["flat", "point", "wheel"] = Field("flat", description="flat footplate, point foot (ANYmal-like), or powered wheel")
    foot_len: float = Field(0.18, ge=0.06, le=0.3, description="flat-foot length (m)")
    toe: bool = Field(False, description="extra toe hinge on flat feet")
    knee_spring: float = Field(0.0, ge=0.0, le=150.0, description="passive knee tendon stiffness (Nm/rad), kangaroo-like energy storage")
    ankle_spring: float = Field(0.0, ge=0.0, le=150.0, description="passive ankle stiffness (Nm/rad)")

    # Upper body and appendages
    arms: bool = Field(True, description="two 2-DOF arms on the chest (used for balance / momentum)")
    arm_len: float = Field(0.5, ge=0.2, le=0.8)
    tail_len: float = Field(0.0, ge=0.0, le=0.8, description="actuated tail for counterbalance; 0 = none")

    # Actuation
    max_torque: float = Field(80.0, ge=20.0, le=250.0, description="per-joint torque limit (Nm); drives energy cost")

    @classmethod
    def from_loose(cls, data: dict) -> "Genome":
        """Build a genome from possibly out-of-range values (e.g. LLM output) by clipping."""
        data = {k: v for k, v in data.items() if k in cls.model_fields}
        for name, v in data.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and cls.model_fields[name].annotation is float:
                lo, hi = _bounds(name)
                data[name] = float(min(hi, max(lo, v)))
        return cls(**data)

    def leg_len(self) -> float:
        return self.thigh_len + self.shin_len

    def descriptor(self) -> tuple[str, int]:
        """Quality-diversity cell: (body class, standing-height bin)."""
        body = f"{self.leg_pairs * 2}leg-{self.foot}-{self.knee_dir[0]}"
        height = self.leg_len() + self.chest_height
        return body, min(4, int(height / 0.3))


def _bounds(name: str) -> tuple[float, float]:
    field = Genome.model_fields[name]
    lo = hi = None
    for m in field.metadata:
        lo = getattr(m, "ge", lo)
        hi = getattr(m, "le", hi)
    return lo, hi


def mutate(g: Genome, rng: random.Random, rate: float = 0.3, scale: float = 0.15) -> Genome:
    """Random mutation: Gaussian on floats (relative to range), flips on discrete genes."""
    data = g.model_dump()
    for name, field in Genome.model_fields.items():
        if rng.random() > rate:
            continue
        v = data[name]
        if isinstance(v, bool):
            data[name] = not v
        elif isinstance(v, float):
            lo, hi = _bounds(name)
            data[name] = v + rng.gauss(0, scale * (hi - lo))
        else:  # Literal choice
            data[name] = rng.choice(list(field.annotation.__args__))
    return Genome.from_loose(data)


SEEDS: dict[str, Genome] = {
    "baseline_humanoid": Genome(),
    "digitigrade_runner": Genome(knee_dir="backward", foot="point", knee_spring=60.0, tail_len=0.4, arms=False),
    "wheel_legged": Genome(foot="wheel", chest_height=0.4),
    "centaur": Genome(leg_pairs=2, pelvis_len=0.6, foot="point", thigh_len=0.25, shin_len=0.25),
}
