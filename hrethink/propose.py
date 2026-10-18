"""Design proposers for the outer loop.

`MutationProposer` needs nothing but a random seed. `LLMProposer` asks Claude
to read the archive and the running lessons file, state a falsifiable
hypothesis, and return a new genome; `reflect()` distils what the last batch
of experiments taught into lessons that steer the next proposals. That
reflection step is what makes the loop self-improving rather than a plain
random search.
"""
from __future__ import annotations

import json
import random
from typing import Optional

from pydantic import BaseModel, Field, create_model

from .genome import Genome, _bounds, mutate

MODEL = "claude-opus-5-5"


def _loose_genome_model() -> type[BaseModel]:
    """Same fields as Genome but with ranges moved into descriptions, so the
    structured-output schema stays simple; values are clipped afterwards."""
    fields = {}
    for name, f in Genome.model_fields.items():
        desc = f.description or name
        if f.annotation is float:
            lo, hi = _bounds(name)
            desc = f"{desc} [range {lo}..{hi}]"
        fields[name] = (f.annotation, Field(description=desc))
    return create_model("GenomeProposal", **fields)


GenomeProposal = _loose_genome_model()


class Proposal(BaseModel):
    hypothesis: str = Field(description="one falsifiable sentence: what this body should do better than its parent, and why")
    inspiration: str = Field(description="adjacent field or animal the idea borrows from, or 'none'")
    parent_id: Optional[int] = Field(description="id of the archive entry this design modifies, or null")
    genome: GenomeProposal  # type: ignore[valid-type]


class Reflection(BaseModel):
    lessons: list[str] = Field(description="short, evidence-backed rules learned so far; cite run ids")
    next_focus: str = Field(description="what the next batch of designs should explore")


SYSTEM = """You are the design lead in an automated humanoid-robot co-design loop.
Each design you propose is built in MuJoCo, gets its own gait tuned by an evolution
strategy, and is scored on a task suite: flat-ground speed, speed over rubble, surviving
a lateral shove, and cost of transport. Score = mean speed x push factor - small CoT penalty.
Your goal is a body that is more agile and useful than today's humanoids (homes, warehouses,
disaster sites) while keeping a humanoid upper body for working in human spaces.
Think like an engineer who borrows from biology, automotive, aerospace and materials.
Prefer bold but testable changes; avoid re-proposing designs already in the archive."""


def _archive_digest(history: list[dict], k: int = 12) -> str:
    top = sorted(history, key=lambda r: -r["metrics"]["score"])[:k]
    recent = history[-k:]
    rows = {r["id"]: r for r in top + recent}
    return json.dumps([
        {"id": r["id"], "hypothesis": r.get("hypothesis", ""), "genome": r["genome"], "metrics": r["metrics"]}
        for r in sorted(rows.values(), key=lambda r: r["id"])
    ], indent=1)


class MutationProposer:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def propose(self, history: list[dict], lessons: str) -> dict:
        pool = sorted(history, key=lambda r: -r["metrics"]["score"])[: max(3, len(history) // 3)]
        parent = self.rng.choice(pool)
        child = mutate(Genome(**parent["genome"]), self.rng)
        return {"genome": child, "parent_id": parent["id"], "hypothesis": "random mutation", "inspiration": "none"}

    def reflect(self, history: list[dict], lessons: str) -> str:
        return lessons


class LLMProposer:
    def __init__(self, model: str = MODEL, effort: str = "high"):
        import anthropic  # imported lazily so the mutation-only loop has no API dependency

        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort

    def _parse(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        resp = self.client.messages.parse(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM,
            output_config={"effort": self.effort},
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,
        )
        if resp.stop_reason == "refusal" or resp.parsed_output is None:
            raise RuntimeError(f"no usable proposal (stop_reason={resp.stop_reason})")
        return resp.parsed_output

    def propose(self, history: list[dict], lessons: str) -> dict:
        prompt = (
            f"Genome schema (fields, meaning, ranges):\n{json.dumps(GenomeProposal.model_json_schema()['properties'], indent=1)}\n\n"
            f"Lessons so far:\n{lessons or '(none yet)'}\n\n"
            f"Archive (best and most recent experiments):\n{_archive_digest(history)}\n\n"
            "Propose ONE new design."
        )
        p: Proposal = self._parse(prompt, Proposal)  # type: ignore[assignment]
        return {"genome": Genome.from_loose(p.genome.model_dump()), "parent_id": p.parent_id,
                "hypothesis": p.hypothesis, "inspiration": p.inspiration}

    def reflect(self, history: list[dict], lessons: str) -> str:
        prompt = (
            f"Current lessons:\n{lessons or '(none yet)'}\n\n"
            f"All experiments so far:\n{_archive_digest(history, k=40)}\n\n"
            "Which hypotheses held up and which did not? Rewrite the lessons list: keep what the "
            "evidence supports, drop what it contradicts, and say what to explore next."
        )
        r: Reflection = self._parse(prompt, Reflection)  # type: ignore[assignment]
        return "\n".join(f"- {line}" for line in r.lessons) + f"\n\nNext focus: {r.next_focus}\n"
