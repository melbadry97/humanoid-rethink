"""The recursive design loop.

    propose body -> build MJCF -> tune gait (inner ES) -> score task suite
        -> archive (MAP-Elites cells by body class x height) -> reflect -> repeat

Usage:
    python -m hrethink.loop --iters 30 --proposer mutation --out runs/demo
    ANTHROPIC_API_KEY=... python -m hrethink.loop --iters 30 --proposer llm --out runs/llm
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .genome import SEEDS, Genome
from .mjcf import build_xml
from .propose import LLMProposer, MutationProposer
from .sim import optimise_controller, score_design


def evaluate(g: Genome, budget: dict, seed: int) -> tuple[dict, np.ndarray]:
    res = optimise_controller(g, seed=seed, **budget)
    return score_design(g, res.params), res.params


def run(iters: int, proposer_name: str, out: Path, reflect_every: int, budget: dict, seed: int = 0) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    hist_path, lessons_path = out / "history.jsonl", out / "lessons.md"
    history = [json.loads(line) for line in hist_path.read_text().splitlines()] if hist_path.exists() else []
    lessons = lessons_path.read_text() if lessons_path.exists() else ""
    proposer = LLMProposer() if proposer_name == "llm" else MutationProposer(seed)
    elites: dict[str, dict] = {}
    for r in history:
        _update_elites(elites, r)

    def record(entry: dict, params: np.ndarray) -> None:
        history.append(entry)
        with hist_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        np.save(out / f"ctrl_{entry['id']:04d}.npy", params)
        if _update_elites(elites, entry):
            xml, _ = build_xml(Genome(**entry["genome"]))
            (out / f"elite_{entry['cell']}.xml").write_text(xml)
        m = entry["metrics"]
        print(f"[{entry['id']:3d}] score={m['score']:+.3f} flat={m['speed_flat']:.2f}m/s rough={m['speed_rough']:.2f}m/s "
              f"push={'ok' if m['push_survived'] else 'fell'} CoT={m['cost_of_transport']:.2f} cell={entry['cell']}  {entry['hypothesis'][:70]}")

    if not history:
        for i, (name, g) in enumerate(SEEDS.items()):
            metrics, params = evaluate(g, budget, seed + i)
            record(_entry(len(history), g, metrics, None, f"seed: {name}", "none"), params)

    for it in range(iters):
        t0 = time.time()
        try:
            p = proposer.propose(history, lessons)
        except Exception as e:  # keep the loop alive if one LLM call fails
            print(f"proposal failed ({e}); falling back to mutation")
            p = MutationProposer(seed + it).propose(history, lessons)
        metrics, params = evaluate(p["genome"], budget, seed + len(history))
        entry = _entry(len(history), p["genome"], metrics, p["parent_id"], p["hypothesis"], p["inspiration"])
        entry["seconds"] = round(time.time() - t0, 1)
        record(entry, params)
        if reflect_every and (it + 1) % reflect_every == 0:
            lessons = proposer.reflect(history, lessons)
            lessons_path.write_text(lessons)

    best = max(history, key=lambda r: r["metrics"]["score"])
    (out / "best.xml").write_text(build_xml(Genome(**best["genome"]))[0])
    (out / "summary.json").write_text(json.dumps({"best": best, "elites": elites}, indent=1))
    print(f"best #{best['id']} score={best['metrics']['score']:+.3f}: {best['hypothesis']}")
    return history


def _entry(i: int, g: Genome, metrics: dict, parent: int | None, hypothesis: str, inspiration: str) -> dict:
    body, h = g.descriptor()
    return {"id": i, "parent_id": parent, "hypothesis": hypothesis, "inspiration": inspiration,
            "cell": f"{body}-h{h}", "genome": g.model_dump(), "metrics": metrics}


def _update_elites(elites: dict, entry: dict) -> bool:
    cur = elites.get(entry["cell"])
    if cur is None or entry["metrics"]["score"] > cur["metrics"]["score"]:
        elites[entry["cell"]] = entry
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--proposer", choices=["mutation", "llm"], default="mutation")
    ap.add_argument("--out", type=Path, default=Path("runs/latest"))
    ap.add_argument("--reflect-every", type=int, default=5)
    ap.add_argument("--generations", type=int, default=20, help="inner ES generations per design")
    ap.add_argument("--popsize", type=int, default=24)
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    budget = {"generations": a.generations, "popsize": a.popsize, "duration": a.duration}
    run(a.iters, a.proposer, a.out, a.reflect_every, budget, a.seed)


if __name__ == "__main__":
    main()
