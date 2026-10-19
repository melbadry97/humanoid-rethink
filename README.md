# humanoid-rethink

Rethinking the humanoid body plan with a **recursive, LLM-driven design loop** in MuJoCo.

Instead of hand-designing one more 30-DOF biped, the loop treats the body itself as the
thing being learned:

```
            ┌──────────────────────────────────────────────────────────────┐
            │                      lessons.md (memory)                     │
            └──────────────▲───────────────────────────────┬───────────────┘
                           │ reflect every N runs          │ read
   ┌───────────────┐   ┌───┴──────────┐   ┌────────────┐   ▼   ┌──────────────┐
   │ archive       │──▶│ proposer     │──▶│ genome →   │──────▶│ inner loop:  │
   │ (MAP-Elites:  │   │ Claude or    │   │ MJCF body  │       │ ES tunes a   │
   │ body class ×  │   │ random       │   └────────────┘       │ CPG gait     │
   │ height)       │◀──┤ mutation     │◀──── task-suite score ─┤ per body     │
   └───────────────┘   └──────────────┘   flat · rubble · push └──────────────┘
                                          · cost of transport
```

Three nested levels of improvement:

1. **Controller** (inner): an evolution strategy tunes a central-pattern-generator gait with IMU
   feedback for each body, so very different bodies are compared fairly.
2. **Body** (outer): a proposer (Claude via the Anthropic API, or random mutation as a
   baseline) writes a new genome plus a falsifiable hypothesis, citing the archive.
3. **Proposer** (meta): every few runs Claude rewrites `lessons.md` from the evidence, which
   steers all later proposals. This is the self-improving part.

## Design space (`hrethink/genome.py`)

Bipeds or four-legged "centaur" bases under a humanoid upper body; human (forward) or
bird/digitigrade (backward) knees; flat feet with optional toes, point feet, or powered
wheels; passive tendon springs at knee and ankle; actuated spine; arms; counterbalancing
tail; per-joint torque limit. Four seeds ship: `baseline_humanoid`, `digitigrade_runner`,
`wheel_legged`, `centaur`.

## Task suite (`hrethink/sim.py::score_design`)

"Modern needs" made measurable: forward speed on flat ground, speed over random rubble,
surviving a lateral shove, and cost of transport. Score is
`mean speed × push factor − 0.02 × CoT`. Edit it to match what you care about.

## Run it

```bash
pip install -e ".[llm,dev]"
pytest -q                                                   # ~1 s smoke tests
python -m hrethink.loop --iters 30 --proposer mutation --out runs/demo_mutation
ANTHROPIC_API_KEY=... python -m hrethink.loop --iters 30 --proposer llm --out runs/llm
python -m mujoco.viewer --mjcf runs/llm/best.xml            # look at the winner (desktop)
```

Each design costs ~5–15 s on a 4-core CPU at the default budget (20 ES generations × 24
rollouts × 8 s sim). The LLM proposer uses `claude-opus-5-5` with structured outputs.

Outputs per run: `history.jsonl` (every design, hypothesis, metrics), `lessons.md`,
`elite_<cell>.xml` per MAP-Elites cell, `best.xml`, `summary.json`.

## Known limits and next steps

- The CPG + ES gait is deliberately cheap and weak; absolute speeds are low. Use it to rank
  bodies, then retrain the shortlisted ones with PPO in MuJoCo Playground (MJX/Warp on GPU)
  or Isaac Lab, and do sim-to-real checks with domain randomisation.
- Let the LLM also edit the task suite and the controller parameterisation (Eureka-style
  reward writing), not just the body.
- Add a hands/manipulation task, stairs, and a fall-and-get-up test.
- Export elites to URDF for Isaac Lab and for competition entries.

See [`docs/RESEARCH.md`](docs/RESEARCH.md) for the landscape survey (simulators, agentic AI in
robotics, competitions with deadlines, what has been tried, and adjacent-field ideas).

## Results so far

| Run | Proposer | Designs | Best score | Best rubble speed | Best body |
| --- | --- | --- | --- | --- | --- |
| `runs/demo_mutation` | random mutation | 4 seeds + 30 | 0.86 | 0.10 m/s | 4 wheels, humanoid torso |
| `runs/llm_v1` | Claude (`claude-opus-5-5`), reflect every 5 | 4 seeds + 20 | 1.48 | 1.33 m/s | 4 wheels on a long (0.8 m), wide (0.45 m) base, low chest, short arms, backward knees with springs |

Caveats Claude itself wrote into `runs/llm_v1/lessons.md`: near-identical genomes can score very
differently on rubble, so wins need replicate seeds; `foot_len` has no effect on wheeled feet; the best
body sits at the edge of the pelvis length and width ranges, so the ranges should widen.
