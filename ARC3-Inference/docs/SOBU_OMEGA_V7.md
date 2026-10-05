# Duck-SOBU-Ω V7 experiment

This branch adds an evidence-first controller candidate without replacing the
original Duck planner.

## Design
- Goal progress, not raw board movement, is the primary success signal.
- Novel actions fail open.
- Repeated empirically useless/fatal state-action pairs can be vetoed.
- Decision-relevant information decays with repeated probes.
- Selection uses a lower-confidence-bound utility.
- The module is dependency-free so it can run in the Kaggle harness.

## Promotion protocol
Run the same model, games, passes, hardware, runtime/action budget and seed for:
1. Duck baseline.
2. SOBU shadow (score/log only; no action changes).
3. SOBU control.

Promote only with paired evidence: score up, levels non-worse, actions/level
non-worse, dead-work non-worse, plus the repository significance check. Until
then this is a candidate, not a claimed leaderboard improvement.
