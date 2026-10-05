"""Duck-SOBU-Ω evidence controller.

Pure, dependency-free decision support. It does not replace Duck's planner; it
scores evidence around candidate actions so the harness can shadow-test it
before control is enabled.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Hashable

@dataclass
class BetaEvidence:
    success: float = 1.0
    failure: float = 1.0
    fatal: float = 1.0
    safe: float = 1.0
    def p_success(self) -> float:
        return self.success / (self.success + self.failure)
    def p_fatal(self) -> float:
        return self.fatal / (self.fatal + self.safe)
    def uncertainty(self) -> float:
        n = self.success + self.failure
        p = self.p_success()
        return math.sqrt(max(0.0, p * (1.0-p) / (n + 1.0)))

@dataclass(frozen=True)
class Utility:
    goal: float
    information: float
    synergy: float
    repeat: float
    inconsistency: float
    fatal: float
    latency: float
    value: float
    lcb: float
    hard_veto: bool

@dataclass
class SobuOmegaController:
    beta: float = 0.30
    eta: float = 0.12
    lambda_repeat: float = 0.22
    lambda_inconsistency: float = 0.10
    lambda_fatal: float = 0.75
    lambda_latency: float = 0.02
    lcb_z: float = 0.70
    veto_min_observations: int = 2
    veto_noop_rate: float = 0.95
    veto_fatal_rate: float = 0.50
    stats: dict[tuple[Hashable, str], BetaEvidence] = field(default_factory=dict)
    visits: dict[tuple[Hashable, str], int] = field(default_factory=dict)

    def _key(self, state_key: Hashable, action: str) -> tuple[Hashable, str]:
        return (state_key, action)

    def observe(self, state_key: Hashable, action: str, *,
                goal_progress: bool, board_changed: bool,
                fatal: bool = False) -> None:
        key = self._key(state_key, action)
        e = self.stats.setdefault(key, BetaEvidence())
        # Goal progress is the primary target. Board change is only weak evidence
        # and never counts as success by itself.
        if goal_progress:
            e.success += 1.0
        else:
            e.failure += 1.0
        if fatal:
            e.fatal += 1.0
        else:
            e.safe += 1.0
        self.visits[key] = self.visits.get(key, 0) + 1

    def assess(self, state_key: Hashable, action: str, *,
               predicted_goal_gain: float = 0.0,
               decision_information: float = 0.0,
               synergy: float = 0.0,
               inconsistency: float = 0.0,
               latency_cost: float = 0.0) -> Utility:
        key = self._key(state_key, action)
        e = self.stats.get(key, BetaEvidence())
        n = self.visits.get(key, 0)
        p = e.p_success()
        p_fatal = e.p_fatal()
        repeat = min(1.0, n / 3.0)
        info = max(0.0, decision_information) / math.sqrt(1.0 + n)
        goal = predicted_goal_gain + 0.20 * (p - 0.5)
        value = (
            goal + self.beta * info + self.eta * max(0.0, synergy)
            - self.lambda_repeat * repeat
            - self.lambda_inconsistency * max(0.0, inconsistency)
            - self.lambda_fatal * p_fatal
            - self.lambda_latency * max(0.0, latency_cost)
        )
        sigma = e.uncertainty()
        lcb = value - self.lcb_z * sigma
        noop_rate = (e.failure - 1.0) / max(1.0, n)
        fatal_rate = (e.fatal - 1.0) / max(1.0, n)
        hard_veto = n >= self.veto_min_observations and (
            noop_rate >= self.veto_noop_rate or fatal_rate >= self.veto_fatal_rate
        )
        return Utility(goal, info, max(0.0, synergy), repeat,
                       max(0.0, inconsistency), p_fatal,
                       max(0.0, latency_cost), value, lcb, hard_veto)

    def choose(self, state_key: Hashable, candidates: list[str], **features) -> str:
        viable = []
        for action in candidates:
            per_action = {k: (v.get(action, 0.0) if isinstance(v, dict) else v)
                          for k, v in features.items()}
            u = self.assess(state_key, action, **per_action)
            if not u.hard_veto:
                viable.append((u.lcb, action))
        if not viable:
            # Fail open: never deadlock the original Duck planner.
            return candidates[0]
        return max(viable, key=lambda item: item[0])[1]
