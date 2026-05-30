"""
Shared utilities for training and simulation loops.

Provides step-record building helpers so train.py and simulate.py
stay in sync without duplicating the serialization logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.core.agent_state import AgentState


def build_step_record(
    step: int,
    agents: list["AgentState"],
    actions: list[dict],
    rewards: list[float],
    info: dict,
) -> dict:
    """Serialize one environment step into a lightweight frontend-friendly dict.

    Used by both train.py (episodic) and simulate.py (continuous) to keep
    the step_history format identical for the frontend animation layer.
    """
    return {
        "step": step,
        "positions": [a.position for a in agents],
        "prices": {
            str(n): round(a.prices.get(n, 0.0), 2)
            for a in agents
            for n in a.owned_nodes
        },
        "actions": [
            {
                "move": act["move"],
                "price_changes": {
                    str(k): v
                    for k, v in act.get("price_changes", {}).items()
                },
            }
            for act in actions
        ],
        "rewards": [round(r, 3) for r in rewards],
        "taxes": {
            str(payer): {str(recv): amt for recv, amt in recvs.items()}
            for payer, recvs in info.get("taxes", {}).items()
        },
        "dest_completions": [
            {"agent": aid, "node": node}
            for aid, node in info.get("dest_completions", [])
        ],
    }


def build_initial_step(agents: list["AgentState"]) -> dict:
    """Build the step-0 record used at the start of an episode / simulation."""
    return {
        "step": 0,
        "positions": [a.position for a in agents],
        "prices": {
            str(n): round(a.prices.get(n, 0.0), 2)
            for a in agents
            for n in a.owned_nodes
        },
        "actions": [],
        "rewards": [],
        "taxes": {},
        "dest_completions": [],
    }