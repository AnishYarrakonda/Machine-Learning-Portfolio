"""
Single-file simulation logger for Graphopoly.

Writes exactly ONE JSON file per simulation run containing:
  - metadata          (run info, timestamps, dimensions)
  - graph             (nodes, edges, ownership, positions)
  - config            (full hyperparameter snapshot)
  - initial_state     (starting positions, destinations, prices)
  - trajectory        (every step of the FINAL episode)
  - training_metrics  (rewards + losses for EVERY episode, for learning curves)
  - aggregate_stats   (final-episode per-agent and per-node aggregates)

No CSVs. No per-episode files. One file. Full picture.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.config import GraphopolyConfig
    from backend.core.graph_world import GraphWorld
    from backend.core.agent_state import AgentState


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _five_point_summary(values: list[float]) -> dict:
    """Compute min / Q1 / median / Q3 / max from a list of values."""
    if not values:
        return {"min": 0, "q1": 0, "median": 0, "q3": 0, "max": 0}
    s = sorted(values)
    n = len(s)

    def percentile(p: float) -> float:
        idx = (n - 1) * p / 100
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        return s[lo] + (s[hi] - s[lo]) * (idx - lo)

    return {
        "min":    round(s[0], 3),
        "q1":     round(percentile(25), 3),
        "median": round(percentile(50), 3),
        "q3":     round(percentile(75), 3),
        "max":    round(s[-1], 3),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main Logger
# ──────────────────────────────────────────────────────────────────────────────

class SimulationLogger:
    """
    Accumulates stats across an entire training run and writes a single
    self-contained JSON file at the end.

    Train-loop contract
    -------------------
    After env.reset()       → call log_initial_state(agents, world)
    Inside step loop        → call log_step(step, agents, actions, rewards, info)
    After each episode      → call log_episode_end(episode, agents, losses)
    After training loop     → call finalize(agents, world) to write the file

    Only the FINAL episode's trajectory is stored in the output (it reflects the
    trained policy). Training metrics (rewards, losses) cover every episode.
    """

    def __init__(self, log_dir: str, config: "GraphopolyConfig"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

        self._run_ts     = int(time.time())
        self._start_iso  = datetime.now(timezone.utc).isoformat()

        # ── Graph / world (set once by save_graph) ──────────────────────────
        self._world: "GraphWorld | None"   = None
        self._layout: dict[int, tuple[float, float]] | None = None

        # ── Initial state (reset each episode, captured after env.reset) ────
        self._initial_state: dict | None = None

        # ── Step accumulator for CURRENT episode ────────────────────────────
        self._cur_trajectory: list[dict]            = []
        self._cur_node_visits: dict[int, int]       = defaultdict(int)
        self._cur_node_tax:    dict[int, float]     = defaultdict(float)
        self._cur_price_hist:  dict[int, list[int]] = defaultdict(list)

        # ── Training-level metrics (one entry per episode) ───────────────────
        self._ep_rewards:  list[dict[str, float]] = []   # {str(aid): reward}
        self._ep_trips:    list[dict[str, int]]   = []   # {str(aid): trips}
        self._ep_policy_loss:  list[float]        = []
        self._ep_value_loss:   list[float]        = []
        self._ep_entropy:      list[float]        = []

        # ── Final-episode state (kept after last log_episode_end call) ───────
        self._final_trajectory: list[dict]            = []
        self._final_node_visits: dict[int, int]       = {}
        self._final_node_tax:    dict[int, float]     = {}
        self._final_price_hist:  dict[int, list[int]] = {}
        self._final_agents: list["AgentState"] | None = None
        self._final_step_count: int                   = 0

        # ── Run counters ──────────────────────────────────────────────────────
        self._total_episodes_run = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Setup
    # ──────────────────────────────────────────────────────────────────────────

    def save_graph(self, world: "GraphWorld") -> None:
        """Cache graph topology (call once before training loop)."""
        self._world  = world
        self._layout = world.get_spring_layout()

    # ──────────────────────────────────────────────────────────────────────────
    # Per-episode reset  →  call right after env.reset()
    # ──────────────────────────────────────────────────────────────────────────

    def log_initial_state(self, agents: list["AgentState"]) -> None:
        """Capture the initial conditions at the start of an episode."""
        self._cur_trajectory.clear()
        self._cur_node_visits.clear()
        self._cur_node_tax.clear()
        self._cur_price_hist.clear()

        prices: dict[str, int] = {}
        for agent in agents:
            for node_id, price in agent.prices.items():
                prices[str(node_id)] = price

        self._initial_state = {
            "agent_positions": {str(a.agent_id): a.position for a in agents},
            "agent_destinations": {
                str(a.agent_id): a.destinations for a in agents
            },
            "agent_owned_nodes": {
                str(a.agent_id): a.owned_nodes for a in agents
            },
            "prices": prices,
            "agent_stats": {
                str(a.agent_id): {
                    "trips_completed": 0,
                    "total_profit":    0.0,
                    "tax_revenue":     0.0,
                    "tax_paid":        0.0,
                    "dest_revenue":    0.0,
                }
                for a in agents
            },
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Per-step logging
    # ──────────────────────────────────────────────────────────────────────────

    def log_step(
        self,
        step: int,
        agents:  list["AgentState"],
        actions: list[dict],
        rewards: list[float],
        info:    dict,
    ) -> None:
        """Record one step of the current episode."""

        # ── Accumulate node visit / tax counters ─────────────────────────────
        for agent in agents:
            self._cur_node_visits[agent.position] += 1

        for payer_id_raw, receivers in info.get("taxes", {}).items():
            payer_id = int(payer_id_raw)
            payer_pos = agents[payer_id].position
            for recv_id_raw, amount in receivers.items():
                if int(recv_id_raw) != payer_id:           # exclude self-tax
                    self._cur_node_tax[payer_pos] += amount

        # ── Track price history per owned node ───────────────────────────────
        for agent in agents:
            for node_id, price in agent.prices.items():
                self._cur_price_hist[node_id].append(price)

        # ── Build cumulative node_stats snapshot for this step ───────────────
        node_stats: dict[str, dict] = {}
        if self._world:
            for nid in range(self._world.num_nodes):
                node_stats[str(nid)] = {
                    "visits":            self._cur_node_visits.get(nid, 0),
                    "revenue_collected": round(self._cur_node_tax.get(nid, 0.0), 3),
                }

        # ── Build cumulative agent_stats snapshot ────────────────────────────
        agent_stats: dict[str, dict] = {
            str(a.agent_id): {
                "trips_completed": a.trips_completed,
                "total_profit":    round(a.cumulative_reward, 3),
                "tax_revenue":     round(a.tax_revenue, 3),
                "tax_paid":        round(a.tax_paid, 3),
                "dest_revenue":    round(a.dest_revenue, 3),
            }
            for a in agents
        }

        # ── Build current prices dict (all owned nodes) ──────────────────────
        prices: dict[str, int] = {}
        for agent in agents:
            for nid, price in agent.prices.items():
                prices[str(nid)] = price

        # ── Assemble step record ─────────────────────────────────────────────
        step_record: dict = {
            "step":            step,
            "agent_positions": {str(a.agent_id): a.position for a in agents},
            "actions": {
                str(i): {
                    "move": act["move"],
                    "price_changes": {
                        str(k): v
                        for k, v in act.get("price_changes", {}).items()
                    },
                }
                for i, act in enumerate(actions)
            },
            "prices":  prices,
            "rewards": {str(i): round(r, 4) for i, r in enumerate(rewards)},
            "taxes": {
                str(k): {str(k2): v2 for k2, v2 in v.items()}
                for k, v in info.get("taxes", {}).items()
            },
            "dest_completions": [
                {"agent": aid, "node": node}
                for aid, node in info.get("dest_completions", [])
            ],
            "node_stats":  node_stats,
            "agent_stats": agent_stats,
        }

        self._cur_trajectory.append(step_record)

    # ──────────────────────────────────────────────────────────────────────────
    # Per-episode end  →  call after PPO update
    # ──────────────────────────────────────────────────────────────────────────

    def log_episode_end(
        self,
        episode:  int,
        agents:   list["AgentState"],
        losses:   dict[int, dict[str, float]],
    ) -> None:
        """
        Store training-level metrics and snapshot the current episode as the
        'final' candidate (the last episode seen wins).
        """
        self._total_episodes_run = episode + 1

        # ── Training metrics ──────────────────────────────────────────────────
        self._ep_rewards.append(
            {str(a.agent_id): round(a.cumulative_reward, 3) for a in agents}
        )
        self._ep_trips.append(
            {str(a.agent_id): a.trips_completed for a in agents}
        )

        # Average PPO losses across agents this episode
        pol, val, ent = [], [], []
        for aid_losses in losses.values():
            pol.append(aid_losses.get("policy_loss", 0.0))
            val.append(aid_losses.get("value_loss",  0.0))
            ent.append(aid_losses.get("entropy",     0.0))
        self._ep_policy_loss.append(round(sum(pol) / len(pol) if pol else 0.0, 6))
        self._ep_value_loss .append(round(sum(val) / len(val) if val else 0.0, 6))
        self._ep_entropy    .append(round(sum(ent) / len(ent) if ent else 0.0, 6))

        # ── Snapshot this episode as the working "final" ──────────────────────
        self._final_trajectory   = list(self._cur_trajectory)
        self._final_node_visits  = dict(self._cur_node_visits)
        self._final_node_tax     = dict(self._cur_node_tax)
        self._final_price_hist   = {k: list(v) for k, v in self._cur_price_hist.items()}
        self._final_agents       = agents
        self._final_step_count   = len(self._cur_trajectory)

    # ──────────────────────────────────────────────────────────────────────────
    # Finalize  →  write the single output file
    # ──────────────────────────────────────────────────────────────────────────

    def finalize(self) -> str:
        """
        Write the single simulation JSON file and return its path.
        Also writes to episodes/temp_latest.json for the web server.
        """
        doc = self._build_document()
        path = self._write(doc)
        self._write_temp_latest(doc)
        return str(path)

    # ──────────────────────────────────────────────────────────────────────────
    # Document builder
    # ──────────────────────────────────────────────────────────────────────────

    def _build_document(self) -> dict:
        world   = self._world
        agents  = self._final_agents or []
        config  = self.config
        N       = world.num_nodes if world else 0
        A       = config.agent.num_agents

        # ── metadata ─────────────────────────────────────────────────────────
        metadata = {
            "episode_id":     f"sim_{self._run_ts}",
            "timestamp":      self._start_iso,
            "finished_at":    datetime.now(timezone.utc).isoformat(),
            "num_steps":      self._final_step_count,
            "num_episodes":   self._total_episodes_run,
            "num_agents":     A,
            "num_nodes":      N,
            "description":    "",
        }

        # ── graph ─────────────────────────────────────────────────────────────
        layout = self._layout or {}
        graph_section: dict = {
            "nodes": [],
            "edges": [],
            "ownership":   {},
            "destinations": {},
            "starting_positions": {},
        }
        if world:
            graph_section["nodes"] = [
                {
                    "id":       nid,
                    "owner":    world.ownership.get(nid, -1),
                    "position": list(layout.get(nid, (0.0, 0.0))),
                }
                for nid in range(N)
            ]
            graph_section["edges"]             = [list(e) for e in world.graph.edges()]
            graph_section["ownership"]         = {str(k): v for k, v in world.ownership.items()}
            graph_section["destinations"]      = {str(k): v for k, v in world.destinations.items()}
            graph_section["starting_positions"] = {str(k): v for k, v in world.starting_positions.items()}

        # ── config ────────────────────────────────────────────────────────────
        config_section = config.to_dict()

        # ── initial_state ─────────────────────────────────────────────────────
        initial_state = self._initial_state or {}

        # ── trajectory ───────────────────────────────────────────────────────
        # The final episode's full step-by-step record.
        trajectory = self._final_trajectory

        # ── training_metrics ─────────────────────────────────────────────────
        training_metrics = {
            "episode_rewards":     self._ep_rewards,       # [{str(aid): reward}, ...]
            "episode_trips":       self._ep_trips,         # [{str(aid): trips}, ...]
            "losses": {
                "policy_loss":     self._ep_policy_loss,   # [float × num_episodes]
                "value_loss":      self._ep_value_loss,
                "entropy_bonus":   self._ep_entropy,
            },
            "num_episodes_trained": self._total_episodes_run,
        }

        # ── aggregate_stats (final episode) ──────────────────────────────────
        steps = max(self._final_step_count, 1)

        agent_agg: dict[str, dict] = {}
        for agent in agents:
            aid = str(agent.agent_id)
            agent_agg[aid] = {
                "total_trips":           agent.trips_completed,
                "total_profit":          round(agent.cumulative_reward, 3),
                "average_profit_per_step": round(agent.cumulative_reward / steps, 6),
                "total_tax_revenue":     round(agent.tax_revenue, 3),
                "total_tax_paid":        round(agent.tax_paid, 3),
                "total_dest_revenue":    round(agent.dest_revenue, 3),
                "owned_nodes":           agent.owned_nodes,
                "destination_nodes":     agent.destinations,
                "final_prices":          {str(k): v for k, v in agent.prices.items()},
                "node_visit_counts":     {str(k): v for k, v in agent.node_visit_counts.items()},
                "last_visited_dest":     agent.last_visited_dest,
            }

        node_agg: dict[str, dict] = {}
        if world:
            dest_of: dict[int, list[int]] = defaultdict(list)
            for aid, dests in world.destinations.items():
                for nid in dests:
                    dest_of[nid].append(aid)

            for nid in range(N):
                prices_hist = self._final_price_hist.get(nid, [])
                visits      = self._final_node_visits.get(nid, 0)
                tax_total   = self._final_node_tax.get(nid, 0.0)
                node_agg[str(nid)] = {
                    "owner":                world.ownership.get(nid, -1),
                    "is_destination_of":    dest_of.get(nid, []),
                    "total_visits":         visits,
                    "avg_visits_per_step":  round(visits / steps, 6),
                    "total_revenue_collected": round(tax_total, 3),
                    "current_price":        prices_hist[-1] if prices_hist else 0,
                    "average_price":        round(sum(prices_hist) / len(prices_hist), 3) if prices_hist else 0,
                    "price_stats":          _five_point_summary(prices_hist),
                    "visits_by_agent": {
                        str(a.agent_id): a.node_visit_counts.get(nid, 0)
                        for a in agents
                    },
                }

        system_agg = {
            "total_system_profit":    round(sum(a.cumulative_reward for a in agents), 3),
            "total_trips_all_agents": sum(a.trips_completed for a in agents),
            "total_taxes_transferred": round(
                sum(self._final_node_tax.values()), 3
            ),
        }

        aggregate_stats = {
            "agents": agent_agg,
            "nodes":  node_agg,
            "system": system_agg,
        }

        return {
            "metadata":         metadata,
            "graph":            graph_section,
            "config":           config_section,
            "initial_state":    initial_state,
            "trajectory":       trajectory,
            "training_metrics": training_metrics,
            "aggregate_stats":  aggregate_stats,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # I/O helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _write(self, doc: dict) -> Path:
        path = self.log_dir / f"simulation_{self._run_ts}.json"
        with open(path, "w") as f:
            json.dump(doc, f, separators=(",", ":"))
        return path

    def _write_temp_latest(self, doc: dict) -> None:
        """Keep episodes/temp_latest.json in sync for /api/episode/latest."""
        # Write to project root episodes/ dir (parent of backend/)
        temp_dir = Path(__file__).parent.parent / "episodes"
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / "temp_latest.json"
        with open(temp_path, "w") as f:
            json.dump(doc, f, separators=(",", ":"))


# ──────────────────────────────────────────────────────────────────────────────
# Backward-compatibility alias so existing imports keep working
# ──────────────────────────────────────────────────────────────────────────────
EpisodeLogger = SimulationLogger
