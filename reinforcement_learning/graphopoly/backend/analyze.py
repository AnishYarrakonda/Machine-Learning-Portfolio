"""
Episode analyzer for Graphopoly.

Reads the single simulation JSON produced by SimulationLogger and builds a
time-indexed metrics timeline suitable for the Analyze Mode frontend.

The input schema (simulation_*.json) is:
  metadata, graph, config, initial_state, trajectory, training_metrics,
  aggregate_stats

Each trajectory step already carries cumulative node_stats and agent_stats,
so the timeline is assembled by mapping those directly onto the schema expected
by chart-registry.js.
"""

from __future__ import annotations


class EpisodeAnalyzer:
    """
    Reads a simulation JSON file and exposes a per-timestep metrics timeline.

    Usage:
        analyzer = EpisodeAnalyzer()
        analyzer.load_episode(json_dict)
        timeline = analyzer.compute_metrics_timeline()
        # timeline[t] → {timestep, agents, nodes, system}
    """

    def __init__(self):
        self.episode_data     = None
        self.graph_data       = None
        self.config           = None
        self.trajectory       = []
        self.training_metrics = None
        self.aggregate_stats  = None
        self.num_steps        = 0
        self.num_agents       = 0
        self.num_nodes        = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Load
    # ──────────────────────────────────────────────────────────────────────────

    def load_episode(self, episode_json: dict) -> None:
        """Parse simulation JSON and extract key sections."""
        self.episode_data     = episode_json
        self.graph_data       = episode_json.get("graph", {})
        self.config           = episode_json.get("config", {})
        self.trajectory       = episode_json.get("trajectory", [])
        self.training_metrics = episode_json.get("training_metrics", {})
        self.aggregate_stats  = episode_json.get("aggregate_stats", {})

        meta             = episode_json.get("metadata", {})
        self.num_steps   = meta.get("num_steps", len(self.trajectory))
        self.num_agents  = meta.get("num_agents", self.config.get("agent", {}).get("num_agents", 0))
        self.num_nodes   = meta.get("num_nodes",  self.config.get("graph", {}).get("num_nodes", 0))

    # ──────────────────────────────────────────────────────────────────────────
    # Timeline builder
    # ──────────────────────────────────────────────────────────────────────────

    def compute_metrics_timeline(self) -> list[dict]:
        """
        Build a flat list of per-timestep metric dicts.

        Each entry:
          {
            "timestep": t,
            "agents":   {str(agent_id): {metric_name: value, ...}},
            "nodes":    {str(node_id):  {metric_name: value, ...}},
            "system":   {metric_name:   value}
          }

        Because the trajectory already stores cumulative agent_stats and
        node_stats at every step, this is mostly a reformat.
        """
        timeline: list[dict] = []

        # Running system totals for reward & price (we derive these on the fly)
        for step_data in self.trajectory:
            t          = step_data.get("step", 0)
            a_stats    = step_data.get("agent_stats", {})
            n_stats    = step_data.get("node_stats",  {})
            prices_now = step_data.get("prices", {})
            rewards    = step_data.get("rewards", {})

            # ── Agent metrics ─────────────────────────────────────────────────
            agents_out: dict[str, dict] = {}
            for aid in range(self.num_agents):
                aid_s = str(aid)
                a = a_stats.get(aid_s, {})
                agents_out[aid_s] = {
                    "cumulative_reward": a.get("total_profit",   0.0),
                    "net_profit":        a.get("total_profit",   0.0),
                    "tax_revenue":       a.get("tax_revenue",    0.0),
                    "tax_paid":          a.get("tax_paid",       0.0),
                    "dest_revenue":      a.get("dest_revenue",   0.0),
                    "trips_completed":   a.get("trips_completed", 0),
                    "step_reward":       float(rewards.get(aid_s, 0.0)),
                    # total visits = sum over all nodes from agent_stats
                    "total_visits":      sum(
                        n_stats.get(str(n), {}).get("visits", 0)
                        for n in range(self.num_nodes)
                    ),
                }

            # ── Node metrics ──────────────────────────────────────────────────
            nodes_out: dict[str, dict] = {}
            ownership  = self.graph_data.get("ownership", {})
            for nid in range(self.num_nodes):
                nid_s = str(nid)
                n = n_stats.get(nid_s, {})
                nodes_out[nid_s] = {
                    "total_visits":         n.get("visits",            0),
                    "revenue_collected":    n.get("revenue_collected", 0.0),
                    "current_price":        int(prices_now.get(nid_s, 0)),
                    "owner":               int(ownership.get(nid_s, -1))
                                            if isinstance(ownership.get(nid_s), (int, str)) else -1,
                    # avg visits/step
                    "avg_visits_per_step":  round(n.get("visits", 0) / max(t + 1, 1), 5),
                }

            # ── System metrics ────────────────────────────────────────────────
            total_reward = sum(
                agents_out[str(a)]["cumulative_reward"]
                for a in range(self.num_agents)
            )
            avg_price = (
                sum(int(prices_now.get(str(n), 0)) for n in range(self.num_nodes))
                / max(self.num_nodes, 1)
            )

            system_out = {
                "total_system_reward":    round(total_reward, 3),
                "avg_node_price":         round(avg_price, 3),
                "revenue_distribution":   {
                    str(a): agents_out[str(a)]["cumulative_reward"]
                    for a in range(self.num_agents)
                },
            }

            timeline.append({
                "timestep": t,
                "agents":   agents_out,
                "nodes":    nodes_out,
                "system":   system_out,
            })

        return timeline

    def get_timestep_slice(self, t: int, timeline: list[dict] | None = None) -> dict:
        """Get all metrics at a specific timestep."""
        if timeline is None or t < 0 or t >= len(timeline):
            return {}
        return timeline[t]
