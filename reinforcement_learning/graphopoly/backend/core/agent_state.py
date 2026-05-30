"""Per-agent mutable state for Graphopoly."""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class AgentState:
    """Tracks one agent's position, ownership, prices, and statistics."""

    agent_id: int
    position: int = 0                       # current node
    destinations: list[int] = field(default_factory=list)
    owned_nodes: list[int] = field(default_factory=list)
    prices: dict[int, float] = field(default_factory=dict)  # {node_id: price}

    # Running stats (reset each episode)
    cumulative_reward: float = 0.0
    tax_revenue: float = 0.0               # total earned from other agents (and self)
    tax_paid: float = 0.0                  # total paid (including to self)
    dest_revenue: float = 0.0              # revenue from destination completions only
    trips_completed: int = 0

    # Destination tracking
    # Any destination yields a reward EXCEPT the one visited immediately before.
    last_visited_dest: int | None = None    # most recently completed destination node
    steps_on_current_node: int = 0          # consecutive steps on same node

    # Per-node visit frequency
    node_visit_counts: dict[int, int] = field(default_factory=dict)

    @property
    def net_profit(self) -> float:
        """Net profit = total revenue - taxes paid to others."""
        return self.cumulative_reward

    def complete_destination(self, node: int) -> None:
        """Called when agent earns a reward at a destination node."""
        self.last_visited_dest = node
        self.trips_completed += 1

    def reset(
        self,
        start_position: int,
        destinations: list[int],
        owned_nodes: list[int],
        price_budget: float = 100.0,
    ) -> None:
        """Reset for a new episode."""
        self.position = start_position
        self.destinations = destinations
        self.owned_nodes = owned_nodes
        # Uniform initial distribution of budget across owned nodes
        num_owned = max(len(owned_nodes), 1)
        uniform_price = price_budget / num_owned
        self.prices = {node: uniform_price for node in owned_nodes}
        self.cumulative_reward = 0.0
        self.tax_revenue = 0.0
        self.tax_paid = 0.0
        self.dest_revenue = 0.0
        self.trips_completed = 0
        self.last_visited_dest = None
        self.steps_on_current_node = 0
        self.node_visit_counts = {}

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "position": self.position,
            "destinations": self.destinations,
            "owned_nodes": self.owned_nodes,
            "prices": {str(k): v for k, v in self.prices.items()},
            "cumulative_reward": self.cumulative_reward,
            "tax_revenue": self.tax_revenue,
            "tax_paid": self.tax_paid,
            "dest_revenue": self.dest_revenue,
            "trips_completed": self.trips_completed,
            "last_visited_dest": self.last_visited_dest,
            "steps_on_current_node": self.steps_on_current_node,
            "node_visit_counts": {str(k): v for k, v in self.node_visit_counts.items()},
        }
