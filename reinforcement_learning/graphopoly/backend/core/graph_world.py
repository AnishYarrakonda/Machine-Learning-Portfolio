"""
Graph representation for Graphopoly.

Wraps a NetworkX graph with territory assignment, destination placement,
and serialization for save/load.
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np


class GraphWorld:
    """Owns the graph topology, node ownership, and destination assignments."""

    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.num_nodes: int = graph.number_of_nodes()

        # Populated by assign_territories / assign_destinations or from_custom
        self.ownership: dict[int, int] = {}              # {node_id: agent_id} — every node must have an owner
        self.destinations: dict[int, list[int]] = {}     # {agent_id: [node_ids]}
        self.starting_positions: dict[int, int] = {}     # {agent_id: node_id}

        # Cache
        self._adj_list: dict[int, list[int]] | None = None
        self._diameter: int | None = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @staticmethod
    def random_connected(
        num_nodes: int,
        num_edges: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> GraphWorld:
        """Generate a random connected graph with exactly `num_edges` edges.

        Algorithm:
            1. Build a random spanning tree (N-1 edges, guarantees connectivity).
            2. Add random edges until num_edges is reached.

        Args:
            num_nodes: Number of nodes (must be >= 2).
            num_edges: Target edge count. Must be >= N-1. None = N-1 + N//2 (moderate density).
            rng: Random number generator.

        Raises:
            ValueError: If num_nodes < 2 or num_edges < N-1 or num_edges > N*(N-1)/2.
        """
        if num_nodes < 2:
            raise ValueError(f"Need at least 2 nodes, got {num_nodes}")

        max_possible = num_nodes * (num_nodes - 1) // 2
        if num_edges is None:
            # Uniform over [N-1, max_possible] — gives equal weight to sparse
            # (trees) and dense (complete) graphs for topology diversity.
            num_edges = int(rng.integers(num_nodes - 1, max_possible + 1))

        if num_edges < num_nodes - 1:
            raise ValueError(
                f"num_edges ({num_edges}) must be >= num_nodes - 1 ({num_nodes - 1}) "
                f"to guarantee connectivity."
            )
        if num_edges > max_possible:
            raise ValueError(
                f"num_edges ({num_edges}) exceeds maximum possible ({max_possible}) "
                f"for {num_nodes} nodes."
            )

        if rng is None:
            rng = np.random.default_rng()

        # Step 1: Random spanning tree (guarantees connectivity)
        g = nx.random_labeled_tree(num_nodes, seed=int(rng.integers(1 << 31)))

        # Step 2: Add random edges until we hit the target
        all_nodes = list(range(num_nodes))
        attempts = 0
        max_attempts = num_edges * 20  # safety limit
        while g.number_of_edges() < num_edges and attempts < max_attempts:
            u, v = int(rng.integers(num_nodes)), int(rng.integers(num_nodes))
            if u != v and not g.has_edge(u, v):
                g.add_edge(u, v)
            attempts += 1

        return GraphWorld(g)

    @staticmethod
    def from_custom(edges: list[tuple[int, int]], num_nodes: int) -> GraphWorld:
        """Build from an explicit edge list."""
        g = nx.Graph()
        g.add_nodes_from(range(num_nodes))
        g.add_edges_from(edges)
        return GraphWorld(g)

    # ------------------------------------------------------------------
    # Assignment
    # ------------------------------------------------------------------

    def assign_territories(
        self,
        num_agents: int,
        rng: np.random.Generator | None = None,
    ) -> dict[int, int]:
        """Randomly assign node ownership. Every node gets an owner.

        Each node independently draws from a uniform distribution over agents.
        No nodes are left unowned.
        """
        if rng is None:
            rng = np.random.default_rng()

        self.ownership = {}
        for node in self.graph.nodes():
            self.ownership[node] = int(rng.integers(0, num_agents))

        return self.ownership

    def assign_destinations(
        self,
        num_agents: int,
        num_destinations: int,
        rng: np.random.Generator | None = None,
    ) -> dict[int, list[int]]:
        """Assign destination nodes per agent.

        Nodes CAN be shared across agents (no exclusivity).
        Each agent gets a number of destinations sampled from a normal distribution
        around ``num_destinations`` (the mean), clamped to [1, num_nodes].
        Spreads each agent's destinations using a greedy max-min-distance heuristic.
        """
        if rng is None:
            rng = np.random.default_rng()

        all_nodes = list(self.graph.nodes())
        if len(all_nodes) < 1:
            raise ValueError("Graph must have at least 1 node for destinations.")

        self.destinations = {}

        for agent_id in range(num_agents):
            # Sample destination count from normal distribution around the mean.
            # The floor is always num_destinations so validation never fails.
            std = max(1.0, num_destinations * 0.3)
            count = int(round(float(rng.normal(num_destinations, std))))
            count = max(num_destinations, min(len(all_nodes), count))

            # Each agent picks from all nodes independently
            candidates = list(all_nodes)
            rng.shuffle(candidates)

            agent_dests: list[int] = []
            # Pick first destination randomly
            agent_dests.append(candidates.pop(0))

            # Greedily pick subsequent destinations to maximize min distance
            for _ in range(count - 1):
                if not candidates:
                    break
                best_node = max(
                    candidates,
                    key=lambda n: min(
                        nx.shortest_path_length(self.graph, n, p)
                        for p in agent_dests
                    ),
                )
                agent_dests.append(best_node)
                candidates.remove(best_node)

            self.destinations[agent_id] = agent_dests

        return self.destinations

    def assign_starting_positions(
        self,
        num_agents: int,
        rng: np.random.Generator | None = None,
    ) -> dict[int, int]:
        """Assign random starting positions for agents."""
        if rng is None:
            rng = np.random.default_rng()

        all_nodes = list(self.graph.nodes())
        self.starting_positions = {}
        for aid in range(num_agents):
            self.starting_positions[aid] = int(rng.choice(all_nodes))
        return self.starting_positions

    def set_ownership(self, ownership: dict[int, int]) -> None:
        """Set ownership directly (used by GUI graph builder)."""
        self.ownership = dict(ownership)

    def set_destinations(self, destinations: dict[int, list[int]]) -> None:
        """Set destinations directly (used by GUI graph builder)."""
        self.destinations = {int(k): list(v) for k, v in destinations.items()}

    def set_starting_positions(self, positions: dict[int, int]) -> None:
        """Set starting positions directly (used by GUI graph builder)."""
        self.starting_positions = {int(k): int(v) for k, v in positions.items()}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, num_agents: int, min_destinations: int = 1,
                 trip_reward: float = 25.0, price_budget: float = 100.0) -> None:
        """Validate the graph meets all requirements.

        Raises ValueError with a descriptive message if any check fails.
        """
        errors = []

        # At least 2 nodes
        if self.num_nodes < 2:
            errors.append(f"Graph must have at least 2 nodes, got {self.num_nodes}.")

        # Must be connected
        if not nx.is_connected(self.graph):
            errors.append("Graph is not connected. All nodes must be reachable from every other node.")

        # Every node must have an owner
        for node in self.graph.nodes():
            if node not in self.ownership:
                errors.append(f"Node {node} has no owner. Every node must be owned by an agent.")
                break  # One message is enough

        if len(self.ownership) != self.num_nodes:
            errors.append(
                f"Ownership covers {len(self.ownership)} nodes but graph has {self.num_nodes}. "
                f"Every node must have an owner."
            )

        # Each agent must have >= min_destinations destinations
        for aid in range(num_agents):
            dests = self.destinations.get(aid, [])
            if len(dests) < min_destinations:
                errors.append(
                    f"Agent {aid} has {len(dests)} destination(s), needs at least {min_destinations}."
                )
            # Check destination nodes actually exist in graph
            for d in dests:
                if d not in self.graph.nodes():
                    errors.append(f"Agent {aid} destination node {d} does not exist in graph.")

        # Enough edges for connectivity
        num_edges = self.graph.number_of_edges()
        if num_edges < self.num_nodes - 1:
            errors.append(
                f"Graph has {num_edges} edges but needs at least {self.num_nodes - 1} "
                f"for connectivity."
            )

        # Positive trip reward and price budget
        if trip_reward <= 0:
            errors.append(f"Trip reward must be positive, got {trip_reward}.")
        if price_budget < 0:
            errors.append(f"Price budget must be non-negative, got {price_budget}.")

        if errors:
            raise ValueError("Graph validation failed:\n  - " + "\n  - ".join(errors))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def adjacency_list(self) -> dict[int, list[int]]:
        if self._adj_list is None:
            self._adj_list = {
                node: sorted(self.graph.neighbors(node))
                for node in self.graph.nodes()
            }
        return self._adj_list

    def get_neighbors(self, node: int) -> list[int]:
        return self.adjacency_list[node]

    @property
    def diameter(self) -> int:
        if self._diameter is None:
            self._diameter = nx.diameter(self.graph)
        return self._diameter

    def shortest_path_lengths(self, source: int) -> dict[int, int]:
        """BFS distances from source to all nodes."""
        return dict(nx.single_source_shortest_path_length(self.graph, source))

    @property
    def max_degree(self) -> int:
        return max(dict(self.graph.degree()).values())

    def owned_nodes_for(self, agent_id: int) -> list[int]:
        """Return sorted list of nodes owned by agent_id."""
        return sorted(n for n, a in self.ownership.items() if a == agent_id)

    def dest_agents_for_node(self, node_id: int) -> list[int]:
        """Return list of agent IDs that have this node as a destination."""
        return [aid for aid, dests in self.destinations.items() if node_id in dests]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "num_nodes": self.num_nodes,
            "edges": list(self.graph.edges()),
            "ownership": {str(k): v for k, v in self.ownership.items()},
            "destinations": {str(k): v for k, v in self.destinations.items()},
            "starting_positions": {str(k): v for k, v in self.starting_positions.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> GraphWorld:
        edges = [tuple(e) for e in d["edges"]]
        world = cls.from_custom(edges, d["num_nodes"])
        world.ownership = {int(k): v for k, v in d.get("ownership", {}).items()}
        # Support legacy "poles" key for backward compatibility with saved episodes
        dests = d.get("destinations") or d.get("poles", {})
        world.destinations = {int(k): v for k, v in dests.items()}
        world.starting_positions = {int(k): v for k, v in d.get("starting_positions", {}).items()}
        return world

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> GraphWorld:
        with open(path) as f:
            return cls.from_dict(json.load(f))

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    def get_spring_layout(self, seed: int = 42, scale: float = 300.0) -> dict[int, tuple[float, float]]:
        """Compute spring layout positions for visualization."""
        pos = nx.spring_layout(self.graph, seed=seed, scale=scale)
        return {node: (float(x), float(y)) for node, (x, y) in pos.items()}

    def get_circular_layout(self, scale: float = 300.0) -> dict[int, tuple[float, float]]:
        """Compute circular layout positions for visualization, scaled to pixel-friendly coords."""
        pos = nx.circular_layout(self.graph, scale=scale)
        return {node: (float(x), float(y)) for node, (x, y) in pos.items()}
