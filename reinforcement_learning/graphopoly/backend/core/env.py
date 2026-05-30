"""
Graphopoly multi-agent environment.

Manages the simulation loop: movement, tax resolution, destination completion,
price changes. Enforces conservation of reward (taxes are pure transfers;
trip rewards are the only injection).
"""

from __future__ import annotations

import numpy as np
import torch

from backend.config import GraphopolyConfig
from backend.core.graph_world import GraphWorld
from backend.core.agent_state import AgentState


class GraphopolyEnv:
    """Multi-agent environment for Graphopoly.

    Follows a reset/step pattern. Not a gym.Env subclass because the
    multi-agent interface is different (list of actions/rewards).
    """

    def __init__(self, config: GraphopolyConfig, world: GraphWorld):
        self.config = config
        self.world = world
        self.agents: list[AgentState] = []
        self.step_count: int = 0
        self._rng = np.random.default_rng(config.seed)

        num_agents = config.agent.num_agents
        # Pre-compute per-agent info
        self._agent_owned: dict[int, list[int]] = {}
        for aid in range(num_agents):
            self._agent_owned[aid] = world.owned_nodes_for(aid)

        # Pre-compute distance tables for observation features
        self._dist_tables: dict[int, dict[int, int]] = {}
        for node in world.graph.nodes():
            self._dist_tables[node] = world.shortest_path_lengths(node)

        # Observation dimension (fixed for this graph + agent config)
        self._obs_dim: int | None = None

        # Cached bidirectional edge_index tensor [2, E] — constant for this graph
        self._edge_index: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def observation_dim(self) -> int:
        """Compute the flattened observation size for one agent."""
        if self._obs_dim is not None:
            return self._obs_dim

        N = self.world.num_nodes
        A = self.config.agent.num_agents

        # Per-node features:
        #   is_owned_by_me (1) + is_owned_by_other (1) + price (1)
        #   + occupancy per agent (A)
        #   + is_any_of_my_dests (1) + is_last_visited_dest (1)
        #   + dist_to_nearest_valid_dest (1) + dist_to_last_dest (1)
        features_per_node = 7 + A

        # Agent-specific global features:
        #   position_onehot (N) + cumulative_reward (1) + trips (1)
        agent_features = N + 2

        self._obs_dim = N * features_per_node + agent_features
        return self._obs_dim

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> list[torch.Tensor]:
        """Reset all agents for a new episode. Returns observations per agent."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.step_count = 0
        num_agents = self.config.agent.num_agents

        self.agents = []
        for aid in range(num_agents):
            destinations = self.world.destinations[aid]
            owned = self._agent_owned[aid]

            # Starting position: custom > configured > first destination
            if self.world.starting_positions and aid in self.world.starting_positions:
                start_pos = self.world.starting_positions[aid]
            elif (self.config.graph.custom_starting_positions
                  and aid in self.config.graph.custom_starting_positions):
                start_pos = self.config.graph.custom_starting_positions[aid]
            else:
                start_pos = destinations[0]

            agent = AgentState(agent_id=aid)
            agent.reset(
                start_position=start_pos,
                destinations=destinations,
                owned_nodes=owned,
                price_budget=self.config.agent.price_budget,
            )
            self.agents.append(agent)

        return [self._build_observation(aid) for aid in range(num_agents)]

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self, actions: list[dict]
    ) -> tuple[list[torch.Tensor], list[float], bool, dict]:
        """Execute one timestep.

        Args:
            actions: list of dicts per agent:
                {
                    'move': int  -- node to move to (-1 or current node = stay),
                    'price_changes': dict[int, int]  -- {node_id: delta} where delta in {-1, 0, +1}
                }

        Returns:
            observations: list of Tensor per agent
            rewards: list of float per agent
            done: bool
            info: dict with step details
        """
        num_agents = self.config.agent.num_agents
        rewards = [0.0] * num_agents
        info = {
            "taxes": {},              # {payer_id: {receiver_id: amount}}
            "dest_completions": [],   # [(agent_id, dest_node)]
            "positions": [],
        }

        # 1. Execute moves simultaneously — track previous positions for standing detection
        prev_positions = [a.position for a in self.agents]
        for aid, action in enumerate(actions):
            agent = self.agents[aid]
            target = action["move"]
            neighbors = self.world.get_neighbors(agent.position)

            if target == agent.position or target == -1:
                # Stay in place
                pass
            elif target in neighbors:
                agent.position = target
            # else: invalid move, agent stays (shouldn't happen with proper masking)

        # Update steps_on_current_node tracking
        for aid in range(num_agents):
            agent = self.agents[aid]
            if agent.position == prev_positions[aid]:
                agent.steps_on_current_node += 1
            else:
                agent.steps_on_current_node = 1  # just arrived

        # 2. Track node visits
        for aid in range(num_agents):
            agent = self.agents[aid]
            agent.node_visit_counts[agent.position] = (
                agent.node_visit_counts.get(agent.position, 0) + 1
            )

        # 3. Resolve taxes (pure transfers) — agents pay on EVERY step they're on a node
        #    Including paying themselves (net 0 but tracked in stats)
        for aid in range(num_agents):
            agent = self.agents[aid]
            node = agent.position

            if node in self.world.ownership:
                owner_id = self.world.ownership[node]
                owner = self.agents[owner_id]
                tax = owner.prices.get(node, 0)

                if owner_id == aid:
                    # Self-tax: net 0 but track in stats for accuracy
                    agent.tax_paid += tax
                    agent.tax_revenue += tax
                    # rewards[aid] unchanged (cancel out)
                else:
                    # Transfer: payer loses, owner gains
                    rewards[aid] -= tax
                    rewards[owner_id] += tax
                    agent.tax_paid += tax
                    owner.tax_revenue += tax

                if aid not in info["taxes"]:
                    info["taxes"][aid] = {}
                info["taxes"][aid][owner_id] = tax

        # 4. Check destination arrivals
        # Any destination yields a reward EXCEPT the one visited immediately before.
        for aid in range(num_agents):
            agent = self.agents[aid]
            pos = agent.position

            # Only destinations count
            if pos not in agent.destinations:
                continue

            # No reward if standing still on this node for 2+ consecutive steps
            moved_this_step = (pos != prev_positions[aid])
            if not moved_this_step and agent.steps_on_current_node > 1:
                continue

            # No reward if this is the destination just visited (must move to a different one)
            if agent.last_visited_dest == pos:
                continue

            rewards[aid] += self.config.agent.trip_reward
            agent.dest_revenue += self.config.agent.trip_reward
            info["dest_completions"].append((aid, pos))
            agent.complete_destination(pos)

        # 5. Apply price changes (absolute prices from softmax budget distribution)
        for aid, action in enumerate(actions):
            agent = self.agents[aid]
            price_changes = action.get("price_changes", {})
            for node_id, price in price_changes.items():
                node_id = int(node_id)
                if node_id in agent.prices:
                    agent.prices[node_id] = max(0.0, float(price))

        # 6. Update cumulative rewards
        for aid in range(num_agents):
            self.agents[aid].cumulative_reward += rewards[aid]

        # 7. Build new observations
        self.step_count += 1
        done = self.step_count >= self.config.train.steps_per_episode

        info["positions"] = [a.position for a in self.agents]
        observations = [self._build_observation(aid) for aid in range(num_agents)]

        # Debug: conservation check — taxes must net to zero
        dest_injection = len(info["dest_completions"]) * self.config.agent.trip_reward
        expected_total = dest_injection  # taxes net to zero, trip rewards are injections
        actual_total = sum(rewards)
        assert abs(actual_total - expected_total) < 1e-6, (
            f"Reward conservation violated: expected {expected_total}, got {actual_total}"
        )

        return observations, rewards, done, info

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _build_observation(self, agent_id: int) -> torch.Tensor:
        """Build flattened observation vector for one agent.

        Layout:
            For each node i in [0, N):
                is_owned_by_me              (1)
                is_owned_by_other           (1)
                price                       (1, normalized)
                occupancy_per_agent         (A, binary)
                is_any_of_my_dests          (1)
                is_last_visited_dest        (1)   # 1 if this was most recently visited dest
                dist_to_nearest_valid_dest  (1, normalized by diameter)
                dist_to_last_dest           (1, normalized by diameter)

            Agent global features:
                position_onehot             (N)
                cumulative_reward           (1, / 100)
                trips_completed             (1, / 10)
        """
        agent = self.agents[agent_id]
        N = self.world.num_nodes
        A = self.config.agent.num_agents
        diameter = max(self.world.diameter, 1)
        price_budget = max(self.config.agent.price_budget, 1.0)

        # Valid destinations = all agent dests except the last visited one
        valid_dests = [d for d in agent.destinations if d != agent.last_visited_dest]

        # Distance table from last visited destination (or position if none)
        last_ref = agent.last_visited_dest if agent.last_visited_dest is not None else agent.position
        last_dists = self._dist_tables.get(last_ref, {})

        # Nearest valid destination distance from each node
        # Precompute: for each node, min distance to any valid dest
        def nearest_valid_dist(node: int) -> float:
            if not valid_dests:
                return 0.0
            return min(
                self._dist_tables.get(d, {}).get(node, diameter)
                for d in valid_dests
            ) / diameter

        # Build per-node features
        node_feats = []
        for node in range(N):
            # Ownership — every node has an owner
            owner = self.world.ownership.get(node, -1)
            is_mine = 1.0 if owner == agent_id else 0.0
            is_other = 1.0 if (owner >= 0 and owner != agent_id) else 0.0

            # Price (from whoever owns it)
            price = 0.0
            if owner >= 0:
                price = self.agents[owner].prices.get(node, 0.0) / price_budget

            # Occupancy: which agents are on this node
            occupancy = [
                1.0 if self.agents[a].position == node else 0.0
                for a in range(A)
            ]

            # Destination flags
            is_dest = 1.0 if node in agent.destinations else 0.0
            is_last = 1.0 if node == agent.last_visited_dest else 0.0

            # Distances
            dist_nearest_valid = nearest_valid_dist(node)
            dist_last = last_dists.get(node, diameter) / diameter

            node_feats.extend([is_mine, is_other, price] + occupancy +
                              [is_dest, is_last, dist_nearest_valid, dist_last])

        # Agent global features
        pos_onehot = [0.0] * N
        pos_onehot[agent.position] = 1.0

        agent_feats = (
            pos_onehot
            + [agent.cumulative_reward / 100.0]
            + [agent.trips_completed / 10.0]
        )

        obs = node_feats + agent_feats
        return torch.tensor(obs, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Action helpers
    # ------------------------------------------------------------------

    def get_move_mask(self, agent_id: int) -> torch.Tensor:
        """Return a boolean mask of size [num_nodes] where True = valid move.

        Valid moves: neighbors of current position + current position (stay).
        """
        agent = self.agents[agent_id]
        mask = torch.zeros(self.world.num_nodes, dtype=torch.bool)
        mask[agent.position] = True  # can stay
        for neighbor in self.world.get_neighbors(agent.position):
            mask[neighbor] = True
        return mask

    def get_owned_nodes(self, agent_id: int) -> list[int]:
        """Return list of nodes owned by this agent."""
        return self._agent_owned[agent_id]

    # ------------------------------------------------------------------
    # GNN helpers
    # ------------------------------------------------------------------

    def get_edge_index(self) -> torch.Tensor:
        """Return bidirectional edge_index [2, E] for the graph (cached).

        Result is constant for the lifetime of this env instance — compute once,
        reuse every step.
        """
        if self._edge_index is None:
            edges = []
            for u, v in self.world.graph.edges():
                edges.append([u, v])
                edges.append([v, u])
            if edges:
                self._edge_index = torch.tensor(
                    edges, dtype=torch.long
                ).t().contiguous()
            else:
                self._edge_index = torch.zeros((2, 0), dtype=torch.long)
        return self._edge_index

    def get_valid_neighbors(self, agent_id: int) -> list[int]:
        """Return adjacent nodes for the given agent's current position.

        These are the movement candidates (excluding 'stay', which is handled
        separately in the GNN by always including current_pos as candidate 0).
        """
        return list(self.world.get_neighbors(self.agents[agent_id].position))

    def _build_shared_node_data(self) -> dict:
        """Compute per-node values that are IDENTICAL for every agent.

        Call once per step; pass the result to get_node_features() for each
        agent to avoid redundant computation (saves ~A× work).

        Returns a dict with:
            agents_at: list[int]   — total agents at each node
            prices:    list[float] — normalised price at each node
            max_price: int
            norm_A:    float       — max(A-1, 1) for opponent normalisation
        """
        N = self.world.num_nodes
        A = self.config.agent.num_agents
        price_budget = max(self.config.agent.price_budget, 1.0)

        agents_at = [0] * N
        for agent in self.agents:
            agents_at[agent.position] += 1

        prices: list[float] = []
        for j in range(N):
            owner = self.world.ownership.get(j, -1)
            price_raw = self.agents[owner].prices.get(j, 0.0) if owner >= 0 else 0.0
            prices.append(price_raw / price_budget)

        return {
            "agents_at": agents_at,
            "prices": prices,
            "price_budget": price_budget,
            "norm_A": float(max(A - 1, 1)),
        }

    def get_node_features(
        self,
        agent_id: int,
        shared: dict | None = None,
    ) -> torch.Tensor:
        """Build per-agent-relative node feature matrix [N, 13].

        Args:
            agent_id: which agent's perspective to build features for.
            shared:   pre-computed shared data from _build_shared_node_data().
                      If None, it is recomputed (convenient for single-agent use
                      or testing; callers should pass it when iterating over all
                      agents in the same step).

        Features (all normalised to [0, 1]):
            0  am_I_here                              — 1 if agent is at node j
            1  is_my_owned_node                       — 1 if agent owns j
            2  price / max_price
            3  is_my_destination                      — j in agent's dest list
            4  is_my_last_visited_dest                — j was most recent reward node
            5  num_opponents_here / max(A-1, 1)       — congestion (opponents only)
            6  num_opponents_targeting_j / max(A-1,1) — how contested j is as a dest
            7  dist_from_j_to_nearest_valid_dest / D  — routing pull toward goal
            8  dist_from_j_to_last_visited_dest / D   — distance from last reward
            9  trip_reward / 100.0                    — global economic scale
            10 price_budget / 1000.0                   — global budget scale context
            11 degree / max_degree                    — structural bottleneck signal
            12 1 / num_nodes                          — graph scale awareness
        """
        if shared is None:
            shared = self._build_shared_node_data()

        agent = self.agents[agent_id]
        N = self.world.num_nodes
        diameter = max(self.world.diameter, 1)
        price_budget: float = shared["price_budget"]
        norm_A: float = shared["norm_A"]
        agents_at: list[int] = shared["agents_at"]
        prices: list[float] = shared["prices"]

        # Valid dests: all agent destinations except the last-visited one
        valid_dests = [d for d in agent.destinations if d != agent.last_visited_dest]

        # Distance table FROM last-visited dest (or current position if none)
        last_ref = (
            agent.last_visited_dest
            if agent.last_visited_dest is not None
            else agent.position
        )
        last_dists = self._dist_tables.get(last_ref, {})

        # Opponents targeting each node as a valid destination
        opp_targeting = [0] * N
        for oid, opp in enumerate(self.agents):
            if oid == agent_id:
                continue
            for d in opp.destinations:
                if d != opp.last_visited_dest:
                    opp_targeting[d] += 1

        trip_norm = self.config.agent.trip_reward / 100.0
        budget_norm = price_budget / 1000.0

        # Structural features
        max_degree = max(self.world.graph.degree(j) for j in range(N))
        max_degree = max(max_degree, 1)
        inv_num_nodes = 1.0 / N

        rows: list[list[float]] = []
        for j in range(N):
            owner = self.world.ownership.get(j, -1)
            price_norm = prices[j]

            # Opponents at this node (exclude self)
            total_at_j = agents_at[j]
            is_me_here = 1.0 if j == agent.position else 0.0
            opp_here = max(total_at_j - (1 if j == agent.position else 0), 0)
            opp_here_norm = min(opp_here, norm_A) / norm_A

            opp_tgt_norm = min(opp_targeting[j], norm_A) / norm_A

            # Distance from j to nearest valid destination
            if valid_dests:
                dist_nearest = (
                    min(self._dist_tables[d].get(j, diameter) for d in valid_dests)
                    / diameter
                )
            else:
                dist_nearest = 0.0

            dist_last = last_dists.get(j, diameter) / diameter

            rows.append([
                is_me_here,                                              # 0
                1.0 if owner == agent_id else 0.0,                      # 1
                price_norm,                                              # 2
                1.0 if j in agent.destinations else 0.0,               # 3
                1.0 if j == agent.last_visited_dest else 0.0,          # 4
                opp_here_norm,                                           # 5
                opp_tgt_norm,                                            # 6
                dist_nearest,                                            # 7
                dist_last,                                               # 8
                trip_norm,                                               # 9
                budget_norm,                                             # 10
                self.world.graph.degree(j) / max_degree,                # 11 structural
                inv_num_nodes,                                           # 12 graph scale
            ])

        return torch.tensor(rows, dtype=torch.float32)  # [N, 13]

    # ------------------------------------------------------------------
    # State snapshot (for logging / GUI)
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """Return full environment state as a JSON-serializable dict."""
        return {
            "step": self.step_count,
            "agents": [a.to_dict() for a in self.agents],
            "prices": {
                str(node): round(self.agents[owner].prices.get(node, 0.0), 2)
                for node, owner in self.world.ownership.items()
            },
            "positions": [a.position for a in self.agents],
        }
