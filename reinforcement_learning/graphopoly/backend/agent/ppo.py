"""
PPO trainer for Graphopoly agents.

All agents share a single GraphopolyGNN network and a SINGLE shared Adam
optimizer. Each agent has its own PPOTrainer wrapping the shared network +
optimizer, collecting its own rollout buffer and performing its own PPO
update. Because the optimizer is shared, each agent's gradient step
correctly accumulates into the same parameter tensor and optimizer state.

Performance: the PPO update uses BATCHED GNN embedding — all transitions
in a batch are stacked into one big disconnected graph, run through the
GNN in a single forward pass, then split back for per-transition head
computations. This is ~50x faster than per-transition forward passes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from backend.config import TrainConfig
from backend.agent.gnn_network import GraphopolyGNN


@dataclass
class Transition:
    """Single timestep of experience (GNN-compatible)."""
    node_features: torch.Tensor   # [N, F]
    current_pos: int
    valid_neighbors: list[int]
    action_move: int
    action_prices: dict[int, float]
    log_prob: float
    value: float
    reward: float = 0.0
    done: bool = False


class RolloutBuffer:
    """Stores one agent's trajectory for a rollout period."""

    def __init__(self) -> None:
        self.transitions: list[Transition] = []
        self.returns: list[float] = []
        self.advantages: list[float] = []

    def add(self, transition: Transition) -> None:
        self.transitions.append(transition)

    def compute_gae(
        self,
        last_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Compute Generalized Advantage Estimation."""
        rewards = [t.reward for t in self.transitions]
        values = [t.value for t in self.transitions]
        dones = [t.done for t in self.transitions]

        n = len(rewards)
        self.advantages = [0.0] * n
        self.returns = [0.0] * n

        gae = 0.0
        next_value = last_value

        for t in reversed(range(n)):
            mask = 0.0 if dones[t] else 1.0
            delta = rewards[t] + gamma * next_value * mask - values[t]
            gae = delta + gamma * gae_lambda * mask * gae
            self.advantages[t] = gae
            self.returns[t] = gae + values[t]
            next_value = values[t]

    def get_batches(self, batch_size: int):
        """Yield index batches (shuffled)."""
        n = len(self.transitions)
        indices = torch.randperm(n).tolist()
        for start in range(0, n, batch_size):
            yield indices[start : start + batch_size]

    def clear(self) -> None:
        self.transitions.clear()
        self.returns.clear()
        self.advantages.clear()


class PPOTrainer:
    """PPO algorithm for one agent, using a shared GNN + shared optimizer.

    Uses batched GNN embedding during updates for dramatically faster training.
    """

    def __init__(
        self,
        network: GraphopolyGNN,
        optimizer: torch.optim.Optimizer,
        config: TrainConfig,
        edge_index: torch.Tensor,
    ):
        self.network = network
        self.optimizer = optimizer
        self.config = config
        self.edge_index = edge_index
        self.buffer = RolloutBuffer()

    # ------------------------------------------------------------------
    # Collection-time methods
    # ------------------------------------------------------------------

    def select_action(
        self,
        node_features: torch.Tensor,
        current_pos: int,
        valid_neighbors: list[int],
        owned_nodes: list[int],
        deterministic: bool = False,
        price_budget: float = 100.0,
    ) -> tuple[dict, float, float]:
        """Sample an action and store transition in buffer."""
        with torch.no_grad():
            action, log_prob, _entropy, value = self.network.get_action_and_value(
                node_features,
                self.edge_index,
                current_pos,
                valid_neighbors,
                owned_nodes,
                deterministic=deterministic,
                price_budget=price_budget,
            )

        self.buffer.add(Transition(
            node_features=node_features.detach(),
            current_pos=current_pos,
            valid_neighbors=list(valid_neighbors),
            action_move=action["move"],
            action_prices=dict(action["price_changes"]),
            log_prob=log_prob.item(),
            value=value.item(),
        ))

        return action, log_prob.item(), value.item()

    def store_reward(self, reward: float, done: bool) -> None:
        """Fill in reward and done flag for the most recent transition."""
        if self.buffer.transitions:
            self.buffer.transitions[-1].reward = reward
            self.buffer.transitions[-1].done = done

    def get_value(
        self,
        node_features: torch.Tensor,
        current_pos: int,
        valid_neighbors: list[int],
    ) -> float:
        """Get bootstrap value estimate (no action sampling)."""
        with torch.no_grad():
            _, _, value = self.network.forward(
                node_features,
                self.edge_index,
                current_pos,
                valid_neighbors,
                [],
            )
        return value.item()

    # ------------------------------------------------------------------
    # Batched embedding helper
    # ------------------------------------------------------------------

    def _build_batched_graph(
        self, batch_indices: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Stack transitions into a single batched graph for one GNN pass.

        All transitions share the same graph topology (same edge_index).
        We create B copies of the edge_index, offset by B*N each.

        Returns:
            batched_features:   [B*N, F]
            batched_edge_index: [2, B*E]
            N:                  nodes per graph
        """
        B = len(batch_indices)
        t0 = self.buffer.transitions[batch_indices[0]]
        N = t0.node_features.size(0)
        E = self.edge_index.size(1)
        device = t0.node_features.device

        # Stack node features
        feat_list = [self.buffer.transitions[idx].node_features for idx in batch_indices]
        batched_features = torch.cat(feat_list, dim=0)  # [B*N, F]

        # Build batched edge_index: offset each copy by graph_idx * N
        offsets = torch.arange(B, device=device).unsqueeze(1) * N  # [B, 1]
        base = self.edge_index.unsqueeze(0).expand(B, -1, -1)      # [B, 2, E]
        batched_edge_index = (base + offsets.unsqueeze(1)).reshape(2, B * E)  # [2, B*E]

        return batched_features, batched_edge_index, N

    # ------------------------------------------------------------------
    # PPO update (batched)
    # ------------------------------------------------------------------

    def update(self, last_value: float, owned_nodes: list[int], price_budget: float = 100.0) -> dict[str, float]:
        """Run PPO update with batched GNN embedding.

        The GNN embedding (expensive: 2-5 GATv2 layers) runs ONCE per batch
        on a stacked graph. Only the cheap head computations run per-transition.
        """
        cfg = self.config

        self.buffer.compute_gae(last_value, cfg.gamma, cfg.gae_lambda)

        # Normalize advantages
        advs = torch.tensor(self.buffer.advantages, dtype=torch.float32)
        if len(advs) > 1:
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        returns = torch.tensor(self.buffer.returns, dtype=torch.float32)
        old_log_probs = torch.tensor(
            [t.log_prob for t in self.buffer.transitions], dtype=torch.float32
        )

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        num_updates = 0

        for _ in range(cfg.ppo_epochs):
            for batch_indices in self.buffer.get_batches(cfg.batch_size):
                # ── Batched GNN embedding (one forward pass) ─────────
                batched_features, batched_edge_index, N = self._build_batched_graph(
                    batch_indices
                )
                all_embeds = self.network.embed_batched(
                    batched_features, batched_edge_index, N
                )  # [B*N, H]

                # Split into per-transition embeddings
                per_graph_embeds = all_embeds.split(N)  # tuple of [N, H]

                # ── Per-transition head computations (cheap) ─────────
                batch_policy_loss = batched_features.new_zeros(())
                batch_value_loss = batched_features.new_zeros(())
                batch_entropy = batched_features.new_zeros(())

                for bi, idx in enumerate(batch_indices):
                    t = self.buffer.transitions[idx]
                    h = per_graph_embeds[bi]  # [N, H]

                    new_log_prob, entropy, new_value = self.network.evaluate_actions_from_embed(
                        h,
                        t.current_pos,
                        t.valid_neighbors,
                        owned_nodes,
                        t.action_move,
                        t.action_prices,
                        price_budget=price_budget,
                    )

                    # Clipped surrogate
                    old_lp = old_log_probs[idx]
                    ratio = torch.exp(new_log_prob - old_lp)
                    adv = advs[idx]
                    surr1 = ratio * adv
                    surr2 = torch.clamp(
                        ratio, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon
                    ) * adv
                    policy_loss = -torch.min(surr1, surr2)

                    value_loss = (new_value - returns[idx]) ** 2

                    batch_policy_loss = batch_policy_loss + policy_loss
                    batch_value_loss = batch_value_loss + value_loss
                    batch_entropy = batch_entropy + entropy

                n_batch = len(batch_indices)
                loss = (
                    batch_policy_loss / n_batch
                    + cfg.value_coef * batch_value_loss / n_batch
                    - cfg.entropy_coef * batch_entropy / n_batch
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), cfg.max_grad_norm
                )
                self.optimizer.step()

                total_policy_loss += (batch_policy_loss / n_batch).item()
                total_value_loss += (batch_value_loss / n_batch).item()
                total_entropy_loss += (batch_entropy / n_batch).item()
                num_updates += 1

        self.buffer.clear()

        denom = max(num_updates, 1)
        return {
            "policy_loss": total_policy_loss / denom,
            "value_loss": total_value_loss / denom,
            "entropy": total_entropy_loss / denom,
        }
