"""
GATv2 policy + value network for Graphopoly.

One network instance is SHARED across all agents (same weights, per-agent
forward passes with per-agent-relative node features).

Architecture:
  [N × F] ──► 2-7× GATv2Conv (dynamic depth) ──► node embeddings [N × H]
  Movement:   vectorised MLP([embed_curr ‖ embed_cand]) → score per candidate
  Pricing:    Linear(embed_owned) → raw scores → softmax → × budget = prices
  Value:      Linear([global_mean ‖ pos_embed]) → scalar  (light centralized critic)

Key design choices:
  - concat=False on GATv2: multi-head output is *averaged* → embedding always H dims
  - Dynamic depth: 2 layers for N≤4, 3 for N≤10, 4 for N≤15, 5 for N≤25, 6 for N≤35, 7 for N≤50
  - Residual connections on layers 1+ (layer 0 changes dims: F→H, so no residual)
  - Batched embedding for PPO updates (stack transitions → one GNN pass)
  - Movement head is fully vectorised (no Python loop over candidates)
  - Pricing head outputs softmax budget allocation (no per-node loop)
  - Value head uses both global mean-pool AND the agent's current-position embedding
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Dirichlet

from torch_geometric.nn import GATv2Conv

from backend.config import NetworkConfig


class GraphopolyGNN(nn.Module):
    """Shared GATv2 policy + value network for all agents.

    Universal: works on any graph size N. Dynamic depth adapts
    the number of message-passing rounds to graph size.
    """

    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        H = config.hidden_dim
        F_in = config.node_feature_dim  # 13

        # ── GATv2 backbone (dynamic depth: use 2-7 of these) ─────────────
        self.gnn_layers = nn.ModuleList()
        in_ch = F_in
        for _ in range(config.max_gnn_layers):
            self.gnn_layers.append(
                GATv2Conv(
                    in_ch, H,
                    heads=config.gat_heads,
                    concat=False,
                    dropout=config.dropout,
                    add_self_loops=True,
                )
            )
            in_ch = H

        # ── Movement head ────────────────────────────────────────────────────
        self.move_head = nn.Sequential(
            nn.Linear(H * 2, config.move_mlp_hidden),
            nn.ReLU(),
            nn.Linear(config.move_mlp_hidden, 1),
        )

        # ── Pricing head (softmax budget allocation) ─────────────────────────
        # Outputs a single score per owned node → softmax → × budget = prices
        self.price_head = nn.Linear(H, 1)

        # ── Value head (light centralized critic) ────────────────────────────
        self.value_head = nn.Sequential(
            nn.Linear(H * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Orthogonal init (standard for PPO)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for head in [self.move_head, self.price_head, self.value_head]:
            last_linear = [m for m in head.modules() if isinstance(m, nn.Linear)][-1]
            nn.init.orthogonal_(last_linear.weight, gain=0.01)

    # ------------------------------------------------------------------
    # Dynamic depth
    # ------------------------------------------------------------------

    @staticmethod
    def _get_depth(num_nodes: int) -> int:
        """Return the number of GATv2 layers to use based on graph size."""
        if num_nodes <= 4:
            return 2
        if num_nodes <= 10:
            return 3
        if num_nodes <= 15:
            return 4
        if num_nodes <= 25:
            return 5
        if num_nodes <= 35:
            return 6
        return 7

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def embed(
        self,
        node_features: torch.Tensor,   # [N, F]
        edge_index: torch.Tensor,       # [2, E]
    ) -> torch.Tensor:
        """Run GATv2 message passing with dynamic depth. Returns [N, H]."""
        depth = self._get_depth(node_features.size(0))
        h = node_features
        for i in range(depth):
            h_new = F.elu(self.gnn_layers[i](h, edge_index))
            if i > 0:
                h = h + h_new  # residual
            else:
                h = h_new       # layer 0: dim change F→H
        return h

    def embed_batched(
        self,
        batched_features: torch.Tensor,  # [B*N, F]
        batched_edge_index: torch.Tensor, # [2, B*E]
        num_nodes: int,                    # N per graph (all same)
    ) -> torch.Tensor:
        """Batched GNN embedding: stack B graphs, one forward pass.

        All graphs in the batch must have the same N (same episode, same graph).
        Returns [B*N, H].
        """
        depth = self._get_depth(num_nodes)
        h = batched_features
        for i in range(depth):
            h_new = F.elu(self.gnn_layers[i](h, batched_edge_index))
            if i > 0:
                h = h + h_new
            else:
                h = h_new
        return h

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        current_pos: int,
        valid_neighbors: list[int],
        owned_nodes: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Single agent forward pass."""
        h = self.embed(node_features, edge_index)

        # Movement (vectorised)
        candidates = [current_pos] + list(valid_neighbors)
        C = len(candidates)
        cand_idx = torch.tensor(candidates, dtype=torch.long, device=h.device)
        cand_embeds = h[cand_idx]
        curr_expand = h[current_pos].unsqueeze(0).expand(C, -1)
        pairs = torch.cat([curr_expand, cand_embeds], dim=1)
        move_logits = self.move_head(pairs).squeeze(-1)

        # Pricing — raw scores for softmax budget allocation
        price_scores: torch.Tensor | None = None
        if owned_nodes:
            owned_idx = torch.tensor(owned_nodes, dtype=torch.long, device=h.device)
            price_scores = self.price_head(h[owned_idx]).squeeze(-1)  # [K]

        # Value
        global_mean = h.mean(dim=0)
        pos_embed = h[current_pos]
        value = self.value_head(
            torch.cat([global_mean, pos_embed])
        ).squeeze(-1)

        return move_logits, price_scores, value

    # ------------------------------------------------------------------
    # Heads-only forward (used with pre-computed embeddings from batch)
    # ------------------------------------------------------------------

    def heads_from_embed(
        self,
        h: torch.Tensor,               # [N, H] — pre-computed embeddings for ONE graph
        current_pos: int,
        valid_neighbors: list[int],
        owned_nodes: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Run movement/pricing/value heads on pre-computed embeddings."""
        candidates = [current_pos] + list(valid_neighbors)
        C = len(candidates)
        cand_idx = torch.tensor(candidates, dtype=torch.long, device=h.device)
        cand_embeds = h[cand_idx]
        curr_expand = h[current_pos].unsqueeze(0).expand(C, -1)
        pairs = torch.cat([curr_expand, cand_embeds], dim=1)
        move_logits = self.move_head(pairs).squeeze(-1)

        price_scores: torch.Tensor | None = None
        if owned_nodes:
            owned_idx = torch.tensor(owned_nodes, dtype=torch.long, device=h.device)
            price_scores = self.price_head(h[owned_idx]).squeeze(-1)  # [K]

        global_mean = h.mean(dim=0)
        pos_embed = h[current_pos]
        value = self.value_head(
            torch.cat([global_mean, pos_embed])
        ).squeeze(-1)

        return move_logits, price_scores, value

    # ------------------------------------------------------------------
    # Pricing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_prices(
        price_scores: torch.Tensor,
        budget: float,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a budget allocation from price scores.

        Args:
            price_scores: [K] raw scores from price head
            budget: total budget to distribute
            deterministic: if True, use softmax directly (no sampling noise)

        Returns:
            prices: [K] prices that sum to budget
            log_prob: scalar log probability
            entropy: scalar entropy
        """
        K = price_scores.size(0)
        if K == 1:
            # Only one node — all budget goes to it, no choice
            prices = price_scores.new_tensor([budget])
            log_prob = price_scores.new_zeros(())
            entropy = price_scores.new_zeros(())
            return prices, log_prob, entropy

        if deterministic:
            weights = F.softmax(price_scores, dim=0)
            prices = weights * budget
            log_prob = price_scores.new_zeros(())
            entropy = price_scores.new_zeros(())
            return prices, log_prob, entropy

        # Use Dirichlet distribution for exploration
        # Convert scores to positive concentration parameters
        concentration = F.softplus(price_scores) + 0.5  # min 0.5 to avoid degenerate distributions
        # MPS doesn't support _sample_dirichlet — sample on CPU and move back
        concentration_cpu = concentration.cpu()
        dist = Dirichlet(concentration_cpu)
        weights_cpu = dist.rsample()  # [K], sums to 1
        weights = weights_cpu.to(price_scores.device)
        prices = weights * budget

        log_prob = dist.log_prob(weights_cpu)
        entropy = dist.entropy()

        return prices, log_prob, entropy

    @staticmethod
    def _evaluate_prices(
        price_scores: torch.Tensor,
        action_prices_tensor: torch.Tensor,
        budget: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Re-evaluate log_prob and entropy for stored price actions.

        Args:
            price_scores: [K] raw scores from price head
            action_prices_tensor: [K] the prices that were actually taken
            budget: total budget

        Returns:
            log_prob: scalar
            entropy: scalar
        """
        K = price_scores.size(0)
        if K == 1:
            return price_scores.new_zeros(()), price_scores.new_zeros(())

        concentration = F.softplus(price_scores) + 0.5
        dist = Dirichlet(concentration)
        # Reconstruct the weights from the prices
        weights = action_prices_tensor / max(budget, 1e-8)
        # Clamp to valid simplex range for numerical stability
        weights = weights.clamp(min=1e-6)
        weights = weights / weights.sum()

        log_prob = dist.log_prob(weights)
        entropy = dist.entropy()
        return log_prob, entropy

    # ------------------------------------------------------------------
    # Action sampling (collection time — no gradients)
    # ------------------------------------------------------------------

    def get_action_and_value(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        current_pos: int,
        valid_neighbors: list[int],
        owned_nodes: list[int],
        deterministic: bool = False,
        price_budget: float = 100.0,
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions from the current policy."""
        candidates = [current_pos] + list(valid_neighbors)
        move_logits, price_scores, value = self.forward(
            node_features, edge_index, current_pos, valid_neighbors, owned_nodes
        )

        move_dist = Categorical(logits=move_logits)
        move_idx = move_logits.argmax() if deterministic else move_dist.sample()
        move_action_node = candidates[move_idx.item()]
        move_lp = move_dist.log_prob(move_idx)
        move_ent = move_dist.entropy()

        price_changes: dict[int, float] = {}
        price_lp = move_logits.new_zeros(())
        price_ent = move_logits.new_zeros(())

        if price_scores is not None and len(owned_nodes) > 0:
            prices, price_lp, price_ent = self._sample_prices(
                price_scores, price_budget, deterministic
            )
            for i, nid in enumerate(owned_nodes):
                price_changes[nid] = prices[i].item()

        dev = node_features.device
        return (
            {"move": move_action_node, "price_changes": price_changes},
            move_lp + price_lp,
            move_ent + price_ent,
            value,
        )

    # ------------------------------------------------------------------
    # Action re-evaluation (PPO update — gradients flow)
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        current_pos: int,
        valid_neighbors: list[int],
        owned_nodes: list[int],
        action_move: int,
        action_prices: dict[int, float],
        price_budget: float = 100.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate stored actions under current policy (single transition)."""
        candidates = [current_pos] + list(valid_neighbors)
        move_logits, price_scores, value = self.forward(
            node_features, edge_index, current_pos, valid_neighbors, owned_nodes
        )

        move_dist = Categorical(logits=move_logits)
        move_idx = torch.tensor(
            candidates.index(action_move), dtype=torch.long, device=node_features.device
        )
        move_lp = move_dist.log_prob(move_idx)
        move_ent = move_dist.entropy()

        price_lp = node_features.new_zeros(())
        price_ent = node_features.new_zeros(())
        if price_scores is not None and len(owned_nodes) > 0:
            action_prices_tensor = torch.tensor(
                [action_prices.get(nid, 0.0) for nid in owned_nodes],
                dtype=torch.float32,
                device=node_features.device,
            )
            price_lp, price_ent = self._evaluate_prices(
                price_scores, action_prices_tensor, price_budget
            )

        return move_lp + price_lp, move_ent + price_ent, value

    def evaluate_actions_from_embed(
        self,
        h: torch.Tensor,              # [N, H] pre-computed embeddings
        current_pos: int,
        valid_neighbors: list[int],
        owned_nodes: list[int],
        action_move: int,
        action_prices: dict[int, float],
        price_budget: float = 100.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate using pre-computed embeddings (used in batched PPO update)."""
        move_logits, price_scores, value = self.heads_from_embed(
            h, current_pos, valid_neighbors, owned_nodes
        )

        candidates = [current_pos] + list(valid_neighbors)
        move_dist = Categorical(logits=move_logits)
        move_idx = torch.tensor(
            candidates.index(action_move), dtype=torch.long, device=h.device
        )
        move_lp = move_dist.log_prob(move_idx)
        move_ent = move_dist.entropy()

        price_lp = h.new_zeros(())
        price_ent = h.new_zeros(())
        if price_scores is not None and len(owned_nodes) > 0:
            action_prices_tensor = torch.tensor(
                [action_prices.get(nid, 0.0) for nid in owned_nodes],
                dtype=torch.float32,
                device=h.device,
            )
            price_lp, price_ent = self._evaluate_prices(
                price_scores, action_prices_tensor, price_budget
            )

        return move_lp + price_lp, move_ent + price_ent, value
