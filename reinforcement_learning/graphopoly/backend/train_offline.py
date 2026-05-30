"""
Unified offline training for Graphopoly GNN models.

Three modes:
  curriculum  -- train on a single node size (variable agents 1-10)
  group       -- train on a GROUP of node sizes (tiny/small/medium/large/xl/xxl/huge)
  universal   -- train on ALL sizes 2-50

Usage:
    python -m backend.train_offline --mode curriculum --nodes 5 --graphs 200 --passes 10
    python -m backend.train_offline --mode group --group medium --graphs 200 --passes 10
    python -m backend.train_offline --mode universal --graphs-per-size 100 --passes 15
    python -m backend.train_offline --mode universal --passes 15 --resume
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, deque
from dataclasses import asdict
from pathlib import Path

# Force unbuffered output so prints appear immediately (even through pipes)
import builtins
_original_print = builtins.print
def _flush_print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _original_print(*args, **kwargs)
builtins.print = _flush_print

import networkx as nx
import numpy as np
import torch

from backend.config import GraphopolyConfig, NetworkConfig
from backend.core.graph_world import GraphWorld
from backend.core.env import GraphopolyEnv
from backend.agent.gnn_network import GraphopolyGNN
from backend.agent.ppo import PPOTrainer

MODELS_DIR = Path(__file__).parent.parent / "models"
TRAINING_DATA_DIR = Path(__file__).parent.parent / "training_data"

# ── Size groups for --mode group ────────────────────────────────────────

SIZE_GROUPS: dict[str, list[int]] = {
    "tiny":   list(range(2, 5)),      # 2-4
    "small":  list(range(5, 8)),      # 5-7
    "medium": list(range(8, 11)),     # 8-10
    "large":  list(range(11, 16)),    # 11-15
    "xl":     list(range(16, 21)),    # 16-20
    "xxl":    list(range(21, 31)),    # 21-30
    "huge":   list(range(31, 51)),    # 31-50
}

# ── Utilities ───────────────────────────────────────────────────────────


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def _detect_device() -> torch.device:
    """Auto-detect best device: CUDA -> CPU.

    MPS (Apple Silicon GPU) is supported but NOT auto-selected because the
    per-transition Python loop in PPO updates causes excessive CPU<->GPU
    synchronization, making MPS ~10-30x SLOWER than CPU for this workload.
    Use --device mps to force it (useful if the PPO loop is later vectorized).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Diverse graph pool builder ──────────────────────────────────────────


def _topology_edges(
    topology: str,
    n: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    """Return an edge list for a named topology on n nodes, or None if inapplicable."""
    max_e = n * (n - 1) // 2

    if topology == "path":
        return [(i, i + 1) for i in range(n - 1)]

    if topology == "ring":
        if n < 3:
            return None
        return [(i, (i + 1) % n) for i in range(n)]

    if topology == "star":
        return [(0, i) for i in range(1, n)]

    if topology == "double_star":
        if n < 4:
            return None
        mid = max(2, n // 2)
        edges = [(0, 1)]
        for i in range(2, mid):
            edges.append((0, i))
        for i in range(mid, n):
            edges.append((1, i))
        return edges

    if topology == "clustered":
        # Two dense clusters connected by a single bridge
        if n < 5:
            return None
        h = n // 2
        edges: list[tuple[int, int]] = []
        for i in range(h):
            for j in range(i + 1, h):
                edges.append((i, j))
        for i in range(h, n):
            for j in range(i + 1, n):
                edges.append((i, j))
        edges.append((h - 1, h))
        return edges

    if topology == "grid":
        if n < 4:
            return None
        cols = max(2, int(round(math.sqrt(n))))
        edges_set: set[tuple[int, int]] = set()
        for i in range(n):
            _, c = divmod(i, cols)
            if c + 1 < cols and i + 1 < n:
                edges_set.add((i, i + 1))
            if i + cols < n:
                edges_set.add((i, i + cols))
        for i in range(n - 1):  # guarantee connectivity
            edges_set.add((i, i + 1))
        return list(edges_set)

    if topology == "dense":
        # 65-90% of maximum possible edges
        target = max(n - 1, int(rng.uniform(0.65, 0.90) * max_e))
        perm = rng.permutation(n).tolist()
        edges_set = {
            (min(perm[i], perm[i + 1]), max(perm[i], perm[i + 1]))
            for i in range(n - 1)
        }
        all_missing = [
            (i, j) for i in range(n) for j in range(i + 1, n)
            if (i, j) not in edges_set
        ]
        idxs = rng.permutation(len(all_missing)).tolist()
        for k in idxs[: max(0, target - len(edges_set))]:
            edges_set.add(all_missing[k])
        return list(edges_set)

    if topology == "hub_spoke":
        # 2-4 hub nodes fully connected to each other, leaves distributed evenly
        if n < 4:
            return None
        num_hubs = min(4, max(2, n // 4))
        edges = []
        for i in range(num_hubs):
            for j in range(i + 1, num_hubs):
                edges.append((i, j))
        for k, leaf in enumerate(range(num_hubs, n)):
            edges.append((k % num_hubs, leaf))
        return edges

    return None


def build_diverse_graphs(
    num_nodes: int,
    num_graphs: int,
    min_agents: int = 1,
    max_agents: int = 10,
    rng: np.random.Generator | None = None,
) -> list[tuple[GraphWorld, int, int]]:
    """Build a structurally diverse pool of GraphWorld instances.

    Topology mix: ~60% structured (path, ring, star, double_star, clustered,
    grid, dense, hub_spoke) + ~40% random at varying edge densities.

    For small N where a topology degenerates, falls back to random.
    """
    if rng is None:
        rng = np.random.default_rng()

    STRUCTURED = ["path", "ring", "star", "double_star", "clustered",
                  "grid", "dense", "hub_spoke"]
    RANDOM_FRAC = 0.40

    n_random = max(1, int(round(num_graphs * RANDOM_FRAC)))
    n_structured = num_graphs - n_random
    per_topo = n_structured // len(STRUCTURED)
    extras = n_structured % len(STRUCTURED)

    plan: list[str] = []
    for i, t in enumerate(STRUCTURED):
        plan.extend([t] * (per_topo + (1 if i < extras else 0)))
    plan.extend(["random"] * n_random)
    rng.shuffle(plan)

    max_e = num_nodes * (num_nodes - 1) // 2
    max_a = min(max_agents, num_nodes)
    num_destinations = min(2, num_nodes)

    def _try_make(topology: str, graph_rng: np.random.Generator) -> GraphWorld | None:
        if topology == "random":
            num_e = int(graph_rng.integers(num_nodes - 1, max_e + 1))
            return GraphWorld.random_connected(num_nodes, num_e, graph_rng)
        edges = _topology_edges(topology, num_nodes, graph_rng)
        if edges is None:
            return GraphWorld.random_connected(num_nodes, None, graph_rng)
        world = GraphWorld.from_custom(edges, num_nodes)
        if not nx.is_connected(world.graph):
            return GraphWorld.random_connected(num_nodes, None, graph_rng)
        return world

    graphs: list[tuple[GraphWorld, int, int]] = []
    for topology in plan:
        num_agents = int(rng.integers(min_agents, max_a + 1))
        graph_rng = np.random.default_rng(int(rng.integers(0, 2 ** 31)))
        try:
            world = _try_make(topology, graph_rng) or GraphWorld.random_connected(
                num_nodes, None, graph_rng
            )
            world.assign_territories(num_agents, graph_rng)
            world.assign_destinations(num_agents, num_destinations, graph_rng)
            world.assign_starting_positions(num_agents, graph_rng)
            world.validate(num_agents, min_destinations=num_destinations,
                           trip_reward=25.0, price_budget=100.0)
            graphs.append((world, num_agents, num_destinations))
        except Exception:
            try:
                world = GraphWorld.random_connected(num_nodes, None, graph_rng)
                world.assign_territories(num_agents, graph_rng)
                world.assign_destinations(num_agents, num_destinations, graph_rng)
                world.assign_starting_positions(num_agents, graph_rng)
                world.validate(num_agents, min_destinations=num_destinations,
                               trip_reward=25.0, price_budget=100.0)
                graphs.append((world, num_agents, num_destinations))
            except Exception:
                continue

    if not graphs:
        raise RuntimeError(f"Could not build any valid {num_nodes}-node diverse graphs")
    return graphs


# ── Random graph pool builder (legacy, imported by other modules) ───────


def build_curriculum_graphs(
    num_nodes: int,
    num_graphs: int,
    min_agents: int = 1,
    max_agents: int = 10,
    rng: np.random.Generator | None = None,
) -> list[tuple[GraphWorld, int, int]]:
    """Build a pool of random graphs with variable agent counts.

    Returns list of (world, num_agents, num_destinations) tuples.
    """
    if rng is None:
        rng = np.random.default_rng()

    max_possible_edges = num_nodes * (num_nodes - 1) // 2
    min_edges = num_nodes - 1
    max_dests = num_nodes  # can't have more destinations than nodes

    graphs: list[tuple[GraphWorld, int, int]] = []
    attempts = 0

    while len(graphs) < num_graphs and attempts < num_graphs * 10:
        attempts += 1
        num_agents = int(rng.integers(min_agents, max_agents + 1))
        num_destinations = min(2, max_dests)  # 2 destinations or max available
        num_edges = int(rng.integers(min_edges, max_possible_edges + 1))

        graph_rng = np.random.default_rng(int(rng.integers(0, 2**31)))

        try:
            world = GraphWorld.random_connected(num_nodes, num_edges, graph_rng)
            world.assign_territories(num_agents, graph_rng)
            world.assign_destinations(num_agents, num_destinations, graph_rng)
            world.assign_starting_positions(num_agents, graph_rng)
            world.validate(
                num_agents,
                min_destinations=num_destinations,
                trip_reward=25.0,
                price_budget=100.0,
            )
            graphs.append((world, num_agents, num_destinations))
        except ValueError:
            continue

    if not graphs:
        raise RuntimeError(f"Could not build any valid {num_nodes}-node graphs")

    return graphs


# ── Shared rollout + PPO helpers ────────────────────────────────────────


def _collect_rollout(
    env: GraphopolyEnv,
    trainers: dict[int, PPOTrainer],
    config: GraphopolyConfig,
    device: torch.device,
) -> int:
    """Run one episode rollout, storing transitions in each trainer's buffer.

    Returns the number of steps actually taken.
    """
    env.reset()
    num_agents = config.agent.num_agents
    price_budget = config.agent.price_budget
    actual_steps = 0

    for step in range(config.train.steps_per_episode):
        shared = env._build_shared_node_data()
        actions: list[dict] = []

        for aid in range(num_agents):
            node_feats = env.get_node_features(aid, shared).to(device)
            current_pos = env.agents[aid].position
            valid_nbrs = env.get_valid_neighbors(aid)
            owned = env.get_owned_nodes(aid)
            action, _lp, _val = trainers[aid].select_action(
                node_feats, current_pos, valid_nbrs, owned,
                price_budget=price_budget,
            )
            actions.append(action)

        _obs, rewards, done, info = env.step(actions)
        actual_steps += 1

        for aid in range(num_agents):
            trainers[aid].store_reward(rewards[aid], done)
        if done:
            break

    return actual_steps


def _ppo_update(
    env: GraphopolyEnv,
    trainers: dict[int, PPOTrainer],
    config: GraphopolyConfig,
    device: torch.device,
) -> list[dict[str, float]]:
    """Run PPO update for all agents and return per-agent loss dicts."""
    num_agents = config.agent.num_agents
    price_budget = config.agent.price_budget
    shared_final = env._build_shared_node_data()
    ep_losses: list[dict[str, float]] = []

    for aid in range(num_agents):
        node_feats = env.get_node_features(aid, shared_final).to(device)
        current_pos = env.agents[aid].position
        valid_nbrs = env.get_valid_neighbors(aid)
        owned = env.get_owned_nodes(aid)
        last_val = trainers[aid].get_value(node_feats, current_pos, valid_nbrs)
        losses = trainers[aid].update(last_val, owned, price_budget=price_budget)
        ep_losses.append(losses)

    return ep_losses


# ── Entropy annealing + plateau detection helper ────────────────────────


class _EntropyController:
    """Manages entropy coefficient annealing and plateau-based bumps."""

    def __init__(self, config: GraphopolyConfig, total_episodes: int):
        self.config = config
        self.base_entropy = config.train.entropy_coef
        self.final_entropy = config.train.entropy_coef_final
        self.anneal_eps = max(1, int(total_episodes * config.train.entropy_anneal_frac))
        self.plateau_window: deque[float] = deque(maxlen=50)
        self.plateau_bumps = 0

    def step(self, ep_global: int, avg_reward: float) -> None:
        """Update the entropy coefficient for this episode."""
        # Base annealing
        if ep_global <= self.anneal_eps:
            ent_coef = self.base_entropy + (
                self.final_entropy - self.base_entropy
            ) * (ep_global / self.anneal_eps)
        else:
            ent_coef = self.final_entropy

        # Plateau bump override
        if self.plateau_bumps > 0:
            self.config.train.entropy_coef = max(ent_coef, self.config.train.entropy_coef)
            self.plateau_bumps -= 1
            if self.plateau_bumps == 0:
                self.config.train.entropy_coef = ent_coef
        else:
            self.config.train.entropy_coef = ent_coef

        # Plateau detection
        self.plateau_window.append(avg_reward)
        if (
            len(self.plateau_window) == self.plateau_window.maxlen
            and ep_global > 200
        ):
            first_half = list(self.plateau_window)[:25]
            second_half = list(self.plateau_window)[25:]
            improvement = (sum(second_half) / 25) - (sum(first_half) / 25)
            if abs(improvement) < 0.5:
                self.config.train.entropy_coef = min(
                    0.05, self.config.train.entropy_coef * 2
                )
                self.plateau_bumps = 20
                self.plateau_window.clear()


# ── Rolling metric tracker ──────────────────────────────────────────────


class _MetricTracker:
    """Rolling-window metric tracking for all modes."""

    def __init__(self, window: int = 100):
        self.window = window
        self.reward_window: deque[float] = deque(maxlen=window)
        self.trip_window: deque[float] = deque(maxlen=window)
        self.policy_loss_window: deque[float] = deque(maxlen=window)
        self.value_loss_window: deque[float] = deque(maxlen=window)
        self.entropy_window: deque[float] = deque(maxlen=window)
        self.tax_paid_window: deque[float] = deque(maxlen=window)
        self.best_trips_window: deque[int] = deque(maxlen=window)
        self.trips_per_step_window: deque[float] = deque(maxlen=window)
        self.dest_rev_window: deque[float] = deque(maxlen=window)
        self.tax_rev_window: deque[float] = deque(maxlen=window)

        self.best_avg_reward = float("-inf")
        self.best_avg_trips = 0.0
        self.prev_smooth_r: float | None = None

    def update(self, env: GraphopolyEnv, ep_losses: list[dict[str, float]], actual_steps: int):
        """Record one episode's metrics from env state and loss dicts."""
        agents = env.agents
        ep_rewards = [a.cumulative_reward for a in agents]
        ep_trips = [a.trips_completed for a in agents]
        ep_tax_paid = [a.tax_paid for a in agents]
        ep_dest_rev = [a.dest_revenue for a in agents]
        ep_tax_rev = [a.tax_revenue for a in agents]
        n = len(agents)

        self.avg_r = sum(ep_rewards) / n
        self.avg_t = sum(ep_trips) / n
        self.avg_pl = sum(l["policy_loss"] for l in ep_losses) / len(ep_losses)
        self.avg_vl = sum(l["value_loss"] for l in ep_losses) / len(ep_losses)
        self.avg_ent = sum(l["entropy"] for l in ep_losses) / len(ep_losses)

        self.ep_rewards = ep_rewards
        self.ep_trips = ep_trips
        self.ep_tax_paid = ep_tax_paid
        self.ep_dest_rev = ep_dest_rev
        self.ep_tax_rev = ep_tax_rev
        self.actual_steps = actual_steps

        self.reward_window.append(self.avg_r)
        self.trip_window.append(self.avg_t)
        self.policy_loss_window.append(self.avg_pl)
        self.value_loss_window.append(self.avg_vl)
        self.entropy_window.append(self.avg_ent)
        self.tax_paid_window.append(sum(ep_tax_paid) / n)
        self.best_trips_window.append(max(ep_trips))
        self.trips_per_step_window.append(sum(ep_trips) / max(actual_steps, 1))
        self.dest_rev_window.append(sum(ep_dest_rev) / n)
        self.tax_rev_window.append(sum(ep_tax_rev) / n)

        self.smooth_r = sum(self.reward_window) / len(self.reward_window)
        self.smooth_t = sum(self.trip_window) / len(self.trip_window)

        if self.smooth_r > self.best_avg_reward:
            self.best_avg_reward = self.smooth_r
        if self.smooth_t > self.best_avg_trips:
            self.best_avg_trips = self.smooth_t

    def smooth(self, name: str) -> float:
        w = getattr(self, f"{name}_window")
        return sum(w) / len(w) if w else 0.0


# ── JSONL record builders ───────────────────────────────────────────────


def _curriculum_record(
    tracker: _MetricTracker,
    ep_global: int,
    pass_num: int,
    graph_idx: int,
    num_agents: int,
    entropy_coef: float,
) -> dict:
    return {
        "episode": ep_global,
        "pass": pass_num,
        "graph_idx": graph_idx,
        "num_agents": num_agents,
        "avg_reward": round(tracker.avg_r, 3),
        "max_reward": round(max(tracker.ep_rewards), 3),
        "min_reward": round(min(tracker.ep_rewards), 3),
        "best_trips": max(tracker.ep_trips),
        "avg_trips": round(tracker.avg_t, 3),
        "trips_per_step": round(sum(tracker.ep_trips) / max(tracker.actual_steps, 1), 4),
        "avg_dest_rev": round(sum(tracker.ep_dest_rev) / len(tracker.ep_dest_rev), 3),
        "avg_tax_rev": round(sum(tracker.ep_tax_rev) / len(tracker.ep_tax_rev), 3),
        "avg_tax_paid": round(sum(tracker.ep_tax_paid) / len(tracker.ep_tax_paid), 3),
        "policy_loss": round(tracker.avg_pl, 6),
        "value_loss": round(tracker.avg_vl, 3),
        "entropy": round(tracker.avg_ent, 4),
        "entropy_coef": round(entropy_coef, 5),
        "per_agent_rewards": [round(r, 2) for r in tracker.ep_rewards],
        "per_agent_trips": list(tracker.ep_trips),
    }


def _multi_size_record(
    tracker: _MetricTracker,
    ep_global: int,
    epoch: int,
    num_nodes: int,
    num_agents: int,
) -> dict:
    return {
        "episode": ep_global,
        "epoch": epoch,
        "num_nodes": num_nodes,
        "num_agents": num_agents,
        "avg_reward": round(tracker.avg_r, 3),
        "avg_trips": round(tracker.avg_t, 3),
        "policy_loss": round(tracker.avg_pl, 6),
        "entropy": round(tracker.avg_ent, 4),
        "depth": GraphopolyGNN._get_depth(num_nodes),
    }


# ═══════════════════════════════════════════════════════════════════════
#  MODE 1: curriculum  (single node size)
# ═══════════════════════════════════════════════════════════════════════


def train_curriculum(
    num_nodes: int = 2,
    num_graphs: int = 200,
    num_passes: int = 10,
    lr: float = 3e-4,
    print_every: int = 5,
    save_every: int = 200,
    device_override: str | None = None,
) -> Path:
    """Train a GNN on a diverse pool of graphs of a single node size."""
    config = GraphopolyConfig()
    config.train.lr = lr
    device = torch.device(device_override) if device_override else _detect_device()

    total_episodes = num_graphs * num_passes

    # ── Header ──────────────────────────────────────────────────────────
    print()
    print("+" + "=" * 62 + "+")
    print("|        GRAPHOPOLY -- CURRICULUM GNN TRAINING                |")
    print("+" + "=" * 62 + "+")
    print()
    print(f"  Graph size:    {num_nodes} nodes")
    print(f"  Pool size:     {num_graphs} graphs (agents 1-10, variable edges)")
    print(f"  Passes:        {num_passes}")
    print(f"  Total episodes:{total_episodes:,}")
    print(f"  Steps/episode: {config.train.steps_per_episode}")
    print(f"  Learning rate: {lr}")
    print(f"  Device:        {device}")
    print()

    # ── Build graph pool ────────────────────────────────────────────────
    print("Building graph pool...")
    rng = np.random.default_rng()
    graphs = build_curriculum_graphs(num_nodes, num_graphs, rng=rng)
    print(f"  {len(graphs)} valid graphs ready")
    ac = Counter(g[1] for g in graphs)
    print(f"  Agent distribution: {dict(sorted(ac.items()))}")
    print()

    # ── Shared network + optimizer ──────────────────────────────────────
    network = GraphopolyGNN(config=config.network).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    param_count = sum(p.numel() for p in network.parameters())
    print(f"  Network:       {param_count:,} parameters")
    print(f"  Architecture:  {config.network.max_gnn_layers}x GATv2 (dynamic depth) "
          f"(H={config.network.hidden_dim}, heads={config.network.gat_heads})")
    print()

    # ── Metrics setup ───────────────────────────────────────────────────
    out_dir = TRAINING_DATA_DIR / f"size_{num_nodes}"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    print(f"  Metrics file:  {metrics_path}")
    print()

    entropy_ctrl = _EntropyController(config, total_episodes)
    tracker = _MetricTracker(window=min(100, total_episodes))
    t0 = time.time()
    ep_global = 0

    # ── Column header ───────────────────────────────────────────────────
    print("-" * 120)
    print(f"{'Episode':>8}  {'Pass':>5}  {'Agents':>6}  {'AvgReward':>10}  {'D':>7}  {'AvgTrips':>8}  "
          f"{'PolicyL':>8}  {'ValueL':>8}  {'Entropy':>8}  "
          f"{'ep/s':>6}  {'ETA':>6}")
    print(f"{'':>8}  {'':>5}  {'':>6}  {'BestTrips':>10}  {'T/Step':>7}  {'DestRev':>8}  "
          f"{'TaxRev':>8}  {'TaxPaid':>8}")
    print("-" * 120)

    with open(metrics_path, "w") as metrics_file:
        for pass_num in range(1, num_passes + 1):
            order = rng.permutation(len(graphs))

            for gi in order:
                world, num_agents, num_destinations = graphs[gi]
                ep_global += 1

                entropy_ctrl.step(ep_global, tracker.avg_r if ep_global > 1 else 0.0)

                config.agent.num_agents = num_agents
                config.agent.num_destinations = num_destinations

                env = GraphopolyEnv(config, world)
                edge_index = env.get_edge_index().to(device)
                trainers = {
                    aid: PPOTrainer(network, optimizer, config.train, edge_index)
                    for aid in range(num_agents)
                }

                actual_steps = _collect_rollout(env, trainers, config, device)
                ep_losses = _ppo_update(env, trainers, config, device)

                tracker.update(env, ep_losses, actual_steps)

                record = _curriculum_record(
                    tracker, ep_global, pass_num, int(gi),
                    num_agents, config.train.entropy_coef,
                )
                metrics_file.write(json.dumps(record) + "\n")
                metrics_file.flush()

                # ── Print ───────────────────────────────────────────────
                if ep_global % print_every == 0 or ep_global == 1:
                    now = time.time()
                    elapsed = now - t0
                    eps_per_sec = ep_global / elapsed
                    eta = (total_episodes - ep_global) / max(eps_per_sec, 0.01)

                    delta_str = ""
                    if tracker.prev_smooth_r is not None:
                        delta = tracker.smooth_r - tracker.prev_smooth_r
                        delta_str = f"{delta:>+7.1f}"
                    else:
                        delta_str = f"{'':>7}"
                    tracker.prev_smooth_r = tracker.smooth_r

                    print(
                        f"{ep_global:>8,}  {pass_num:>5}  {num_agents:>6}  "
                        f"{tracker.smooth_r:>+10.2f}  {delta_str}  {tracker.smooth_t:>8.2f}  "
                        f"{tracker.smooth('policy_loss'):>8.4f}  {tracker.smooth('value_loss'):>8.4f}  "
                        f"{tracker.smooth('entropy'):>8.3f}  "
                        f"{eps_per_sec:>5.1f}   {_fmt_time(eta):>5}"
                    )
                    print(
                        f"{'':>8}  {'':>5}  {'':>6}  "
                        f"{tracker.smooth('best_trips'):>10.1f}  "
                        f"{tracker.smooth('trips_per_step'):>7.3f}  "
                        f"{tracker.smooth('dest_rev'):>8.1f}  "
                        f"{tracker.smooth('tax_rev'):>8.1f}  "
                        f"{tracker.smooth('tax_paid'):>8.1f}"
                    )

                # ── Checkpoint ──────────────────────────────────────────
                if ep_global % save_every == 0:
                    MODELS_DIR.mkdir(exist_ok=True)
                    save_path = MODELS_DIR / f"model_{num_nodes}.pt"
                    torch.save({
                        "model_state_dict": network.state_dict(),
                        "network_config": asdict(config.network),
                        "num_nodes": num_nodes,
                        "episode": ep_global,
                        "best_avg_reward": tracker.best_avg_reward,
                    }, save_path)
                    print(f"         -> checkpoint saved to {save_path}")

            print(f"\n  -- Pass {pass_num}/{num_passes} complete "
                  f"(avg_reward={tracker.smooth_r:+.2f}, avg_trips={tracker.smooth_t:.2f}) --\n")

    # ── Final save ──────────────────────────────────────────────────────
    MODELS_DIR.mkdir(exist_ok=True)
    save_path = MODELS_DIR / f"model_{num_nodes}.pt"
    torch.save({
        "model_state_dict": network.state_dict(),
        "network_config": asdict(config.network),
        "num_nodes": num_nodes,
        "episode": ep_global,
        "best_avg_reward": tracker.best_avg_reward,
    }, save_path)

    elapsed = time.time() - t0
    WINDOW = tracker.window
    print()
    print("-" * 120)
    print()
    print("  CURRICULUM TRAINING COMPLETE")
    print()
    print(f"  Total time:           {_fmt_time(elapsed)}")
    print(f"  Total episodes:       {ep_global:,}")
    print(f"  Throughput:           {ep_global / elapsed:.1f} ep/s")
    print()
    print(f"  Best avg reward:      {tracker.best_avg_reward:+.2f}  (rolling {WINDOW}-ep window)")
    print(f"  Best avg trips:       {tracker.best_avg_trips:.2f}")
    print(f"  Final avg reward:     {tracker.smooth_r:+.2f}")
    print(f"  Final avg trips:      {tracker.smooth_t:.2f}")
    print()
    print(f"  Metrics saved to:     {metrics_path}")
    print(f"  Model saved to:       {save_path}")
    print()
    return save_path


# ═══════════════════════════════════════════════════════════════════════
#  MODE 2 & 3: group / universal  (multiple node sizes)
# ═══════════════════════════════════════════════════════════════════════


def train_multi_size(
    sizes: list[int],
    label: str,
    graphs_per_size: int = 100,
    num_passes: int = 15,
    lr: float = 3e-4,
    benchmark_per_size: int = 3,
    resume: bool = False,
    print_every: int = 25,
    device_override: str | None = None,
) -> Path:
    """Train a single GNN on graphs across multiple node sizes.

    Used by both ``group`` and ``universal`` modes.
    """
    config = GraphopolyConfig()
    config.train.lr = lr
    device = torch.device(device_override) if device_override else _detect_device()

    num_sizes = len(sizes)

    # ── Header ──────────────────────────────────────────────────────────
    print()
    print("+" + "=" * 62 + "+")
    print(f"|  GRAPHOPOLY -- {label.upper():^46} |")
    print("+" + "=" * 62 + "+")
    print()

    # ── Build graph pools ───────────────────────────────────────────────
    size_range_str = f"{min(sizes)}-{max(sizes)}" if num_sizes > 1 else str(sizes[0])
    print(f"Building graph pools for sizes {size_range_str}...")
    rng = np.random.default_rng(42)

    train_graphs: list[tuple[GraphWorld, int, int, int]] = []
    benchmark_graphs: list[tuple[GraphWorld, int, int, int]] = []

    for n in sizes:
        total_needed = graphs_per_size + benchmark_per_size
        pool = build_curriculum_graphs(n, total_needed, rng=rng)
        for i, (world, na, nd) in enumerate(pool):
            if i < graphs_per_size:
                train_graphs.append((world, na, nd, n))
            elif i < graphs_per_size + benchmark_per_size:
                benchmark_graphs.append((world, na, nd, n))

    total_train = len(train_graphs)
    total_benchmark = len(benchmark_graphs)
    total_episodes = total_train * num_passes

    print(f"  Sizes:           {size_range_str} ({num_sizes} sizes)")
    print(f"  Training graphs: {total_train} ({graphs_per_size}/size)")
    print(f"  Benchmark:       {total_benchmark} ({benchmark_per_size}/size)")
    print(f"  Passes/epochs:   {num_passes}")
    print(f"  Total episodes:  {total_episodes:,}")
    print(f"  Features/node:   {config.network.node_feature_dim}")
    print(f"  Max GNN depth:   {config.network.max_gnn_layers} (dynamic: 2-7)")
    print(f"  Device:          {device}")
    print()

    # ── Network + optimizer ─────────────────────────────────────────────
    model_name = f"model_{label.replace(' ', '_')}"
    network = GraphopolyGNN(config=config.network).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    start_epoch = 0
    ep_global = 0
    best_bmk_reward = float("-inf")

    if resume:
        model_path = MODELS_DIR / f"{model_name}.pt"
        if model_path.exists():
            ckpt = torch.load(model_path, map_location=device, weights_only=True)
            net_cfg = NetworkConfig(**{
                k: v for k, v in ckpt["network_config"].items()
                if k in NetworkConfig.__dataclass_fields__
            })
            network = GraphopolyGNN(config=net_cfg).to(device)
            network.load_state_dict(ckpt["model_state_dict"])
            optimizer = torch.optim.Adam(network.parameters(), lr=lr)
            start_epoch = ckpt.get("epoch", 0)
            ep_global = ckpt.get("episode", 0)
            best_bmk_reward = ckpt.get("best_benchmark_reward", float("-inf"))
            print(f"  Resumed from epoch {start_epoch}, episode {ep_global}")

    param_count = sum(p.numel() for p in network.parameters())
    print(f"  Network:         {param_count:,} parameters")
    print()

    # ── Metrics output ──────────────────────────────────────────────────
    safe_label = label.replace(" ", "_")
    out_dir = TRAINING_DATA_DIR / safe_label
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    epochs_path = out_dir / "epochs.jsonl"
    print(f"  Metrics:         {metrics_path}")
    print()

    # ── Tracking ────────────────────────────────────────────────────────
    entropy_ctrl = _EntropyController(config, total_episodes)
    tracker = _MetricTracker(window=50)

    # Per-size rolling windows for the periodic size breakdown
    size_reward_windows: dict[int, deque] = {n: deque(maxlen=20) for n in sizes}
    size_trip_windows: dict[int, deque] = {n: deque(maxlen=20) for n in sizes}

    t0 = time.time()
    last_size_print = 0

    # ── Column header ───────────────────────────────────────────────────
    print("-" * 100)
    print(f"{'Epoch':>5}  {'Episode':>8}  {'N':>3}  {'A':>3}  {'Depth':>5}  "
          f"{'AvgRew':>8}  {'AvgTrip':>7}  "
          f"{'PolicyL':>8}  {'Ent':>6}  "
          f"{'ep/s':>6}  {'Elapsed':>7}  {'ETA':>6}")
    print("-" * 100)

    file_mode = "a" if resume else "w"
    with open(metrics_path, file_mode) as mf, open(epochs_path, file_mode) as ef:
        for epoch in range(start_epoch + 1, start_epoch + num_passes + 1):
            order = rng.permutation(total_train)
            epoch_rewards: list[float] = []
            epoch_trips: list[float] = []
            epoch_t0 = time.time()

            for oi, idx in enumerate(order):
                world, num_agents, num_destinations, num_nodes = train_graphs[idx]
                ep_global += 1

                entropy_ctrl.step(ep_global, tracker.avg_r if ep_global > 1 else 0.0)

                config.agent.num_agents = num_agents
                config.agent.num_destinations = num_destinations

                env = GraphopolyEnv(config, world)
                edge_index = env.get_edge_index().to(device)
                trainers = {
                    aid: PPOTrainer(network, optimizer, config.train, edge_index)
                    for aid in range(num_agents)
                }

                actual_steps = _collect_rollout(env, trainers, config, device)
                ep_losses = _ppo_update(env, trainers, config, device)

                tracker.update(env, ep_losses, actual_steps)
                epoch_rewards.append(tracker.avg_r)
                epoch_trips.append(tracker.avg_t)

                # Per-size tracking
                size_reward_windows[num_nodes].append(tracker.avg_r)
                size_trip_windows[num_nodes].append(tracker.avg_t)

                # JSONL record
                record = _multi_size_record(
                    tracker, ep_global, epoch, num_nodes, num_agents,
                )
                mf.write(json.dumps(record) + "\n")

                # ── Print every N episodes ──────────────────────────────
                if ep_global % print_every == 0:
                    now = time.time()
                    elapsed = now - t0
                    eps_per_sec = ep_global / elapsed if elapsed > 0 else 0
                    remaining_eps = total_episodes - (ep_global - start_epoch * total_train)
                    eta = remaining_eps / max(eps_per_sec, 0.01)
                    depth = GraphopolyGNN._get_depth(num_nodes)

                    print(
                        f"{epoch:>5}  {ep_global:>8,}  {num_nodes:>3}  {num_agents:>3}  "
                        f"{depth:>5}  "
                        f"{tracker.smooth_r:>+8.2f}  {tracker.smooth_t:>7.2f}  "
                        f"{tracker.smooth('policy_loss'):>8.4f}  "
                        f"{tracker.smooth('entropy'):>6.3f}  "
                        f"{eps_per_sec:>5.1f}   {_fmt_time(elapsed):>6}  {_fmt_time(eta):>5}"
                    )

                # Print per-size breakdown periodically
                if ep_global % (print_every * 10) == 0 and ep_global > last_size_print:
                    last_size_print = ep_global
                    # Pick a representative sample of sizes to display
                    display_sizes = sorted(set(
                        [sizes[0], sizes[-1]]
                        + [sizes[len(sizes) // 4], sizes[len(sizes) // 2],
                           sizes[3 * len(sizes) // 4]]
                    ))
                    parts = []
                    for n in display_sizes:
                        if n in size_reward_windows:
                            w = size_reward_windows[n]
                            if w:
                                sr = sum(w) / len(w)
                                tw = size_trip_windows[n]
                                st = sum(tw) / len(tw) if tw else 0
                                parts.append(f"N={n}:{sr:>+.1f}r/{st:.1f}t")
                    if parts:
                        print(f"         Size breakdown: {' | '.join(parts)}")

            # ── End of epoch: flush + benchmark ─────────────────────────
            mf.flush()
            epoch_elapsed = time.time() - epoch_t0
            epoch_avg_r = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0
            epoch_avg_t = sum(epoch_trips) / len(epoch_trips) if epoch_trips else 0
            print(f"\n  -- Epoch {epoch} done ({_fmt_time(epoch_elapsed)}) -- "
                  f"avg reward: {epoch_avg_r:+.2f}, avg trips: {epoch_avg_t:.2f}")
            print(f"     Running benchmark ({total_benchmark} graphs)...")

            bmk_results: dict[int, list[float]] = {n: [] for n in sizes}
            bmk_trips: dict[int, list[float]] = {n: [] for n in sizes}

            network.eval()
            with torch.no_grad():
                for world, num_agents, num_destinations, num_nodes in benchmark_graphs:
                    config.agent.num_agents = num_agents
                    config.agent.num_destinations = num_destinations

                    env = GraphopolyEnv(config, world)
                    edge_index = env.get_edge_index().to(device)
                    env.reset()

                    price_budget = config.agent.price_budget
                    for step in range(config.train.steps_per_episode):
                        shared = env._build_shared_node_data()
                        actions: list[dict] = []
                        for aid in range(num_agents):
                            node_feats = env.get_node_features(aid, shared).to(device)
                            current_pos = env.agents[aid].position
                            valid_nbrs = env.get_valid_neighbors(aid)
                            owned = env.get_owned_nodes(aid)
                            action, _, _, _ = network.get_action_and_value(
                                node_feats, edge_index,
                                current_pos, valid_nbrs, owned,
                                deterministic=True,
                                price_budget=price_budget,
                            )
                            actions.append(action)
                        _obs, rewards, _done, info = env.step(actions)

                    ep_rewards = [a.cumulative_reward for a in env.agents]
                    ep_trips = [a.trips_completed for a in env.agents]
                    bmk_results[num_nodes].append(sum(ep_rewards) / len(ep_rewards))
                    bmk_trips[num_nodes].append(sum(ep_trips) / len(ep_trips))

            network.train()

            bmk_summary = {}
            overall_bmk = []
            for n in sizes:
                if bmk_results[n]:
                    avg_r = sum(bmk_results[n]) / len(bmk_results[n])
                    avg_t = sum(bmk_trips[n]) / len(bmk_trips[n])
                    bmk_summary[n] = {"reward": round(avg_r, 2), "trips": round(avg_t, 2)}
                    overall_bmk.append(avg_r)

            overall_avg = sum(overall_bmk) / len(overall_bmk) if overall_bmk else 0

            print(f"     Benchmark results:")
            # Show a sample of sizes
            display_sizes = sorted(set(
                [sizes[0], sizes[-1]]
                + [sizes[len(sizes) // 4], sizes[len(sizes) // 2],
                   sizes[3 * len(sizes) // 4]]
            ))
            for n in display_sizes:
                if n in bmk_summary:
                    s = bmk_summary[n]
                    print(f"       N={n:>2}: reward={s['reward']:>+7.2f}  trips={s['trips']:.2f}")
            print(f"       Overall: {overall_avg:+.2f}")

            epoch_record = {
                "epoch": epoch,
                "episode": ep_global,
                "avg_train_reward": round(epoch_avg_r, 3),
                "avg_train_trips": round(epoch_avg_t, 3),
                "benchmark": {str(k): v for k, v in bmk_summary.items()},
                "benchmark_overall": round(overall_avg, 3),
            }
            ef.write(json.dumps(epoch_record) + "\n")
            ef.flush()

            # ── Checkpoint ──────────────────────────────────────────────
            MODELS_DIR.mkdir(exist_ok=True)
            save_path = MODELS_DIR / f"{model_name}.pt"
            ckpt_data = {
                "model_state_dict": network.state_dict(),
                "network_config": asdict(config.network),
                "epoch": epoch,
                "episode": ep_global,
                "best_benchmark_reward": max(best_bmk_reward, overall_avg),
                "trained_sizes": sizes,
                "node_feature_dim": config.network.node_feature_dim,
                "max_gnn_layers": config.network.max_gnn_layers,
            }
            torch.save(ckpt_data, save_path)

            if overall_avg > best_bmk_reward:
                best_bmk_reward = overall_avg
                torch.save(ckpt_data, MODELS_DIR / f"{model_name}_best.pt")
                print(f"     * New best benchmark: {best_bmk_reward:+.2f}")

            print()

    # ── Summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("=" * 100)
    print()
    print(f"  {label.upper()} TRAINING COMPLETE")
    print()
    print(f"  Total time:           {_fmt_time(elapsed)}")
    print(f"  Total episodes:       {ep_global:,}")
    print(f"  Throughput:           {ep_global / elapsed:.1f} ep/s")
    print()
    print(f"  Best benchmark:       {best_bmk_reward:+.2f}")
    print(f"  Best avg reward:      {tracker.best_avg_reward:+.2f}  (rolling {tracker.window}-ep)")
    print()
    print(f"  Model:                {save_path}")
    print(f"  Best model:           {MODELS_DIR / f'{model_name}_best.pt'}")
    print(f"  Metrics:              {metrics_path}")
    print()

    summary = {
        "total_time_seconds": round(elapsed, 1),
        "total_episodes": ep_global,
        "epochs": num_passes,
        "best_benchmark_reward": round(best_bmk_reward, 3),
        "best_avg_reward": round(tracker.best_avg_reward, 3),
        "graphs_per_size": graphs_per_size,
        "sizes": sizes,
        "device": str(device),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return save_path


# ═══════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Unified offline training for Graphopoly GNN models",
    )
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["curriculum", "group", "universal"],
        help="Training mode: curriculum (single size), group (size group), universal (all 2-50)",
    )

    # Curriculum-specific
    parser.add_argument("--nodes", type=int, default=2,
                        help="Graph size for curriculum mode (default: 2)")

    # Group-specific
    parser.add_argument("--group", type=str, default=None,
                        choices=list(SIZE_GROUPS.keys()),
                        help="Size group name for group mode")

    # Shared
    parser.add_argument("--graphs", type=int, default=None,
                        help="Graphs per size (curriculum: total pool, group/universal: per size)")
    parser.add_argument("--graphs-per-size", type=int, default=None,
                        help="Alias for --graphs (group/universal modes)")
    parser.add_argument("--passes", type=int, default=10,
                        help="Number of passes/epochs through the pool (default: 10)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--print-every", type=int, default=None,
                        help="Print metrics every N episodes")
    parser.add_argument("--save-every", type=int, default=200,
                        help="Checkpoint every N episodes (curriculum mode, default: 200)")
    parser.add_argument("--benchmark", type=int, default=3,
                        help="Benchmark graphs per size for group/universal (default: 3)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoint (group/universal)")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (cpu/mps/cuda). Default: auto-detect")

    args = parser.parse_args()

    if args.mode == "curriculum":
        num_graphs = args.graphs if args.graphs is not None else 200
        print_every = args.print_every if args.print_every is not None else 5

        if not 2 <= args.nodes <= 50:
            parser.error("--nodes must be between 2 and 50")

        train_curriculum(
            num_nodes=args.nodes,
            num_graphs=num_graphs,
            num_passes=args.passes,
            lr=args.lr,
            print_every=print_every,
            save_every=args.save_every,
            device_override=args.device,
        )

    elif args.mode == "group":
        if args.group is None:
            parser.error("--group is required for group mode")

        sizes = SIZE_GROUPS[args.group]
        graphs_per_size = args.graphs_per_size or args.graphs or 100
        print_every = args.print_every if args.print_every is not None else 25
        label = f"group_{args.group}"

        train_multi_size(
            sizes=sizes,
            label=label,
            graphs_per_size=graphs_per_size,
            num_passes=args.passes,
            lr=args.lr,
            benchmark_per_size=args.benchmark,
            resume=args.resume,
            print_every=print_every,
            device_override=args.device,
        )

    elif args.mode == "universal":
        sizes = list(range(2, 51))  # 2-50 nodes
        graphs_per_size = args.graphs_per_size or args.graphs or 100
        print_every = args.print_every if args.print_every is not None else 25

        train_multi_size(
            sizes=sizes,
            label="universal",
            graphs_per_size=graphs_per_size,
            num_passes=args.passes,
            lr=args.lr,
            benchmark_per_size=args.benchmark,
            resume=args.resume,
            print_every=print_every,
            device_override=args.device,
        )


if __name__ == "__main__":
    main()
