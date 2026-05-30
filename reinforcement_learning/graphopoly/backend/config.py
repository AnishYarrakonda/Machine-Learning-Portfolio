"""
Graphopoly configuration — all tunable parameters in one place.

Modify values here directly. The GUI reads this at startup;
changes take effect on next training run.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json
from pathlib import Path


def _auto_device() -> str:
    """Detect the best available compute device: MPS → CUDA → CPU."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


@dataclass
class GraphConfig:
    """Graph topology parameters (used only for random generation)."""
    num_nodes: int = 2
    num_edges: int | None = None  # None = auto (spanning tree + extras). Must be >= N-1.

    # Override with explicit graph (set by GUI graph builder)
    custom_edges: list[tuple[int, int]] | None = None
    custom_ownership: dict[int, int] | None = None       # {node_id: agent_id}
    custom_destinations: dict[int, list[int]] | None = None  # {agent_id: [node_ids]}
    custom_starting_positions: dict[int, int] | None = None  # {agent_id: node_id}


@dataclass
class AgentConfig:
    """Per-agent parameters."""
    num_agents: int = 2
    num_destinations: int = 2
    trip_reward: float = 25.0        # reward per destination-to-destination trip completion
    price_budget: float = 100.0      # fixed total pricing budget per agent (distributed via softmax)

    def __post_init__(self):
        if self.trip_reward <= 0:
            raise ValueError(f"trip_reward must be positive, got {self.trip_reward}")
        if self.price_budget < 0:
            raise ValueError(f"price_budget must be non-negative, got {self.price_budget}")


@dataclass
class TrainConfig:
    """PPO and training loop parameters."""
    steps_per_episode: int = 75
    num_episodes: int = 5000
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    entropy_coef_final: float = 0.001    # anneal entropy to this value
    entropy_anneal_frac: float = 0.3     # fraction of training over which to anneal
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    batch_size: int = 64


@dataclass
class NetworkConfig:
    """Per-agent GNN architecture (GATv2)."""
    node_feature_dim: int = 13    # per-node input features
    hidden_dim: int = 64          # node embedding size per GATv2 layer
    max_gnn_layers: int = 7       # max GATv2 layers (dynamic depth: 2-7 based on graph size)
    gat_heads: int = 4            # attention heads per layer (concat=False → output stays hidden_dim)
    move_mlp_hidden: int = 32     # hidden size of movement scoring MLP
    dropout: float = 0.0


@dataclass
class LogConfig:
    """Logging parameters."""
    log_dir: str = "episodes/"
    log_every: int = 50         # stats snapshot every N steps within an episode
    save_full_history: bool = True


@dataclass
class GraphopolyConfig:
    """Top-level config aggregating all sub-configs."""
    graph: GraphConfig = field(default_factory=GraphConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    log: LogConfig = field(default_factory=LogConfig)
    seed: int = 42
    device: str = field(default_factory=_auto_device)  # auto-detected: mps → cuda → cpu

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> GraphopolyConfig:
        # Handle legacy key renames
        train_d = d.get("train", {})
        if "episode_length" in train_d and "steps_per_episode" not in train_d:
            train_d["steps_per_episode"] = train_d.pop("episode_length")
        if "rollout_length" in train_d:
            train_d.pop("rollout_length", None)

        log_d = d.get("log", {})
        if "log_interval" in log_d and "log_every" not in log_d:
            log_d["log_every"] = log_d.pop("log_interval")

        net_d = d.get("network", {})
        if "hidden_dims" in net_d and "hidden_dim" not in net_d:
            net_d.pop("hidden_dims")  # discard old MLP key; new GNN defaults apply
        if "num_gnn_layers" in net_d and "max_gnn_layers" not in net_d:
            net_d["max_gnn_layers"] = net_d.pop("num_gnn_layers")

        agent_d = d.get("agent", {})
        # Legacy: max_price / initial_price → price_budget
        if "max_price" in agent_d and "price_budget" not in agent_d:
            agent_d["price_budget"] = float(agent_d.pop("max_price")) * 5
        agent_d.pop("max_price", None)
        agent_d.pop("initial_price", None)

        graph_d = d.get("graph", {})
        if "edge_probability" in graph_d and "num_edges" not in graph_d:
            graph_d.pop("edge_probability")
        if "min_degree" in graph_d:
            graph_d.pop("min_degree", None)

        return cls(
            graph=GraphConfig(**{k: v for k, v in graph_d.items() if k in GraphConfig.__dataclass_fields__}),
            agent=AgentConfig(**{k: v for k, v in agent_d.items() if k in AgentConfig.__dataclass_fields__}),
            train=TrainConfig(**{k: v for k, v in train_d.items() if k in TrainConfig.__dataclass_fields__}),
            network=NetworkConfig(**{k: v for k, v in net_d.items() if k in NetworkConfig.__dataclass_fields__}),
            log=LogConfig(**{k: v for k, v in log_d.items() if k in LogConfig.__dataclass_fields__}),
            seed=d.get("seed", 42),
            device=d.get("device", _auto_device()),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> GraphopolyConfig:
        with open(path) as f:
            return cls.from_dict(json.load(f))
