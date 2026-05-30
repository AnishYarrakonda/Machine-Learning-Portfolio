"""
Inference-only simulation loop for Graphopoly.

Loads a pre-trained (or randomly initialised) GNN model and runs
a single continuous session without learning.  Used by the web UI
"Start Simulation" button.

The simulation keeps stepping until the user presses Stop.  There is
no episode boundary — prices, positions, and cumulative rewards are
never reset mid-run.  A callback fires every `steps_per_episode`
steps (default 100) so the frontend can animate progress, but the
environment state is *not* reset between callbacks.

Provides the same callback / stop / pause interface as train.train()
so the WebSocket plumbing stays identical.
"""

from __future__ import annotations

import threading
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import torch

from backend.config import GraphopolyConfig, NetworkConfig
from backend.core.graph_world import GraphWorld
from backend.core.env import GraphopolyEnv
from backend.core.runner_utils import build_step_record, build_initial_step
from backend.agent.gnn_network import GraphopolyGNN

MODELS_DIR = Path(__file__).parent.parent / "models"


def _ensure_model(network_config: NetworkConfig) -> Path:
    """Return path to the universal model file, creating a randomly initialised one if needed."""
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / "model_universal.pt"
    if not path.exists():
        network = GraphopolyGNN(config=network_config)
        torch.save({
            "model_state_dict": network.state_dict(),
            "network_config": asdict(network_config),
        }, path)
    return path


def simulate(
    config: GraphopolyConfig,
    world: GraphWorld,
    callback: Callable[[dict], None] | None = None,
    stop_event: threading.Event | None = None,
    pause_event: threading.Event | None = None,
) -> dict:
    """Run a single continuous inference-only session (no PPO updates).

    The environment is reset **once** at the start.  Steps continue
    indefinitely until the user sets ``stop_event``.  A callback fires
    every ``steps_per_episode`` steps with the accumulated step history
    so the frontend can animate and save data.

    Args:
        config:      Full config (used for env setup + device selection).
        world:       Pre-built graph with ownership / destinations assigned.
        callback:    Called periodically — same dict shape as train.train().
        stop_event:  Set by the UI to stop.
        pause_event: Clear to pause, set to resume.

    Returns:
        Final stats dict (same keys as train.train()).
    """
    device = torch.device(config.device)
    num_agents = config.agent.num_agents
    N = world.num_nodes

    # ── Load (or create) model ────────────────────────────────────────────
    model_path = _ensure_model(config.network)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    net_cfg = NetworkConfig(**{
        k: v for k, v in checkpoint["network_config"].items()
        if k in NetworkConfig.__dataclass_fields__
    })
    network = GraphopolyGNN(config=net_cfg)
    network.load_state_dict(checkpoint["model_state_dict"])
    network.to(device)
    network.eval()

    env = GraphopolyEnv(config, world)
    edge_index = env.get_edge_index().to(device)
    _spring_layout = world.get_spring_layout()

    stopped_early = False
    callback_count = 0
    global_step = 0

    # ── Single reset at the start of the simulation ───────────────────────
    env.reset()

    # How often to fire callbacks (so frontend can animate)
    callback_interval = config.train.steps_per_episode

    # step_history holds the CURRENT WINDOW only (reset after each broadcast).
    # This bounds the WS payload to ~callback_interval entries regardless of
    # how long the simulation runs, preventing browser OOM on long sessions.
    step_history: list[dict] = [build_initial_step(env.agents)]

    # ── Continuous step loop ──────────────────────────────────────────────
    while True:
        # Stop check
        if stop_event is not None and stop_event.is_set():
            stopped_early = True
            break

        # Pause check
        if pause_event is not None and not pause_event.is_set():
            pause_event.wait()

        # ── One step ──────────────────────────────────────────────────────
        shared = env._build_shared_node_data()
        actions: list[dict] = []

        with torch.no_grad():
            for aid in range(num_agents):
                node_feats = env.get_node_features(aid, shared).to(device)
                current_pos = env.agents[aid].position
                valid_nbrs = env.get_valid_neighbors(aid)
                owned = env.get_owned_nodes(aid)
                action, _, _, _ = network.get_action_and_value(
                    node_feats, edge_index,
                    current_pos, valid_nbrs, owned,
                    price_budget=config.agent.price_budget,
                )
                actions.append(action)

        # Override the done flag — we never want the env to declare done
        env.step_count = 0  # prevent env.step() from returning done=True
        _obs, rewards, _done, info = env.step(actions)

        global_step += 1

        step_history.append(build_step_record(global_step, env.agents, actions, rewards, info))

        # ── Periodic callback so frontend gets updates ────────────────────
        if global_step % callback_interval == 0:
            callback_count += 1

            if callback is not None:
                episode_rewards = [a.cumulative_reward for a in env.agents]
                episode_trips = [a.trips_completed for a in env.agents]
                callback({
                    "episode":         callback_count,
                    "total_episodes":  0,
                    "env_snapshot":    env.snapshot(),
                    "graph_data":      world.to_dict(),
                    "episode_rewards": [round(r, 2) for r in episode_rewards],
                    "episode_trips":   episode_trips,
                    "losses":          {},
                    "agent_details":   [a.to_dict() for a in env.agents],
                    "step_history":    step_history,
                    "graph_embedded":  world.to_dict(),
                    "layout":          _spring_layout,
                    "config_snapshot": config.to_dict(),
                    "stopped_early":   False,
                })

            # Reset the window — carry the last step forward as the new step-0
            # so the next animation window starts from the correct position.
            last = step_history[-1]
            step_history = [{**last, "step": 0, "actions": [], "rewards": [], "taxes": {}, "dest_completions": []}]

    # ── Final callback on stop ────────────────────────────────────────────
    episode_rewards = [a.cumulative_reward for a in env.agents]
    episode_trips = [a.trips_completed for a in env.agents]

    if callback is not None:
        callback({
            "episode":         callback_count + 1,
            "total_episodes":  0,
            "env_snapshot":    env.snapshot(),
            "graph_data":      world.to_dict(),
            "episode_rewards": [round(r, 2) for r in episode_rewards],
            "episode_trips":   episode_trips,
            "losses":          {},
            "agent_details":   [a.to_dict() for a in env.agents],
            "step_history":    step_history,
            "graph_embedded":  world.to_dict(),
            "layout":          _spring_layout,
            "config_snapshot": config.to_dict(),
            "stopped_early":   stopped_early,
        })

    return {
        "final_rewards": episode_rewards,
        "final_trips":   episode_trips,
        "stopped_early": stopped_early,
    }
