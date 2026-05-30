"""
Graphopoly training loop.

Can be called standalone or from the web GUI via server.py.
Supports callbacks for real-time updates and stop/pause events
for GUI control.
"""

from __future__ import annotations

import threading
from typing import Callable

import torch
from pathlib import Path

from backend.config import GraphopolyConfig
from backend.core.graph_world import GraphWorld
from backend.core.env import GraphopolyEnv
from backend.core.runner_utils import build_step_record, build_initial_step
from backend.agent.gnn_network import GraphopolyGNN
from backend.agent.ppo import PPOTrainer
from backend.logger import SimulationLogger


def train(
    config: GraphopolyConfig,
    world: GraphWorld,
    callback: Callable[[dict], None] | None = None,
    stop_event: threading.Event | None = None,
    pause_event: threading.Event | None = None,
) -> dict:
    """Run the full training loop.

    Args:
        config: Full config.
        world: Pre-built graph with ownership and destinations assigned.
        callback: Called after each episode with a metrics dict for GUI updates.
                  Signature: callback({"episode": int, "env_snapshot": dict, ...})
        stop_event: Set to stop training early.
        pause_event: Clear to pause, set to resume. If None, never pauses.

    Returns:
        Final training stats dict (run_dir, final_rewards, final_trips, stopped_early).
    """
    device = torch.device(config.device)
    env = GraphopolyEnv(config, world)
    num_agents = config.agent.num_agents

    # ── Build shared GNN + single shared optimizer ───────────────────────────
    # All agents share the same network weights and the same Adam state.
    # One optimizer avoids conflicting momentum / adaptive-rate drift that
    # would arise from A independent optimizers on the same parameters.
    shared_network = GraphopolyGNN(
        config=config.network,
    ).to(device)

    shared_optimizer = torch.optim.Adam(
        shared_network.parameters(),
        lr=config.train.lr,
    )

    # edge_index is constant throughout training (same graph)
    edge_index = env.get_edge_index().to(device)

    trainers: dict[int, PPOTrainer] = {
        aid: PPOTrainer(shared_network, shared_optimizer, config.train, edge_index)
        for aid in range(num_agents)
    }

    logger = SimulationLogger(config.log.log_dir, config)
    logger.save_graph(world)

    # Compute spring layout once — reused in every callback (not recomputed each episode)
    _spring_layout = world.get_spring_layout()

    stopped_early = False
    episode = 0
    # Initialise here so return statement is safe even if loop exits immediately
    episode_rewards: list[float] = []
    episode_trips: list[int] = []

    while True:
        # ── Stop check ───────────────────────────────────────────────────────
        if stop_event is not None and stop_event.is_set():
            stopped_early = True
            break

        # ── Pause check (blocks until resumed) ──────────────────────────────
        if pause_event is not None and not pause_event.is_set():
            pause_event.wait()

        # ── Reset episode ────────────────────────────────────────────────────
        env.reset()
        logger.log_initial_state(env.agents)

        # Step history for frontend animation (kept in memory per episode)
        step_history: list[dict] = [build_initial_step(env.agents)]

        # ── Step loop ────────────────────────────────────────────────────────
        for step in range(config.train.steps_per_episode):
            # Responsive pause / stop inside the step loop
            if pause_event is not None and not pause_event.is_set():
                pause_event.wait()
            if stop_event is not None and stop_event.is_set():
                stopped_early = True
                break

            # Compute shared node data once — all agents use it this step
            shared = env._build_shared_node_data()

            # Collect actions from all agents
            actions: list[dict] = []
            for aid in range(num_agents):
                node_feats   = env.get_node_features(aid, shared).to(device)
                current_pos  = env.agents[aid].position
                valid_nbrs   = env.get_valid_neighbors(aid)
                owned        = env.get_owned_nodes(aid)
                action, _lp, _val = trainers[aid].select_action(
                    node_feats, current_pos, valid_nbrs, owned,
                    price_budget=config.agent.price_budget,
                )
                actions.append(action)

            _obs_unused, rewards, done, info = env.step(actions)

            # Store rewards in each trainer's rollout buffer
            for aid in range(num_agents):
                trainers[aid].store_reward(rewards[aid], done)

            # Log this step (logger keeps cumulative node stats internally)
            logger.log_step(step, env.agents, actions, rewards, info)

            # Build the lightweight step record for frontend animation
            step_history.append(build_step_record(step + 1, env.agents, actions, rewards, info))

            if done:
                break

        # ── Stop check after inner loop ──────────────────────────────────────
        if stop_event is not None and stop_event.is_set():
            stopped_early = True

        # ── PPO update (skip on early stop) ──────────────────────────────────
        losses: dict[int, dict[str, float]] = {}
        if not stopped_early:
            # Compute shared data once for the bootstrap value step
            shared_final = env._build_shared_node_data()
            for aid in range(num_agents):
                node_feats  = env.get_node_features(aid, shared_final).to(device)
                current_pos = env.agents[aid].position
                valid_nbrs  = env.get_valid_neighbors(aid)
                owned       = env.get_owned_nodes(aid)
                last_val    = trainers[aid].get_value(node_feats, current_pos, valid_nbrs)
                losses[aid] = trainers[aid].update(last_val, owned, price_budget=config.agent.price_budget)

        # ── Log episode (training metrics + snapshot trajectory) ─────────────
        episode_rewards = [a.cumulative_reward for a in env.agents]
        episode_trips   = [a.trips_completed   for a in env.agents]
        logger.log_episode_end(episode, env.agents, losses)

        # ── Callback for GUI ─────────────────────────────────────────────────
        episode += 1

        if callback is not None:
            callback({
                "episode":         episode,
                "total_episodes":  0,  # 0 = unlimited — user stops manually
                "env_snapshot":    env.snapshot(),
                "graph_data":      world.to_dict(),
                "episode_rewards": [round(r, 2) for r in episode_rewards],
                "episode_trips":   episode_trips,
                "losses": {
                    aid: {k: round(v, 4) for k, v in l.items()}
                    for aid, l in losses.items()
                },
                "agent_details":   [a.to_dict() for a in env.agents],
                "step_history":    step_history,
                "graph_embedded":  world.to_dict(),
                "layout":          _spring_layout,   # pre-computed, not recalculated each episode
                "config_snapshot": config.to_dict(),
                "stopped_early":   stopped_early,
            })

        if stopped_early:
            break

    # ── Write final simulation JSON ──────────────────────────────────────────
    output_path = logger.finalize()

    return {
        "run_file":     output_path,
        "run_dir":      str(Path(output_path).parent),
        "final_rewards": episode_rewards,
        "final_trips":   episode_trips,
        "stopped_early": stopped_early,
    }


def train_standalone() -> None:
    """Run training from command line with default config."""
    import numpy as np

    config = GraphopolyConfig()
    rng    = np.random.default_rng(config.seed)

    world = GraphWorld.random_connected(
        config.graph.num_nodes,
        config.graph.num_edges,
        rng,
    )
    world.assign_territories(config.agent.num_agents, rng)
    world.assign_destinations(config.agent.num_agents, config.agent.num_destinations, rng)
    world.assign_starting_positions(config.agent.num_agents, rng)

    world.validate(
        config.agent.num_agents,
        min_destinations=config.agent.num_destinations,
        trip_reward=config.agent.trip_reward,
        price_budget=config.agent.price_budget,
    )

    def print_callback(metrics: dict) -> None:
        ep   = metrics["episode"]
        rews = metrics["episode_rewards"]
        trips = metrics["episode_trips"]
        if ep % 50 == 0:
            print(f"Episode {ep:5d} | Rewards: {rews} | Trips: {trips}")

    result = train(config, world, callback=print_callback)
    print(f"\nTraining complete. Simulation saved to: {result['run_file']}")


if __name__ == "__main__":
    train_standalone()
