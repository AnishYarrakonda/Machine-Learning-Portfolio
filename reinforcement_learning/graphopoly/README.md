# Graphopoly — Multi-Agent RL on Graphs

A game-theoretic multi-agent reinforcement learning research platform where agents navigate a graph, earn rewards by travelling between destination nodes, and compete by setting tolls on owned territory. Agents learn via **PPO** using a **shared GATv2 Graph Neural Network** — enabling emergent economic strategies to be studied across arbitrary graph topologies.

---

## Quick Start

**Requirements:** Python 3.10+, Node 18+

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install frontend dependencies (first time only)
cd frontend && npm install && cd ..

# 3. Start everything — backend + frontend together
python3 main.py
```

Open **http://localhost:5173** in your browser.

> **Tip:** If the UI looks outdated, do a hard refresh: **Cmd+Shift+R** (Mac) or **Ctrl+Shift+R** (Windows/Linux).

### Backend only
```bash
python3 main.py --backend-only
```

---

## Offline Training

```bash
# Train on a pool of random 8-node graphs
python -m backend.train_offline --nodes 8 --pool-size 50 --episodes 5000

# Curriculum training across many graph sizes and agent counts
python -m backend.train_offline --nodes 8 --pool-size 100 --episodes 10000 --print-every 10
```

Trained models are saved to `models/latest_model.pt`. The web UI loads this model automatically when you start a simulation.

---

## Project Structure

```
graphopoly/
├── main.py                    # Single entry point — starts FastAPI + Vite together
├── CLAUDE.md                  # AI assistant context (architecture, decisions, bugs)
├── requirements.txt
│
├── models/
│   └── latest_model.pt        # Pre-trained GNN (auto-loaded by simulation)
│
├── backend/
│   ├── config.py              # All hyperparameters — edit to change defaults
│   ├── server.py              # FastAPI + WebSocket server
│   ├── simulate.py            # Inference-only simulation loop (used by web UI)
│   ├── train.py               # Single-graph training loop
│   ├── train_offline.py       # CLI offline training with graph pools
│   ├── analyze.py             # Post-episode analysis utilities
│   ├── agent/
│   │   ├── gnn_network.py     # GraphopolyGNN: shared GATv2 policy + value network
│   │   └── ppo.py             # PPO trainer (shared network + shared optimizer)
│   └── core/
│       ├── env.py             # Multi-agent environment (reward conservation enforced)
│       ├── graph_world.py     # Graph creation, territory assignment, BFS utilities
│       └── agent_state.py     # Per-agent mutable state (position, prices, stats)
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx            # Root layout: resizable sidebar + graph canvas + bottom panel
│   │   ├── styles/tokens.css  # Design tokens (colors, spacing, typography)
│   │   ├── components/        # Graph renderer, panels, charts, shared UI
│   │   ├── stores/            # Zustand state (training, replay, graph, config, UI)
│   │   ├── hooks/             # useWebSocket, usePlayback, useSimulationPlayback
│   │   └── lib/               # Chart registry (26 charts), CSV export, chart theme
│   └── vite.config.ts         # Dev proxy: /api → :8000, /ws → :8000
│
├── episodes/                  # Saved episode JSON files (auto-created per session)
└── logs/                      # Training logs
```

---

## Web UI Guide

### Sidebar (left panel — drag right edge to resize)
- **Build Tools** — place nodes, connect edges, assign ownership, mark destinations
- **Graph Generator** — configure nodes/agents/destinations, click Generate Random
- **Start Simulation** — loads the trained model and runs inference-only episodes
- **Simulation Config** — trip reward, max price, animation speed
- **Display Toggles** — show/hide node IDs, prices, destinations, agents
- **Agent Palette** — customize per-agent colors via the color picker

### Graph Canvas (center)
- **Drag** nodes to rearrange layout
- **Scroll** to zoom, **drag background** to pan
- **Normalize** button re-arranges nodes into a clean circle

### Bottom Panel (drag handle to resize — 15% to 85% of window height)

**Live Status tab** — real-time per-agent stats during simulation:
- Net reward, trips completed, destination revenue, tax revenue/paid
- Live node prices with ownership colors

**Analysis & Replay tab** — post-simulation analysis:
- Scrub through every step with the replay slider
- **26 charts** across 4 categories: Agent Performance, Node Economics, System Overview, Temporal
- **Click any chart** to expand it fullscreen within the panel
- **Export CSV** — download all chart data as a ZIP

---

## How It Works

### The Game

- **Graph**: N nodes connected by edges. Agents live on nodes and traverse edges each step.
- **Destinations**: Each agent has assigned destination nodes. Completing a trip between two different destinations earns `trip_reward`.
- **Territory**: Every node is owned by exactly one agent. Owners set a price; anyone who steps on that node pays the toll to the owner.
- **Pricing**: Each step, agents use a Dirichlet-based pricing strategy — allocating a fixed budget across their owned nodes.
- **Conservation law**: Taxes are pure transfers (zero-sum). Trip rewards are the *only* money injected. `total_reward = trips_completed × trip_reward` — always.

### Neural Network — GATv2 GNN

All agents share a single **GraphopolyGNN** (~23,500 parameters). Each agent sees the graph from its own perspective with 12 node features encoding position, ownership, prices, destinations, congestion, and economic context.

**Architecture:**
- 2× GATv2Conv layers (4 attention heads, 64-dim embeddings)
- Movement head: scores candidate moves
- Pricing head: Dirichlet-based budget allocation across owned nodes
- Value head: estimates future expected reward

**Why GATv2?** Dynamic attention — the score between nodes i and j depends on both embeddings simultaneously, making it more expressive than GAT for economic reasoning.

### Learning — PPO

Training runs offline. Each episode:
1. A random graph is drawn from the training pool
2. Agents collect a rollout (forward pass → sample actions → observe rewards)
3. **GAE** computes advantages (γ=0.99, λ=0.95)
4. **PPO** runs 4 update epochs with clipped surrogate loss
5. Shared Adam optimizer — prevents conflicting momentum across agents
6. Entropy annealing: coefficient decays 0.01 → 0.001 over first half of training

---

## Config Reference (`backend/config.py`)

| Category | Key Parameters |
|----------|---------------|
| Graph | `num_nodes` (default 10), `num_edges` (None = random) |
| Agents | `num_agents` (5), `num_destinations` (2), `trip_reward` (10.0), `max_price` (20) |
| Training | `steps_per_episode` (100), `lr` (3e-4), `gamma` (0.99), `ppo_epochs` (4) |
| GNN | `hidden_dim` (64), `num_gnn_layers` (2), `gat_heads` (4) |
| Device | Auto: MPS → CUDA → CPU |

---

## Research Questions

- Do agents prioritize **commuting** (more trips) or **toll extraction** (high prices)?
- Does **bottleneck monopolization** emerge — agents pricing key bridge nodes to extract from all routes?
- Do agents set **strategic pricing** to reroute opponents away from their destinations?
- Does the system converge to a **Nash equilibrium**, or do strategies cycle indefinitely?
- How does **graph topology** affect equilibrium — hub-and-spoke vs grid vs random?

---

## Dependencies

```
torch>=2.0
torch_geometric>=2.4     # GATv2Conv
networkx>=3.0            # Graph algorithms (BFS, diameter, layout)
numpy>=1.24
fastapi>=0.100
uvicorn[standard]>=0.23
pydantic>=2.0
websockets>=11.0
```
