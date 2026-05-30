"""
FastAPI backend for Graphopoly web UI.

Serves the static frontend and provides:
- REST endpoints for graph building, saving/loading, and config
- WebSocket for real-time training updates
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import GraphopolyConfig
from backend.core.graph_world import GraphWorld
from backend.train import train
from backend.simulate import simulate as run_simulation
from backend.analyze import EpisodeAnalyzer
from backend.db_export import export_episode_to_zip

# Project root (parent of backend/)
PROJECT_ROOT = Path(__file__).parent.parent
EPISODES_DIR = PROJECT_ROOT / "episodes"
GRAPHS_DIR   = PROJECT_ROOT / "graphs"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# ------------------------------------------------------------------
# State
# ------------------------------------------------------------------

_state = {
    "config": GraphopolyConfig(),
    "world": None,
    "training": False,
    "stop_event": threading.Event(),
    "pause_event": threading.Event(),
    "websockets": set(),
    "train_thread": None,
    "event_loop": None,
    "custom_layout": None,  # User-arranged layout positions (persisted across training)
    "run_id": None,         # Current run's unique identifier
    "run_name": "",         # Human-readable name for current run
    "run_mode": None,       # "train" or "simulate"
    "run_started_at": None, # ISO timestamp of run start
}
_state["pause_event"].set()


# ------------------------------------------------------------------
# Lifespan
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["event_loop"] = asyncio.get_event_loop()
    EPISODES_DIR.mkdir(exist_ok=True)
    GRAPHS_DIR.mkdir(exist_ok=True)
    yield


app = FastAPI(title="Graphopoly", lifespan=lifespan)

# CORS for Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------------

class GraphBuildRequest(BaseModel):
    num_nodes: int
    edges: list[list[int]]
    ownership: dict[str, int] = {}
    destinations: dict[str, list[int]] = {}
    starting_positions: dict[str, int] = {}


class RandomGraphRequest(BaseModel):
    num_nodes: int = 8
    num_edges: int | None = None  # None = auto
    num_agents: int = 2
    num_destinations: int = 2


class ConfigUpdate(BaseModel):
    agent: dict = {}
    train: dict = {}
    network: dict = {}
    log: dict = {}


class StartRequest(BaseModel):
    run_name: str = ""


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _check_training_lock():
    """Return a 400 JSONResponse if training/simulation is active, else None."""
    if _state["training"]:
        return JSONResponse(
            {"status": "error", "message": "Cannot modify graph while simulation is running. Stop it first."},
            status_code=400,
        )
    return None


def _serialize_layout(layout: dict) -> dict:
    """Ensure layout keys are strings for JSON serialization and values are lists."""
    return {str(k): [float(v[0]), float(v[1])] for k, v in layout.items()}


def _make_run_id(run_name: str) -> str:
    """Generate a unique, filesystem-safe run identifier."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name:
        safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", run_name.strip())[:40]
        return f"{ts}_{safe}"
    return ts


def _write_run_meta(run_id: str, final_metrics: dict) -> None:
    """Write a small sidecar .meta.json so experiments can be listed cheaply."""
    config_snap = final_metrics.get("config_snapshot", {})
    agent_cfg = config_snap.get("agent", {})
    meta = {
        "run_id":       run_id,
        "run_name":     _state.get("run_name", ""),
        "mode":         _state.get("run_mode", "simulate"),
        "started_at":   _state.get("run_started_at", ""),
        "finished_at":  datetime.now().isoformat(),
        "num_episodes": final_metrics.get("episode", 0),
        "num_agents":   agent_cfg.get("num_agents", 0),
        "num_nodes":    len(final_metrics.get("graph_data", {}).get("nodes", [])),
        "final_rewards": final_metrics.get("episode_rewards", []),
        "final_trips":   final_metrics.get("episode_trips", []),
    }
    path = EPISODES_DIR / f"{run_id}.meta.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


# ------------------------------------------------------------------
# WebSocket broadcast
# ------------------------------------------------------------------

async def broadcast(message: dict) -> None:
    dead = set()
    for ws in _state["websockets"]:
        try:
            await ws.send_json(message)
        except Exception:
            dead.add(ws)
    _state["websockets"] -= dead


def sync_broadcast(message: dict) -> None:
    loop = _state.get("event_loop")
    if loop is not None and loop.is_running():
        asyncio.run_coroutine_threadsafe(broadcast(message), loop)


# ------------------------------------------------------------------
# REST endpoints
# ------------------------------------------------------------------

@app.get("/")
async def index():
    return RedirectResponse(url="http://localhost:5173", status_code=302)


@app.post("/api/graph/build")
async def build_graph(req: GraphBuildRequest):
    lock_resp = _check_training_lock()
    if lock_resp:
        return lock_resp

    if req.num_nodes > 50:
        return JSONResponse(
            {"status": "error", "message": "Maximum 50 nodes allowed."},
            status_code=400,
        )

    edges = [tuple(e) for e in req.edges]
    world = GraphWorld.from_custom(edges, req.num_nodes)

    ownership = {int(k): v for k, v in req.ownership.items()}
    destinations = {int(k): v for k, v in req.destinations.items()}
    starting_positions = {int(k): v for k, v in req.starting_positions.items()}

    if ownership:
        world.set_ownership(ownership)
    if destinations:
        world.set_destinations(destinations)

    config = _state["config"]

    # Auto-detect agent count from the graph data itself.
    # This ensures training uses the right number of agents for custom graphs.
    max_owner = (max(ownership.values()) + 1) if ownership else 0
    max_dest_key = (max(destinations.keys()) + 1) if destinations else 0
    auto_num_agents = max(max_owner, max_dest_key, config.agent.num_agents, 1)
    if auto_num_agents > 10:
        return JSONResponse(
            {"status": "error", "message": "Maximum 10 agents allowed."},
            status_code=400,
        )
    config.agent.num_agents = auto_num_agents

    # Auto-assign starting positions if not provided
    if starting_positions:
        world.set_starting_positions(starting_positions)
    else:
        rng = np.random.default_rng()
        world.assign_starting_positions(auto_num_agents, rng)

    # Validate the graph
    try:
        world.validate(
            auto_num_agents,
            min_destinations=config.agent.num_destinations,
            trip_reward=config.agent.trip_reward,
            price_budget=config.agent.price_budget,
        )
    except ValueError as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=400)

    _state["world"] = world
    _state["config"] = config
    layout = _serialize_layout(world.get_spring_layout())
    _state["custom_layout"] = layout

    return {
        "status": "ok",
        "graph": world.to_dict(),
        "layout": layout,
    }


@app.post("/api/graph/random")
async def random_graph(req: RandomGraphRequest):
    lock_resp = _check_training_lock()
    if lock_resp:
        return lock_resp

    config = _state["config"]
    config.agent.num_agents = req.num_agents
    config.agent.num_destinations = req.num_destinations

    # Validate node and agent counts
    if req.num_nodes < 2:
        return JSONResponse(
            {"status": "error", "message": "Need at least 2 nodes."},
            status_code=400,
        )
    if req.num_nodes > 50:
        return JSONResponse(
            {"status": "error", "message": "Maximum 50 nodes allowed."},
            status_code=400,
        )
    if req.num_agents > 10:
        return JSONResponse(
            {"status": "error", "message": "Maximum 10 agents allowed."},
            status_code=400,
        )
    if req.num_nodes < req.num_destinations:
        return JSONResponse(
            {"status": "error", "message": f"Need at least {req.num_destinations} nodes for {req.num_destinations} destinations per agent."},
            status_code=400,
        )

    # Validate edge count
    max_edges = req.num_nodes * (req.num_nodes - 1) // 2
    min_edges = req.num_nodes - 1
    num_edges = req.num_edges
    if num_edges is not None:
        if num_edges < min_edges:
            return JSONResponse(
                {"status": "error", "message": f"Need at least {min_edges} edges for {req.num_nodes} nodes."},
                status_code=400,
            )
        if num_edges > max_edges:
            return JSONResponse(
                {"status": "error", "message": f"Max {max_edges} edges for {req.num_nodes} nodes."},
                status_code=400,
            )

    # Use a fresh random seed every time so graph is different on each call
    seed = random.randint(0, 2**31 - 1)
    rng = np.random.default_rng(seed)

    try:
        world = GraphWorld.random_connected(req.num_nodes, num_edges, rng)
        world.assign_territories(req.num_agents, rng)
        world.assign_destinations(req.num_agents, req.num_destinations, rng)
        world.assign_starting_positions(req.num_agents, rng)

        # Validate
        world.validate(
            req.num_agents,
            min_destinations=req.num_destinations,
            trip_reward=config.agent.trip_reward,
            price_budget=config.agent.price_budget,
        )
    except ValueError as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=400)

    _state["world"] = world
    _state["config"] = config
    layout = _serialize_layout(world.get_circular_layout())
    _state["custom_layout"] = layout

    return {
        "status": "ok",
        "graph": world.to_dict(),
        "layout": layout,
    }


class SyncLayoutRequest(BaseModel):
    layout: dict[str, list[float]]


@app.post("/api/graph/sync-layout")
async def sync_layout(req: SyncLayoutRequest):
    """Store the user's node layout positions so training uses them."""
    _state["custom_layout"] = {str(k): [float(v[0]), float(v[1])] for k, v in req.layout.items()}
    return {"status": "ok"}


# ------------------------------------------------------------------
# Config endpoints
# ------------------------------------------------------------------

@app.get("/api/config")
async def get_config():
    return _state["config"].to_dict()


@app.post("/api/config")
async def update_config(req: ConfigUpdate):
    config = _state["config"]
    for key, val in req.agent.items():
        if hasattr(config.agent, key):
            setattr(config.agent, key, val)
    for key, val in req.train.items():
        if hasattr(config.train, key):
            setattr(config.train, key, val)
    for key, val in req.network.items():
        if hasattr(config.network, key):
            setattr(config.network, key, val)
    for key, val in req.log.items():
        if hasattr(config.log, key):
            setattr(config.log, key, val)
    return {"status": "ok", "config": config.to_dict()}


# ------------------------------------------------------------------
# Training endpoints
# ------------------------------------------------------------------

@app.post("/api/train/start")
async def start_training(req: StartRequest = StartRequest()):
    if _state["training"]:
        return {"status": "error", "message": "Training already running"}
    if _state["world"] is None:
        return {"status": "error", "message": "No graph loaded. Build or generate one first."}

    run_id = _make_run_id(req.run_name)
    _state["stop_event"].clear()
    _state["pause_event"].set()
    _state["training"] = True
    _state["run_id"] = run_id
    _state["run_name"] = req.run_name or run_id
    _state["run_mode"] = "train"
    _state["run_started_at"] = datetime.now().isoformat()

    config = _state["config"]
    world = _state["world"]

    def run_training():
        last_metrics: dict = {}
        try:
            EPISODES_DIR.mkdir(exist_ok=True)

            def on_episode(metrics: dict):
                nonlocal last_metrics
                if _state["custom_layout"]:
                    metrics["layout"] = _state["custom_layout"]
                elif "layout" in metrics:
                    metrics["layout"] = _serialize_layout(metrics["layout"])
                sync_broadcast({"type": "episode_update", "data": metrics})
                last_metrics = metrics
                try:
                    with open(EPISODES_DIR / "temp_latest.json", "w") as _f:
                        json.dump(metrics, _f)
                    with open(EPISODES_DIR / f"{run_id}.json", "w") as _f:
                        json.dump(metrics, _f)
                except Exception:
                    pass

            result = train(
                config, world,
                callback=on_episode,
                stop_event=_state["stop_event"],
                pause_event=_state["pause_event"],
            )

            if last_metrics:
                _write_run_meta(run_id, last_metrics)

            if result.get("stopped_early"):
                sync_broadcast({"type": "training_stopped", "data": {**result, "run_id": run_id}})
            else:
                sync_broadcast({"type": "training_complete", "data": {**result, "run_id": run_id}})
        except Exception as e:
            import traceback
            sync_broadcast({"type": "training_error", "data": {"error": str(e), "trace": traceback.format_exc()}})
        finally:
            _state["training"] = False

    thread = threading.Thread(target=run_training, daemon=True)
    _state["train_thread"] = thread
    thread.start()

    return {"status": "ok", "message": "Training started", "run_id": run_id}


@app.post("/api/train/stop")
async def stop_training():
    _state["stop_event"].set()
    _state["pause_event"].set()  # unblock if paused so it can exit
    return {"status": "ok"}


@app.post("/api/train/pause")
async def pause_training():
    _state["pause_event"].clear()
    return {"status": "ok", "paused": True}


@app.post("/api/train/resume")
async def resume_training():
    _state["pause_event"].set()
    return {"status": "ok", "paused": False}


# ------------------------------------------------------------------
# Simulation (inference-only) endpoints
# ------------------------------------------------------------------

@app.post("/api/simulate/start")
async def start_simulation(req: StartRequest = StartRequest()):
    """Load the model for the current graph's size and run inference-only episodes."""
    if _state["training"]:
        return {"status": "error", "message": "Already running"}
    if _state["world"] is None:
        return {"status": "error", "message": "No graph loaded. Build or generate one first."}

    run_id = _make_run_id(req.run_name)
    _state["stop_event"].clear()
    _state["pause_event"].set()
    _state["training"] = True
    _state["run_id"] = run_id
    _state["run_name"] = req.run_name or run_id
    _state["run_mode"] = "simulate"
    _state["run_started_at"] = datetime.now().isoformat()

    config = _state["config"]
    world = _state["world"]

    def run():
        last_metrics: dict = {}
        try:
            EPISODES_DIR.mkdir(exist_ok=True)

            def on_episode(metrics: dict):
                nonlocal last_metrics
                if _state["custom_layout"]:
                    metrics["layout"] = _state["custom_layout"]
                elif "layout" in metrics:
                    metrics["layout"] = _serialize_layout(metrics["layout"])
                sync_broadcast({"type": "episode_update", "data": metrics})
                last_metrics = metrics
                try:
                    with open(EPISODES_DIR / "temp_latest.json", "w") as _f:
                        json.dump(metrics, _f)
                    with open(EPISODES_DIR / f"{run_id}.json", "w") as _f:
                        json.dump(metrics, _f)
                except Exception:
                    pass

            result = run_simulation(
                config, world,
                callback=on_episode,
                stop_event=_state["stop_event"],
                pause_event=_state["pause_event"],
            )

            if last_metrics:
                _write_run_meta(run_id, last_metrics)

            if result.get("stopped_early"):
                sync_broadcast({"type": "training_stopped", "data": {**result, "run_id": run_id}})
            else:
                sync_broadcast({"type": "training_complete", "data": {**result, "run_id": run_id}})
        except Exception as e:
            import traceback
            sync_broadcast({"type": "training_error", "data": {"error": str(e), "trace": traceback.format_exc()}})
        finally:
            _state["training"] = False

    thread = threading.Thread(target=run, daemon=True)
    _state["train_thread"] = thread
    thread.start()

    return {"status": "ok", "message": "Simulation started", "run_id": run_id}


# ------------------------------------------------------------------
# Episode endpoints
# ------------------------------------------------------------------

@app.post("/api/episode/save")
async def save_episode(body: dict):
    """Save a full episode (graph + step history) to episodes/ directory."""
    EPISODES_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ep = body.get("episode", 0)
    custom_name = body.pop("custom_name", None)
    if custom_name:
        safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", custom_name)[:64]
        filename = f"{safe}.json"
    else:
        filename = f"episode_{ts}_ep{ep:05d}.json"
    path = EPISODES_DIR / filename
    with open(path, "w") as f:
        json.dump(body, f, indent=2)
    return {"status": "ok", "path": str(path), "filename": filename}


@app.post("/api/episode/load")
async def load_episode(file: UploadFile = File(...)):
    """Upload and parse an episode JSON file. Returns graph + step history."""
    content = await file.read()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return JSONResponse({"status": "error", "message": "Invalid JSON"}, 400)
    return {"status": "ok", "data": data}


@app.get("/api/episode/latest")
async def latest_episode():
    """Return the most recently auto-saved temp episode."""
    path = EPISODES_DIR / "temp_latest.json"
    if not path.exists():
        return JSONResponse({"status": "error", "message": "No episode saved yet."}, 404)
    with open(path) as f:
        data = json.load(f)
    return {"status": "ok", "data": data}


@app.get("/api/episode/list")
async def list_episodes():
    """List saved episode files."""
    EPISODES_DIR.mkdir(exist_ok=True)
    files = sorted(EPISODES_DIR.glob("*.json"), reverse=True)
    return {"files": [f.name for f in files[:50]]}


# ------------------------------------------------------------------
# Analyze Mode
# ------------------------------------------------------------------

@app.post("/api/analyze/compute")
async def compute_analysis(data: dict):
    """
    Accept episode JSON, compute all metrics timeseries for visualization.
    """
    try:
        analyzer = EpisodeAnalyzer()
        analyzer.load_episode(data)
        timeline = analyzer.compute_metrics_timeline()

        return {
            "status": "ok",
            "timeline": timeline,
            "graph_data": analyzer.graph_data,
            "config": analyzer.config,
            "num_steps": analyzer.num_steps,
            "num_agents": analyzer.num_agents,
            "num_nodes": analyzer.num_nodes,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


# ------------------------------------------------------------------
# Data Export
# ------------------------------------------------------------------

@app.get("/api/export/data")
async def export_data():
    """Export the latest episode as a ZIP (SQLite dump + CSVs + raw JSON)."""
    path = EPISODES_DIR / "temp_latest.json"
    if not path.exists():
        return JSONResponse({"status": "error", "message": "No episode data to export yet."}, 404)
    with open(path) as f:
        data = json.load(f)
    try:
        zip_bytes = export_episode_to_zip(data)
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"Export failed: {e}"}, 500)
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=graphopoly_data.zip"},
    )


# ------------------------------------------------------------------
# Experiments (named run history)
# ------------------------------------------------------------------

@app.get("/api/experiments")
async def list_experiments():
    """List all completed runs by reading their .meta.json sidecars."""
    EPISODES_DIR.mkdir(exist_ok=True)
    metas = []
    for meta_path in sorted(EPISODES_DIR.glob("*.meta.json"), reverse=True):
        try:
            with open(meta_path) as f:
                metas.append(json.load(f))
        except Exception:
            pass
    return {"experiments": metas}


@app.get("/api/experiments/{run_id}")
async def load_experiment(run_id: str):
    """Load the full episode data for a saved run."""
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "", run_id)
    path = EPISODES_DIR / f"{safe}.json"
    if not path.exists():
        return JSONResponse({"status": "error", "message": "Run not found."}, 404)
    with open(path) as f:
        data = json.load(f)
    return {"status": "ok", "data": data}


@app.delete("/api/experiments/{run_id}")
async def delete_experiment(run_id: str):
    """Delete a saved run (both episode JSON and meta sidecar)."""
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "", run_id)
    deleted = []
    for suffix in [".json", ".meta.json"]:
        p = EPISODES_DIR / f"{safe}{suffix}"
        if p.exists():
            p.unlink()
            deleted.append(p.name)
    return {"status": "ok", "deleted": deleted}


# ------------------------------------------------------------------
# Graph Library (save / load named graph configurations)
# ------------------------------------------------------------------

class SaveGraphRequest(BaseModel):
    name: str
    graph: dict   # GraphBuildRequest-compatible fields
    layout: dict  # {node_id: [x, y]}


@app.post("/api/graphs/save")
async def save_graph_config(req: SaveGraphRequest):
    """Persist the current graph topology + layout under a user-chosen name."""
    GRAPHS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", req.name.strip())[:48]
    graph_id = f"{ts}_{safe_name}" if safe_name else ts
    doc = {
        "graph_id":  graph_id,
        "name":      req.name,
        "saved_at":  datetime.now().isoformat(),
        "num_nodes": req.graph.get("num_nodes", 0),
        "num_agents": max(
            (max(req.graph.get("ownership", {}).values(), default=-1) + 1),
            len(req.graph.get("destinations", {})),
            1,
        ),
        "graph":  req.graph,
        "layout": req.layout,
    }
    with open(GRAPHS_DIR / f"{graph_id}.json", "w") as f:
        json.dump(doc, f, indent=2)
    return {"status": "ok", "graph_id": graph_id}


@app.get("/api/graphs")
async def list_graphs():
    """List all saved graph configurations (metadata only)."""
    GRAPHS_DIR.mkdir(exist_ok=True)
    graphs = []
    for p in sorted(GRAPHS_DIR.glob("*.json"), reverse=True):
        try:
            with open(p) as f:
                doc = json.load(f)
            graphs.append({
                "graph_id":  doc.get("graph_id", p.stem),
                "name":      doc.get("name", p.stem),
                "saved_at":  doc.get("saved_at", ""),
                "num_nodes": doc.get("num_nodes", 0),
                "num_agents": doc.get("num_agents", 0),
            })
        except Exception:
            pass
    return {"graphs": graphs}


@app.get("/api/graphs/{graph_id}")
async def load_graph_config(graph_id: str):
    """Return the full saved graph (topology + layout) by ID."""
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "", graph_id)
    path = GRAPHS_DIR / f"{safe}.json"
    if not path.exists():
        return JSONResponse({"status": "error", "message": "Graph not found."}, 404)
    with open(path) as f:
        doc = json.load(f)
    return {"status": "ok", "data": doc}


@app.delete("/api/graphs/{graph_id}")
async def delete_graph_config(graph_id: str):
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "", graph_id)
    path = GRAPHS_DIR / f"{safe}.json"
    if path.exists():
        path.unlink()
    return {"status": "ok"}


# ------------------------------------------------------------------
# Status
# ------------------------------------------------------------------

@app.get("/api/status")
async def get_status():
    return {
        "training":  _state["training"],
        "paused":    not _state["pause_event"].is_set(),
        "has_graph": _state["world"] is not None,
        "run_id":    _state.get("run_id"),
        "run_name":  _state.get("run_name", ""),
        "run_mode":  _state.get("run_mode"),
    }


# ------------------------------------------------------------------
# WebSocket
# ------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _state["websockets"].add(ws)
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        _state["websockets"].discard(ws)



def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the backend server (called from main.py)."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
