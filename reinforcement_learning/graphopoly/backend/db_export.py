"""
SQLite + CSV export for Graphopoly episode data.

Takes a loaded episode JSON dict and produces a ZIP file containing:
  - data.sql         (SQLite dump as SQL text — importable)
  - episode_info.csv
  - trajectory.csv   (per-step data)
  - training_rewards.csv
  - training_losses.csv
  - agent_stats.csv
  - node_stats.csv
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
import zipfile
from typing import Any


def _csv_bytes(rows: list[list[Any]], headers: list[str]) -> bytes:
    """Write rows + header to CSV bytes."""
    bio = io.StringIO()
    writer = csv.writer(bio)
    writer.writerow(headers)
    writer.writerows(rows)
    return bio.getvalue().encode("utf-8")


def _populate_db(conn: sqlite3.Connection, ep: dict) -> None:
    """Create tables and insert episode data into an in-memory SQLite db."""
    cur = conn.cursor()

    # ── episode_info ──────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE episode_info (
            episode_id TEXT,
            timestamp TEXT,
            finished_at TEXT,
            num_steps INTEGER,
            num_episodes INTEGER,
            num_agents INTEGER,
            num_nodes INTEGER,
            description TEXT
        )
    """)
    meta = ep.get("metadata", {})
    cur.execute("INSERT INTO episode_info VALUES (?,?,?,?,?,?,?,?)", (
        meta.get("episode_id", ""),
        meta.get("timestamp", ""),
        meta.get("finished_at", ""),
        meta.get("num_steps", 0),
        meta.get("num_episodes", 0),
        meta.get("num_agents", 0),
        meta.get("num_nodes", 0),
        meta.get("description", ""),
    ))

    # ── trajectory_steps ──────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE trajectory_steps (
            step INTEGER,
            agent_positions TEXT,
            prices TEXT,
            rewards TEXT,
            taxes TEXT,
            dest_completions TEXT,
            node_stats TEXT,
            agent_stats TEXT
        )
    """)
    for step in ep.get("trajectory", []):
        cur.execute("INSERT INTO trajectory_steps VALUES (?,?,?,?,?,?,?,?)", (
            step.get("step", 0),
            json.dumps(step.get("agent_positions", {})),
            json.dumps(step.get("prices", {})),
            json.dumps(step.get("rewards", {})),
            json.dumps(step.get("taxes", {})),
            json.dumps(step.get("dest_completions", [])),
            json.dumps(step.get("node_stats", {})),
            json.dumps(step.get("agent_stats", {})),
        ))

    # ── training_rewards ──────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE training_rewards (
            episode_num INTEGER,
            agent_id TEXT,
            reward REAL,
            trips INTEGER
        )
    """)
    tm = ep.get("training_metrics", {})
    ep_rewards = tm.get("episode_rewards", [])
    ep_trips = tm.get("episode_trips", [])
    for i, (rew_dict, trip_dict) in enumerate(zip(ep_rewards, ep_trips)):
        for aid, r in rew_dict.items():
            t = trip_dict.get(aid, 0)
            cur.execute("INSERT INTO training_rewards VALUES (?,?,?,?)", (i, aid, r, t))

    # ── training_losses ───────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE training_losses (
            episode_num INTEGER,
            policy_loss REAL,
            value_loss REAL,
            entropy REAL
        )
    """)
    losses = tm.get("losses", {})
    pol = losses.get("policy_loss", [])
    val = losses.get("value_loss", [])
    ent = losses.get("entropy_bonus", [])
    for i, (p, v, e) in enumerate(zip(pol, val, ent)):
        cur.execute("INSERT INTO training_losses VALUES (?,?,?,?)", (i, p, v, e))

    # ── agent_stats ───────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE agent_stats (
            agent_id TEXT,
            total_trips INTEGER,
            total_profit REAL,
            avg_profit_per_step REAL,
            total_tax_revenue REAL,
            total_tax_paid REAL,
            total_dest_revenue REAL,
            owned_nodes TEXT,
            destination_nodes TEXT
        )
    """)
    for aid, stats in ep.get("aggregate_stats", {}).get("agents", {}).items():
        cur.execute("INSERT INTO agent_stats VALUES (?,?,?,?,?,?,?,?,?)", (
            aid,
            stats.get("total_trips", 0),
            stats.get("total_profit", 0.0),
            stats.get("average_profit_per_step", 0.0),
            stats.get("total_tax_revenue", 0.0),
            stats.get("total_tax_paid", 0.0),
            stats.get("total_dest_revenue", 0.0),
            json.dumps(stats.get("owned_nodes", [])),
            json.dumps(stats.get("destination_nodes", [])),
        ))

    # ── node_stats ────────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE node_stats (
            node_id TEXT,
            owner INTEGER,
            total_visits INTEGER,
            avg_visits_per_step REAL,
            total_revenue_collected REAL,
            current_price REAL,
            average_price REAL
        )
    """)
    for nid, stats in ep.get("aggregate_stats", {}).get("nodes", {}).items():
        cur.execute("INSERT INTO node_stats VALUES (?,?,?,?,?,?,?)", (
            nid,
            stats.get("owner", -1),
            stats.get("total_visits", 0),
            stats.get("avg_visits_per_step", 0.0),
            stats.get("total_revenue_collected", 0.0),
            stats.get("current_price", 0),
            stats.get("average_price", 0.0),
        ))

    conn.commit()


def export_episode_to_zip(episode_data: dict) -> bytes:
    """
    Convert episode JSON data to a ZIP archive containing:
    - data.sql  (SQLite dump — SQL text, importable)
    - *.csv     (one per table)
    - episode.json (raw episode data)
    """
    conn = sqlite3.connect(":memory:")
    _populate_db(conn, episode_data)

    # ── SQL dump ─────────────────────────────────────────────────────────────
    sql_lines = list(conn.iterdump())
    sql_text = "\n".join(sql_lines).encode("utf-8")

    # ── CSV exports ──────────────────────────────────────────────────────────
    def table_csv(table: str, headers: list[str]) -> bytes:
        rows = conn.execute(f"SELECT * FROM {table}").fetchall()
        return _csv_bytes([list(r) for r in rows], headers)

    ep_csv = table_csv("episode_info", [
        "episode_id", "timestamp", "finished_at", "num_steps",
        "num_episodes", "num_agents", "num_nodes", "description"
    ])
    traj_csv = table_csv("trajectory_steps", [
        "step", "agent_positions", "prices", "rewards",
        "taxes", "dest_completions", "node_stats", "agent_stats"
    ])
    rewards_csv = table_csv("training_rewards", ["episode_num", "agent_id", "reward", "trips"])
    losses_csv  = table_csv("training_losses", ["episode_num", "policy_loss", "value_loss", "entropy"])
    agent_csv   = table_csv("agent_stats", [
        "agent_id", "total_trips", "total_profit", "avg_profit_per_step",
        "total_tax_revenue", "total_tax_paid", "total_dest_revenue",
        "owned_nodes", "destination_nodes"
    ])
    node_csv = table_csv("node_stats", [
        "node_id", "owner", "total_visits", "avg_visits_per_step",
        "total_revenue_collected", "current_price", "average_price"
    ])

    conn.close()

    # ── Assemble ZIP ─────────────────────────────────────────────────────────
    zip_bio = io.BytesIO()
    with zipfile.ZipFile(zip_bio, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("data.sql",                sql_text)
        zf.writestr("episode_info.csv",        ep_csv)
        zf.writestr("trajectory.csv",          traj_csv)
        zf.writestr("training_rewards.csv",    rewards_csv)
        zf.writestr("training_losses.csv",     losses_csv)
        zf.writestr("agent_stats.csv",         agent_csv)
        zf.writestr("node_stats.csv",          node_csv)
        zf.writestr("episode.json",            json.dumps(episode_data, indent=2).encode("utf-8"))

    return zip_bio.getvalue()
