# Graphopoly — Claude Context

This file stores key facts, design decisions, and architectural notes for Claude to reference across sessions and devices.

---

## How to Run

```bash
python3 main.py            # Starts both backend (port 8000) + Vite frontend (port 5173)
python3 main.py --backend-only   # Backend only
```

Open **http://localhost:5173** in the browser. Hard refresh (Cmd+Shift+R) if UI looks stale.

---

## Project Layout

```
graphopoly/
├── main.py                    # Single entry point — starts FastAPI + Vite together
├── CLAUDE.md                  # This file
├── README.md
├── requirements.txt
├── backend/
│   ├── server.py              # FastAPI app, WebSocket /ws, REST /api/*
│   ├── config.py              # All hyperparameters
│   ├── simulate.py            # Inference-only simulation (used by web UI)
│   ├── train.py               # Single-graph training
│   ├── train_offline.py       # CLI: offline training with graph pools
│   ├── analyze.py             # Post-episode analysis (called by /api/analyze)
│   ├── agent/
│   │   ├── gnn_network.py     # GraphopolyGNN — GATv2 shared policy + value net
│   │   └── ppo.py             # PPO trainer
│   └── core/
│       ├── env.py             # Multi-agent environment
│       ├── graph_world.py     # Graph creation & BFS utilities
│       └── agent_state.py     # Per-agent mutable state
├── models/
│   └── latest_model.pt        # Current best model (used by simulate.py)
├── frontend/
│   ├── src/
│   │   ├── App.tsx            # Root layout: sidebar + graph canvas + bottom panel
│   │   ├── styles/
│   │   │   ├── tokens.css     # ALL CSS custom properties (colors, spacing, type)
│   │   │   ├── globals.css    # Body, typography classes, animations
│   │   │   └── shared.css     # .btn, .input, .toggle, .range-slider
│   │   ├── components/
│   │   │   ├── layout/
│   │   │   │   ├── AppShell.tsx
│   │   │   │   └── Header.tsx
│   │   │   ├── graph/
│   │   │   │   ├── GraphCanvas.tsx   # SVG defs (gradients, filters), zoom/pan, mode clicks
│   │   │   │   └── GraphRenderer.tsx # Node/edge/agent SVG rendering
│   │   │   ├── panels/
│   │   │   │   ├── SettingsPanel.tsx # Sidebar: build tools, graph gen, simulation controls
│   │   │   │   ├── LiveStatsPanel.tsx
│   │   │   │   └── AnalysisReplayPanel.tsx
│   │   │   ├── charts/
│   │   │   │   ├── ChartDisplay.tsx  # Grid of charts; click to expand single chart
│   │   │   │   ├── ChartNavigator.tsx
│   │   │   │   └── ChartWrapper.tsx
│   │   │   ├── shared/
│   │   │   │   ├── Accordion.tsx
│   │   │   │   ├── Button.tsx
│   │   │   │   ├── ColorPicker.tsx   # Custom HSV picker — popup uses position:fixed
│   │   │   │   └── ...
│   │   │   └── onboarding/
│   │   │       └── OnboardingOverlay.tsx  # Tour — localStorage key: graphopoly_onboarded_v3
│   │   ├── stores/            # Zustand: uiStore, trainingStore, graphStore, configStore, replayStore, analyzeStore
│   │   ├── hooks/             # useWebSocket, usePlayback, useSimulationPlayback, useKeyboard
│   │   └── lib/
│   │       ├── chartRegistry.ts  # 26 charts across 4 categories
│   │       └── chartTheme.ts     # AGENT_COLORS canonical source
│   └── vite.config.ts         # Proxy: /api → http://localhost:8000, /ws → ws://localhost:8000
```

---

## Critical Bug Fixes (Do Not Revert)

### MPS Dirichlet Crash (`backend/agent/gnn_network.py`)
`aten::_sample_dirichlet` is not implemented on Apple Silicon MPS. Fixed by sampling on CPU:
```python
concentration_cpu = concentration.cpu()
dist = Dirichlet(concentration_cpu)
weights_cpu = dist.rsample()
weights = weights_cpu.to(price_scores.device)
log_prob = dist.log_prob(weights_cpu)
entropy = dist.entropy()
```

### ColorPicker Popup Clipping (`frontend/src/components/shared/ColorPicker.tsx`)
The sidebar has `overflow-y: auto` which clips `position: absolute` popups. Fix: use `position: fixed` with `getBoundingClientRect()` to compute viewport-relative position.

---

## Design System — tokens.css

**Current theme: Industrial Gray (charcoal, no glows/gradients)**
- `--color-bg: #0b0c0d` — near-black canvas
- `--color-bg-elevated: #111214` — sidebar/header
- `--color-bg-surface: #161719` — bottom panel
- `--color-accent: #0066FF` — action blue (clean, no glow)
- `--color-border: rgba(255,255,255,0.07)` — subtle 1px panel separators
- **No radial gradients on canvas** — flat `var(--color-bg)`
- **No box-shadow glows** — shadows are pure drop shadows only

Font sizes (increased from original for readability):
- `--text-xs: 12px`, `--text-sm: 13px`, `--text-base: 14px`, `--text-md: 15px`, `--text-lg: 17px`

---

## Layout Architecture (App.tsx)

```
AppShell (fixed header 56px, body below)
  ├── <aside>  — resizable sidebar (drag right edge, 180–600px)
  │             collapsed width: 56px (toggleSidebar in uiStore)
  │             SettingsPanel inside
  └── <main>   — flex column
        ├── <section flex:1>  — GraphCanvas (fills remaining height)
        ├── <div 10px>        — bottom drag handle (ns-resize)
        └── <section height:42vh>  — bottom panel (Live Stats / Analysis)
```

- Sidebar width: controlled by `sidebarWidth` state in App.tsx (default 300px)
- Bottom panel height: controlled by `bottomPanelHeight` state (default 42%, range 15–85%)
- Both are drag-resizable

---

## Graph Visualization (GraphCanvas + GraphRenderer)

**SVG `<defs>` in GraphCanvas.tsx:**
- `dotgrid` pattern — subtle dot background
- `nodeShadow` filter — feDropShadow
- `agentGlow` filter — feGaussianBlur + feMerge
- `nodeGradientDefault` — unowned node radial gradient
- `nodeGradient-{i}` — per-agent radial gradient (3D lighting at cx=35% cy=35%)
- `vignette` — radial gradient overlay (transparent center → dark edges)

**Node rendering:** gradient fill + outer rim circle + inner highlight + hover dashed ring
**Edge rendering:** 1px 12% opacity, hover 2px 35%, round linecaps
**Agent dots:** 11px radius, agentGlow filter, agent-pulse animation

---

## Chart System

- 26 charts across 4 categories: Agent Performance, Node Economics, System Overview, Temporal
- `frontend/src/lib/chartRegistry.ts` — all chart definitions + `chartsByCategory()`
- `ChartDisplay.tsx` — grid view (`minmax(480px, 1fr)`); click any card to expand fullscreen
- `ChartNavigator.tsx` — category pills + agent/node filter toggles

---

## State Management (Zustand stores)

| Store | Key state |
|-------|-----------|
| `uiStore` | mode, nodeSize, agentColors, isSidebarCollapsed, toggleSidebar |
| `trainingStore` | isTraining, isPaused, agentDetails, stepHistory, simAnimStep, currentPrices |
| `graphStore` | data (GraphData), layout (node positions) |
| `configStore` | config (from /api/config) |
| `replayStore` | episodeData, currentStep, totalSteps, isPlaying |
| `analyzeStore` | activeCategory, episodeData, timeline, selectedAgents, selectedNodes |

**Agent colors canonical source:** `frontend/src/lib/chartTheme.ts` → `AGENT_COLORS`
`uiStore.DEFAULT_AGENT_COLORS` imports from chartTheme.

---

## WebSocket Protocol

- Backend sends JSON frames over `/ws`
- Message types: `graph_update`, `step_update`, `training_complete`, `error`
- `useWebSocket` hook manages connection + reconnect logic
- `useSimulationPlayback` drives animated step replays

---

## API Endpoints (FastAPI)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/config` | GET | Returns current config |
| `/api/graph/generate` | POST | Generate random graph |
| `/api/graph/sync-layout` | POST | Sync node positions from frontend |
| `/api/train/start` | POST | Start simulation |
| `/api/train/pause` | POST | Pause simulation |
| `/api/train/resume` | POST | Resume simulation |
| `/api/train/stop` | POST | Stop simulation |
| `/api/episode/latest` | GET | Get latest completed episode JSON |
| `/api/analyze` | POST | Compute analysis timeline from episode |
| `/ws` | WS | Real-time step updates |

---

## Onboarding Tour

- **Trigger:** runs once, skipped after — key: `localStorage.graphopoly_onboarded_v3`
- **Reset:** click the `?` button in the header top-right
- 7 steps, SVG spotlight overlay with cutout on anchor elements

---

## Known Issues / Gotchas

1. **ALWAYS use port 5173, never port 8000** — port 8000 is the API backend only. Going to port 8000 will redirect to 5173, but the live frontend is at 5173 (Vite dev server). If you navigate to 8000 directly, you'll get redirected.
2. **Do NOT run `npm run build`** — building creates `frontend/dist/`, which used to be served by FastAPI at port 8000 (stale build). The static serving has been removed from server.py. The app runs exclusively through Vite dev server at 5173.
3. **Vite proxy only works in `npm run dev`** — the preview tool (port 5173 standalone) doesn't have the proxy, so API calls 502 in preview. This is expected; full stack works with `python3 main.py`.
4. **Hard refresh needed** if UI looks stale — browser caches Vite bundles. Use Cmd+Shift+R.
5. **MPS device** — Apple Silicon Mac Mini. Dirichlet sampling must always stay on CPU (see fix above).
6. **Linter reverts** — some CSS-in-JS inline style changes get reverted by the linter if they contain unused imports. Always verify with preview after CSS token changes.

---

## Recent Changes (reverse chronological)

### 2026-03-24 (second pass)
- Fixed stale UI issue: removed static file serving (`frontend/dist/`) from `server.py` — FastAPI at port 8000 was serving an old built bundle, causing users to see stale UI
- Port 8000 now redirects to `http://localhost:5173` (the live Vite dev server)
- `main.py` now waits for Vite to signal readiness before printing the startup URL box
- Startup output now shows a clean bordered box with the single clickable URL

### 2026-03-24
- Applied Industrial Gray theme: `#0b0c0d` backgrounds, `#0066FF` accent, no glows/gradients
- Fixed ColorPicker popup clipping with `position: fixed` + viewport-relative coordinates
- Added click-to-expand chart feature in ChartDisplay (Maximize2/Minimize2 toggle)
- Made sidebar drag-resizable (right edge drag handle, 180–600px range)
- `python3 main.py` now starts both frontend and backend by default
- Removed ASCII banner from main.py output
- Updated OnboardingOverlay to use CSS variables instead of hardcoded colors
- Removed dead code: ControlsPanel.tsx (465 lines), useMagnetic.ts hook
- Consolidated AGENT_COLORS to chartTheme.ts canonical source
- Phase 4 graph viz: SVG defs (gradients/filters), 3D node lighting, enhanced edges/agents
- Increased all font sizes (text-xs: 10→12px, text-base: 12.5→14px, etc.)
- ChartDisplay shows all charts in category as CSS grid (was single chart selector)
- Default bottom panel height: 35% → 42%, drag range: 10–80% → 15–85%
- Fixed MPS Dirichlet crash in gnn_network.py
