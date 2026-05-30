import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useGraphStore } from '../../stores/graphStore';
import { useTrainingStore } from '../../stores/trainingStore';
import { useUIStore, DEFAULT_AGENT_COLORS, UIMode } from '../../stores/uiStore';
import { useConfigStore } from '../../stores/configStore';
import { api, SavedGraphMeta } from '../../api/client';
import {
  Play, Pause, Square, Layout, Activity, Gauge, FlaskConical,
  MousePointer2, CircleDashed, Spline, User, MapPin, Trash2,
  PanelLeftClose, PanelLeftOpen, Save, FolderOpen,
} from 'lucide-react';
import { Accordion, Stepper, NumberInput, Button, Toggle, ColorPicker } from '../shared';

export const SettingsPanel: React.FC = () => {
  const loadGraph = useGraphStore(s => s.loadGraph);
  const graphData = useGraphStore(s => s.data);
  const graphLayout = useGraphStore(s => s.layout);
  const clearAll = useGraphStore(s => s.clearAll);

  const {
    isTraining, isPaused,
    pauseTraining: storeSetPaused,
    resumeTraining: storeSetResumed,
  } = useTrainingStore();
  const config = useConfigStore(s => s.config);
  const updateConfig = useConfigStore(s => s.updateConfig);
  const loadConfig = useConfigStore(s => s.loadConfig);

  const {
    mode, setMode,
    showIds, showPrices, showDests, showAgents,
    nodeSize, agentColors,
    animSpeed,
    toggleShowIds, toggleShowPrices, toggleShowDests, toggleShowAgents,
    setNodeSize, setAgentColor, resetAgentColors,
    setAnimSpeed,
    isSidebarCollapsed, toggleSidebar,
  } = useUIStore();

  const [numNodes, setNumNodes] = useState(8);
  const [numEdges, setNumEdges] = useState<number | ''>('');
  const [numAgents, setNumAgents] = useState(2);
  const [numDests, setNumDests] = useState(2);
  const [error, setError] = useState<string | null>(null);
  const [runName, setRunName] = useState('');

  // Graph library state
  const [savedGraphs, setSavedGraphs] = useState<SavedGraphMeta[]>([]);
  const [graphLibOpen, setGraphLibOpen] = useState(false);
  const [saveGraphName, setSaveGraphName] = useState('');
  const [savingGraph, setSavingGraph] = useState(false);

  const [priceBudget, setPriceBudgetLocal] = useState(config?.agent?.price_budget ?? 100);
  const [tripReward, setTripRewardLocal] = useState(config?.agent?.trip_reward ?? 25);

  useEffect(() => {
    if (config?.agent) {
      setPriceBudgetLocal(config.agent.price_budget ?? 100);
      setTripRewardLocal(config.agent.trip_reward ?? 25);
    }
  }, [config]);

  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pushConfig = (patch: { price_budget?: number; trip_reward?: number }) => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(async () => {
      try {
        await api.config.update({ agent: patch });
        updateConfig({ agent: { ...config?.agent, ...patch } as any });
      } catch (e) {
        console.error('Config update failed:', e);
      }
    }, 300);
  };

  const handlePriceBudget = (val: number) => {
    setPriceBudgetLocal(val);
    pushConfig({ price_budget: val });
  };
  const handleTripReward = (val: number) => {
    setTripRewardLocal(val);
    pushConfig({ trip_reward: val });
  };

  const liveNumAgents = React.useMemo(() => {
    if (!graphData) return numAgents;
    const ownerMax = Math.max(0, ...Object.values(graphData.ownership || {}).map(v => Number(v) + 1));
    const destMax = Math.max(0, ...Object.keys(graphData.destinations || {}).map(k => Number(k) + 1));
    return Math.max(ownerMax, destMax, numAgents, 1);
  }, [graphData, numAgents]);

  const handleNumAgentsChange = (val: number) => {
    setNumAgents(val);
    api.config.update({ agent: { num_agents: val } })
      .then(res => { if (res.config) loadConfig(res.config); })
      .catch((e: unknown) => console.error('Agent count update failed:', e));
  };

  const fetchSavedGraphs = useCallback(async () => {
    try {
      const res = await api.graphs.list();
      setSavedGraphs(res.graphs);
    } catch { /* non-critical */ }
  }, []);

  useEffect(() => { fetchSavedGraphs(); }, [fetchSavedGraphs]);

  const handleSaveGraph = async () => {
    if (!graphData || !saveGraphName.trim()) return;
    setSavingGraph(true);
    try {
      const layoutForApi: Record<string, [number, number]> = {};
      if (graphLayout) {
        for (const [k, v] of Object.entries(graphLayout)) layoutForApi[String(k)] = v;
      }
      await api.graphs.save(saveGraphName.trim(), {
        num_nodes: graphData.num_nodes,
        edges: graphData.edges,
        ownership: Object.fromEntries(Object.entries(graphData.ownership).map(([k, v]) => [String(k), Number(v)])),
        destinations: Object.fromEntries(Object.entries(graphData.destinations).map(([k, v]) => [String(k), v as number[]])),
        starting_positions: Object.fromEntries(Object.entries(graphData.starting_positions || {}).map(([k, v]) => [String(k), Number(v)])),
      }, layoutForApi);
      setSaveGraphName('');
      await fetchSavedGraphs();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Save failed');
    } finally {
      setSavingGraph(false);
    }
  };

  const handleLoadSavedGraph = async (graphId: string) => {
    if (isTraining) return;
    try {
      const res = await api.graphs.load(graphId);
      if (res.status === 'ok' && res.data) {
        const { graph, layout } = res.data;
        const graphResponse = await api.graph.build({
          num_nodes: graph.num_nodes,
          edges: graph.edges,
          ownership: graph.ownership,
          destinations: graph.destinations,
          starting_positions: graph.starting_positions,
        });
        const parsedLayout: Record<number, [number, number]> = {};
        for (const [k, v] of Object.entries(layout || {})) {
          parsedLayout[Number(k)] = v as [number, number];
        }
        loadGraph(graphResponse.graph, Object.keys(parsedLayout).length > 0 ? parsedLayout : graphResponse.layout);
        useTrainingStore.getState().resetEpisodeData();
        setGraphLibOpen(false);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Load failed');
    }
  };

  const handleDeleteSavedGraph = async (graphId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!window.confirm('Delete this saved graph?')) return;
    await api.graphs.delete(graphId).catch(() => {});
    setSavedGraphs(prev => prev.filter(g => g.graph_id !== graphId));
  };

  const handleRandom = async () => {
    if (isTraining) return;
    setError(null);
    try {
      const res = await api.graph.random({
        num_nodes: numNodes,
        num_edges: numEdges === '' ? null : numEdges,
        num_agents: numAgents,
        num_destinations: numDests,
      });
      loadGraph(res.graph, res.layout);
      useTrainingStore.getState().resetEpisodeData();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to generate graph');
    }
  };

  const startSimulation = async () => {
    if (isTraining) return;
    setError(null);
    try {
      if (!graphData) { setError('Build graph first'); return; }
      useTrainingStore.getState().startTraining(runName.trim());
      await api.graph.build({
        num_nodes: graphData.num_nodes,
        edges: graphData.edges,
        ownership: Object.fromEntries(Object.entries(graphData.ownership).map(([k, v]) => [String(k), Number(v)])),
        destinations: Object.fromEntries(Object.entries(graphData.destinations).map(([k, v]) => [String(k), v as number[]])),
        starting_positions: Object.fromEntries(Object.entries(graphData.starting_positions || {}).map(([k, v]) => [String(k), Number(v)])),
      });
      if (graphLayout) {
        const layoutForApi: Record<string, [number, number]> = {};
        for (const [k, v] of Object.entries(graphLayout)) layoutForApi[String(k)] = v;
        await api.graph.syncLayout(layoutForApi);
      }
      await api.simulate.start(runName.trim());
      setRunName('');
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to start simulation');
    }
  };

  const stopTraining = () => api.train.stop().catch(console.error);
  const pauseTraining = async () => { await api.train.pause().catch(console.error); storeSetPaused(); };
  const resumeTraining = async () => { await api.train.resume().catch(console.error); storeSetResumed(); };

  const buildTools: { m: UIMode; icon: React.ReactNode; label: string }[] = [
    { m: 'view', icon: <MousePointer2 size={17} />, label: 'Pointer' },
    { m: 'build_node', icon: <CircleDashed size={17} />, label: 'Add Node' },
    { m: 'build_edge', icon: <Spline size={17} />, label: 'Add Edge' },
    { m: 'build_owner', icon: <User size={17} />, label: 'Set Owner' },
    { m: 'build_dest', icon: <MapPin size={17} />, label: 'Set Dest' },
  ];

  // Collapsed sidebar: show only icons
  if (isSidebarCollapsed) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', height: '100%', alignItems: 'center', padding: '12px 0', gap: 4 }}>
        <button
          onClick={toggleSidebar}
          title="Expand sidebar"
          style={{
            width: 36, height: 36, display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: 'none', border: 'none', color: 'var(--color-text-dim)', cursor: 'pointer',
            borderRadius: 'var(--radius-btn)', marginBottom: 8,
          }}
          onMouseEnter={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.04)'; e.currentTarget.style.color = 'var(--color-text)'; }}
          onMouseLeave={e => { e.currentTarget.style.background = 'none'; e.currentTarget.style.color = 'var(--color-text-dim)'; }}
        >
          <PanelLeftOpen size={16} />
        </button>

        {/* Build tools as vertical icons */}
        {buildTools.map(t => (
          <button
            key={t.m}
            title={t.label}
            onClick={() => setMode(t.m)}
            disabled={isTraining && t.m !== 'view'}
            style={{
              width: 36, height: 36, display: 'flex', alignItems: 'center', justifyContent: 'center',
              background: mode === t.m ? 'var(--color-accent-surface)' : 'none',
              border: mode === t.m ? '1px solid rgba(129,140,248,0.2)' : '1px solid transparent',
              color: mode === t.m ? 'var(--color-accent)' : 'var(--color-text-dim)',
              cursor: isTraining && t.m !== 'view' ? 'not-allowed' : 'pointer',
              borderRadius: 'var(--radius-btn)',
              opacity: isTraining && t.m !== 'view' ? 0.35 : 1,
              transition: 'all var(--transition-fast)',
            }}
          >
            {t.icon}
          </button>
        ))}
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>

      {/* ── SESSION CONTROL ──────────────────────────── */}
      <div style={{ padding: '20px', borderBottom: '1px solid var(--color-border)', background: 'var(--color-accent-surface)' }}>
        {!isTraining && (
          <input
            value={runName}
            onChange={e => setRunName(e.target.value)}
            placeholder="Run name (optional)"
            maxLength={48}
            style={{
              width: '100%', boxSizing: 'border-box',
              marginBottom: 10, padding: '7px 10px',
              background: 'rgba(255,255,255,0.04)',
              border: '1px solid var(--color-border)',
              borderRadius: 'var(--radius-btn)',
              color: 'var(--color-text)',
              fontSize: 'var(--text-sm)',
              outline: 'none',
            }}
            onFocus={e => { e.currentTarget.style.borderColor = 'var(--color-accent)'; }}
            onBlur={e => { e.currentTarget.style.borderColor = 'var(--color-border)'; }}
          />
        )}
        <div style={{ display: 'flex', gap: 8, marginBottom: error ? 12 : 0 }}>
          {isTraining ? (
            <>
              {isPaused ? (
                <Button variant="primary" onClick={resumeTraining} style={{ flex: 1, height: 42 }}>
                  <Play size={16} fill="currentColor" /> Resume
                </Button>
              ) : (
                <Button variant="warning" onClick={pauseTraining} style={{ flex: 1, height: 42 }}>
                  <Pause size={16} fill="currentColor" /> Pause
                </Button>
              )}
              <Button variant="danger" onClick={stopTraining} style={{ flex: 1, height: 42 }}>
                <Square size={16} fill="currentColor" /> Stop
              </Button>
            </>
          ) : (
            <Button
              variant="primary"
              onClick={startSimulation}
              disabled={!graphData}
              style={{
                width: '100%',
                height: 48,
                fontSize: 'var(--text-md)',
                animation: graphData && !isTraining ? 'subtle-pulse 2s infinite' : 'none',
              }}
            >
              <Play size={17} fill="currentColor" /> Start Simulation
            </Button>
          )}
        </div>
        {error && (
          <div style={{
            color: 'var(--color-danger)',
            fontSize: 'var(--text-sm)',
            fontWeight: 600,
            textAlign: 'center',
          }}>
            {error}
          </div>
        )}
      </div>

      {/* ── BUILD TOOLS (icon toolbar) ───────────────── */}
      <div style={{
        padding: '12px 20px',
        borderBottom: '1px solid var(--color-border)',
        display: 'flex',
        alignItems: 'center',
        gap: 4,
      }}>
        {buildTools.map(t => (
          <button
            key={t.m}
            title={t.label}
            onClick={() => setMode(t.m)}
            disabled={isTraining && t.m !== 'view'}
            style={{
              width: 42,
              height: 42,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: mode === t.m ? 'var(--color-accent-surface)' : 'transparent',
              border: mode === t.m ? '1px solid rgba(129,140,248,0.2)' : '1px solid transparent',
              color: mode === t.m ? 'var(--color-accent)' : 'var(--color-text-dim)',
              cursor: isTraining && t.m !== 'view' ? 'not-allowed' : 'pointer',
              borderRadius: 'var(--radius-btn)',
              opacity: isTraining && t.m !== 'view' ? 0.35 : 1,
              transition: 'all var(--transition-fast)',
            }}
            onMouseEnter={e => {
              if (mode !== t.m && !(isTraining && t.m !== 'view'))
                e.currentTarget.style.background = 'rgba(255,255,255,0.04)';
            }}
            onMouseLeave={e => {
              if (mode !== t.m)
                e.currentTarget.style.background = 'transparent';
            }}
          >
            {t.icon}
          </button>
        ))}

        <div style={{ flex: 1 }} />

        <button
          title="Clear canvas"
          onClick={() => { if (window.confirm('Clear all graph data?')) clearAll(); }}
          disabled={isTraining}
          style={{
            width: 42, height: 42,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: 'transparent',
            border: '1px solid transparent',
            color: 'var(--color-text-dim)',
            cursor: isTraining ? 'not-allowed' : 'pointer',
            borderRadius: 'var(--radius-btn)',
            opacity: isTraining ? 0.35 : 1,
            transition: 'all var(--transition-fast)',
          }}
          onMouseEnter={e => { if (!isTraining) { e.currentTarget.style.color = 'var(--color-danger)'; e.currentTarget.style.background = 'rgba(239,68,68,0.08)'; }}}
          onMouseLeave={e => { e.currentTarget.style.color = 'var(--color-text-dim)'; e.currentTarget.style.background = 'transparent'; }}
        >
          <Trash2 size={17} />
        </button>
      </div>

      {/* ── ACCORDION SECTIONS ───────────────────────── */}
      <div style={{ flex: 1, overflowY: 'auto' }}>
        <Accordion title="Graph Generator" icon={<Layout size={17} />} defaultOpen={!isTraining}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 4 }}>
            <Stepper label="Nodes" value={numNodes} onChange={setNumNodes} min={2} max={50} />
            <NumberInput label="Edges" value={numEdges} onChangeValue={setNumEdges} placeholder="Auto" />
            <Stepper label="Agents" value={numAgents} onChange={handleNumAgentsChange} min={1} max={10} />
            <Stepper label="Avg Dests" value={numDests} onChange={setNumDests} min={1} max={8} />
          </div>
          <Button
            variant="secondary"
            onClick={handleRandom}
            disabled={isTraining}
            data-tour="generate-btn"
            style={{ width: '100%', marginTop: 20, height: 40 }}
          >
            Generate Random
          </Button>

          {/* ── Graph Library ── */}
          <div style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid var(--color-border)' }}>
            {/* Save current graph */}
            {graphData && (
              <div style={{ display: 'flex', gap: 6, marginBottom: 8 }}>
                <input
                  value={saveGraphName}
                  onChange={e => setSaveGraphName(e.target.value)}
                  placeholder="Save graph as…"
                  maxLength={48}
                  disabled={isTraining}
                  style={{
                    flex: 1, padding: '5px 8px',
                    background: 'rgba(255,255,255,0.04)',
                    border: '1px solid var(--color-border)',
                    borderRadius: 'var(--radius-btn)',
                    color: 'var(--color-text)',
                    fontSize: 'var(--text-sm)',
                    outline: 'none',
                    opacity: isTraining ? 0.5 : 1,
                  }}
                  onFocus={e => { e.currentTarget.style.borderColor = 'var(--color-accent)'; }}
                  onBlur={e => { e.currentTarget.style.borderColor = 'var(--color-border)'; }}
                  onKeyDown={e => { if (e.key === 'Enter') handleSaveGraph(); }}
                />
                <button
                  onClick={handleSaveGraph}
                  disabled={!saveGraphName.trim() || savingGraph || isTraining}
                  title="Save graph"
                  style={{
                    width: 32, height: 32, display: 'flex', alignItems: 'center', justifyContent: 'center',
                    background: 'rgba(255,255,255,0.05)', border: '1px solid var(--color-border)',
                    borderRadius: 'var(--radius-btn)', color: 'var(--color-text-dim)',
                    cursor: 'pointer', flexShrink: 0,
                    opacity: (!saveGraphName.trim() || savingGraph || isTraining) ? 0.35 : 1,
                  }}
                >
                  <Save size={13} />
                </button>
              </div>
            )}

            {/* Load saved graph */}
            <button
              onClick={() => { setGraphLibOpen(o => !o); if (!graphLibOpen) fetchSavedGraphs(); }}
              disabled={isTraining}
              style={{
                width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                background: 'none', border: '1px solid var(--color-border)',
                borderRadius: 'var(--radius-btn)', padding: '6px 10px',
                color: 'var(--color-text-dim)', fontSize: 'var(--text-sm)', cursor: 'pointer',
                opacity: isTraining ? 0.5 : 1,
              }}
            >
              <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <FolderOpen size={13} /> Load Saved Graph
              </span>
              <span style={{ fontSize: 10, color: 'var(--color-text-dim)' }}>{savedGraphs.length}</span>
            </button>

            {graphLibOpen && (
              <div style={{
                marginTop: 6, border: '1px solid var(--color-border)', borderRadius: 8,
                overflow: 'hidden', background: 'rgba(0,0,0,0.2)',
              }}>
                {savedGraphs.length === 0 ? (
                  <div style={{ padding: '12px', textAlign: 'center', fontSize: 'var(--text-xs)', color: 'var(--color-text-dim)' }}>
                    No saved graphs yet
                  </div>
                ) : savedGraphs.map(g => (
                  <div
                    key={g.graph_id}
                    onClick={() => handleLoadSavedGraph(g.graph_id)}
                    style={{
                      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                      padding: '8px 12px', cursor: 'pointer', borderBottom: '1px solid var(--color-border)',
                    }}
                    onMouseEnter={e => { (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.03)'; }}
                    onMouseLeave={e => { (e.currentTarget as HTMLElement).style.background = 'transparent'; }}
                  >
                    <div>
                      <div style={{ fontSize: 'var(--text-sm)', fontWeight: 600, color: 'var(--color-text)' }}>{g.name}</div>
                      <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-dim)' }}>{g.num_nodes}n · {g.num_agents}a</div>
                    </div>
                    <button
                      onClick={e => handleDeleteSavedGraph(g.graph_id, e)}
                      style={{ background: 'none', border: 'none', padding: 4, color: 'var(--color-text-dim)', cursor: 'pointer', borderRadius: 4 }}
                      onMouseEnter={e => { (e.currentTarget as HTMLElement).style.color = 'var(--color-danger)'; }}
                      onMouseLeave={e => { (e.currentTarget as HTMLElement).style.color = 'var(--color-text-dim)'; }}
                    >
                      <Trash2 size={12} />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </Accordion>

        <Accordion title="Simulation Config" icon={<Activity size={17} />} defaultOpen={isTraining}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 20, marginTop: 4 }}>
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                <span className="text-label">Price Budget</span>
                <span style={{ fontSize: 'var(--text-sm)', fontWeight: 600, color: 'var(--color-text)', fontFamily: 'var(--font-mono)', fontVariantNumeric: 'tabular-nums' }}>${priceBudget}</span>
              </div>
              <input type="range" className="range-slider" min={20} max={500} step={10} value={priceBudget} onChange={e => handlePriceBudget(Number(e.target.value))} disabled={isTraining} style={{ width: '100%' }} />
            </div>
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                <span className="text-label">Trip Reward</span>
                <span style={{ fontSize: 'var(--text-sm)', fontWeight: 600, color: 'var(--color-text)', fontFamily: 'var(--font-mono)', fontVariantNumeric: 'tabular-nums' }}>+{tripReward}</span>
              </div>
              <input type="range" className="range-slider" min={5} max={100} step={5} value={tripReward} onChange={e => handleTripReward(Number(e.target.value))} disabled={isTraining} style={{ width: '100%' }} />
            </div>
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                <span className="text-label">Animation Delay</span>
                <span style={{ fontSize: 'var(--text-sm)', fontWeight: 600, color: animSpeed === 0 ? 'var(--color-success)' : 'var(--color-text)', fontFamily: 'var(--font-mono)', fontVariantNumeric: 'tabular-nums' }}>{animSpeed === 0 ? 'Turbo' : `${animSpeed}ms`}</span>
              </div>
              <input type="range" className="range-slider" min={0} max={1000} step={50} value={animSpeed} onChange={e => setAnimSpeed(Number(e.target.value))} style={{ width: '100%' }} />
            </div>
          </div>
        </Accordion>

        <Accordion title="Display Toggles" icon={<Gauge size={17} />}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px 16px', marginTop: 4 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
              <span style={{ fontSize: 'var(--text-base)', fontWeight: 500, color: showIds ? 'var(--color-text)' : 'var(--color-text-dim)' }}>IDs</span>
              <Toggle checked={showIds} onChange={toggleShowIds} />
            </div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
              <span style={{ fontSize: 'var(--text-base)', fontWeight: 500, color: showPrices ? 'var(--color-text)' : 'var(--color-text-dim)' }}>Prices</span>
              <Toggle checked={showPrices} onChange={toggleShowPrices} />
            </div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
              <span style={{ fontSize: 'var(--text-base)', fontWeight: 500, color: showDests ? 'var(--color-text)' : 'var(--color-text-dim)' }}>Dests</span>
              <Toggle checked={showDests} onChange={toggleShowDests} />
            </div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
              <span style={{ fontSize: 'var(--text-base)', fontWeight: 500, color: showAgents ? 'var(--color-text)' : 'var(--color-text-dim)' }}>Agents</span>
              <Toggle checked={showAgents} onChange={toggleShowAgents} />
            </div>
          </div>
          <div style={{ marginTop: 24 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
              <span className="text-label">Node Size</span>
              <span style={{ fontSize: 'var(--text-sm)', fontWeight: 600, color: 'var(--color-text)', fontFamily: 'var(--font-mono)' }}>{nodeSize}px</span>
            </div>
            <input type="range" className="range-slider" min={16} max={54} step={2} value={nodeSize} onChange={e => setNodeSize(Number(e.target.value))} style={{ width: '100%' }} />
          </div>
        </Accordion>

        <Accordion title="Agent Palette" icon={<FlaskConical size={17} />}>
          <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 12 }}>
            <button onClick={resetAgentColors} style={{
              background: 'none', border: 'none', color: 'var(--color-accent)',
              fontSize: 'var(--text-sm)', fontWeight: 600, cursor: 'pointer', padding: 0,
            }}>
              Reset Colors
            </button>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(44px, 1fr))', gap: 12 }}>
            {Array.from({ length: liveNumAgents }, (_, i) => {
              const color = agentColors[i % agentColors.length] ?? DEFAULT_AGENT_COLORS[i % DEFAULT_AGENT_COLORS.length];
              return (
                <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
                  <ColorPicker value={color} onChange={hex => setAgentColor(i, hex)} />
                  <span style={{ fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--color-text-dim)' }}>A{i}</span>
                </div>
              );
            })}
          </div>
        </Accordion>
      </div>

      {/* ── COLLAPSE TOGGLE ──────────────────────────── */}
      <button
        onClick={toggleSidebar}
        title="Collapse sidebar"
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 6,
          padding: '10px 20px',
          background: 'none',
          border: 'none',
          borderTop: '1px solid var(--color-border)',
          color: 'var(--color-text-dim)',
          cursor: 'pointer',
          fontSize: 'var(--text-sm)',
          fontWeight: 500,
          transition: 'all var(--transition-fast)',
        }}
        onMouseEnter={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.02)'; e.currentTarget.style.color = 'var(--color-text)'; }}
        onMouseLeave={e => { e.currentTarget.style.background = 'none'; e.currentTarget.style.color = 'var(--color-text-dim)'; }}
      >
        <PanelLeftClose size={14} />
        Collapse
      </button>
    </div>
  );
};
