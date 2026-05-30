import React, { useEffect, useRef, useState } from 'react';
import { useTrainingStore } from '../../stores/trainingStore';
import { useReplayStore } from '../../stores/replayStore';
import { useUIStore } from '../../stores/uiStore';
import { useConfigStore } from '../../stores/configStore';
import { useGraphStore } from '../../stores/graphStore';
import { AGENT_COLORS } from '../../lib/chartTheme';
import { Activity, ChevronRight, ChevronLeft } from 'lucide-react';

const fmt = (v: number) => Number.isInteger(v) ? String(v) : v.toFixed(1);

export const LiveStatsOverlay: React.FC = () => {
  const { isTraining, agentDetails, stepHistory, simAnimStep, currentPrices, currentEpisode } = useTrainingStore();
  const { episodeData, currentStep: replayStep } = useReplayStore();
  const agentColors = useUIStore(s => s.agentColors);
  const config = useConfigStore(s => s.config);
  const graphData = useGraphStore(s => s.data);

  const isLive = isTraining && stepHistory.length > 0;
  const liveStep = isLive ? stepHistory[Math.min(simAnimStep, stepHistory.length - 1)] : null;
  const replayTrajectoryStep = !isLive && episodeData?.trajectory ? episodeData.trajectory[replayStep] : null;

  const numAgents = agentDetails.length || episodeData?.metadata?.num_agents || config?.agent?.num_agents || 0;
  const displayStep = isLive ? (liveStep?.step ?? simAnimStep) : replayStep;

  // Auto-open when training starts, auto-close when it stops
  const [open, setOpen] = useState(false);
  const prevTrainingRef = useRef(false);
  useEffect(() => {
    if (isTraining && !prevTrainingRef.current) setOpen(true);
    if (!isTraining && prevTrainingRef.current) setOpen(false);
    prevTrainingRef.current = isTraining;
  }, [isTraining]);

  // Nothing to show at all
  const hasData = isLive || (replayTrajectoryStep != null) || agentDetails.length > 0;
  if (!hasData && !open) return null;

  const PANEL_W = 256;

  return (
    <>
      {/* Toggle tab — always visible on the right edge */}
      <button
        onClick={() => setOpen(o => !o)}
        title={open ? 'Hide live stats' : 'Show live stats'}
        style={{
          position: 'absolute',
          right: open ? PANEL_W : 0,
          top: '50%',
          transform: 'translateY(-50%)',
          zIndex: 30,
          width: 20,
          height: 72,
          background: 'var(--color-bg-elevated)',
          border: '1px solid var(--color-border)',
          borderRight: open ? 'none' : '1px solid var(--color-border)',
          borderRadius: open ? '6px 0 0 6px' : '6px 0 0 6px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 4,
          cursor: 'pointer',
          color: isTraining ? 'var(--color-accent)' : 'var(--color-text-dim)',
          transition: 'right var(--transition-base)',
        }}
      >
        {open ? <ChevronRight size={12} /> : <ChevronLeft size={12} />}
        <Activity size={11} style={{ opacity: 0.7 }} />
      </button>

      {/* Panel */}
      {open && (
        <div style={{
          position: 'absolute',
          right: 0,
          top: 0,
          width: PANEL_W,
          height: '100%',
          background: 'var(--color-bg-elevated)',
          borderLeft: '1px solid var(--color-border)',
          display: 'flex',
          flexDirection: 'column',
          zIndex: 25,
          overflowY: 'auto',
        }}>
          {/* Header */}
          <div style={{
            padding: '10px 14px',
            borderBottom: '1px solid var(--color-border)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
              {isTraining && <div style={{
                width: 6, height: 6, borderRadius: '50%',
                background: 'var(--color-accent)',
                animation: 'pulse 1.5s infinite',
              }} />}
              <span style={{ fontSize: 'var(--text-sm)', fontWeight: 600, color: 'var(--color-text)' }}>
                Live Stats
              </span>
            </div>
            <div style={{ display: 'flex', gap: 12 }}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 10, color: 'var(--color-text-muted)', fontWeight: 500, letterSpacing: '0.05em', textTransform: 'uppercase' }}>Step</div>
                <div style={{ fontSize: 'var(--text-sm)', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--color-text)' }}>{displayStep}</div>
              </div>
              {isTraining && currentEpisode > 0 && (
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 10, color: 'var(--color-text-muted)', fontWeight: 500, letterSpacing: '0.05em', textTransform: 'uppercase' }}>Ep</div>
                  <div style={{ fontSize: 'var(--text-sm)', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--color-text)' }}>{currentEpisode}</div>
                </div>
              )}
            </div>
          </div>

          {/* Agent cards */}
          <div style={{ flex: 1, padding: '8px 10px', display: 'flex', flexDirection: 'column', gap: 6 }}>
            {numAgents === 0 && (
              <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', opacity: 0.3 }}>
                <span style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-dim)' }}>No data yet</span>
              </div>
            )}
            {Array.from({ length: numAgents }, (_, i) => {
              const aid = String(i);
              const color = agentColors[i % agentColors.length] ?? AGENT_COLORS[i % AGENT_COLORS.length];
              const detail = agentDetails.find(a => a.agent_id === i);
              const replayStats = replayTrajectoryStep?.agent_stats?.[aid];

              const reward = detail?.cumulative_reward ?? replayStats?.total_profit ?? 0;
              const trips = detail?.trips_completed ?? replayStats?.trips_completed ?? 0;
              const taxRev = detail?.tax_revenue ?? replayStats?.tax_revenue ?? 0;
              const taxPaid = detail?.tax_paid ?? replayStats?.tax_paid ?? 0;
              const destRev = detail?.dest_revenue ?? replayStats?.dest_revenue ?? 0;
              const profit = reward >= 0;

              return (
                <div key={i} style={{
                  background: 'rgba(255,255,255,0.025)',
                  border: `1px solid ${color}22`,
                  borderLeft: `3px solid ${color}`,
                  borderRadius: 8,
                  padding: '8px 10px',
                }}>
                  {/* Agent name + P/L */}
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <div style={{ width: 8, height: 8, borderRadius: '50%', background: color }} />
                      <span style={{ fontSize: 'var(--text-sm)', fontWeight: 700, color }}> A{i}</span>
                    </div>
                    <span style={{
                      fontSize: 10, fontWeight: 700,
                      padding: '1px 6px', borderRadius: 4,
                      background: profit ? 'rgba(34,197,94,0.1)' : 'rgba(239,68,68,0.1)',
                      color: profit ? 'var(--color-success)' : 'var(--color-danger)',
                    }}>
                      {profit ? '+' : ''}{fmt(reward)}
                    </span>
                  </div>

                  {/* Metrics grid */}
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 8px' }}>
                    {[
                      ['Trips', trips],
                      ['Dest', destRev],
                      ['Tax in', taxRev],
                      ['Tax out', taxPaid],
                    ].map(([label, val]) => (
                      <div key={label as string} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span style={{ fontSize: 10, color: 'var(--color-text-muted)', fontWeight: 500 }}>{label}</span>
                        <span style={{ fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--color-text)', fontFamily: 'var(--font-mono)' }}>
                          {fmt(val as number)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Price bar */}
          {(graphData?.num_nodes ?? 0) > 0 && (
            <div style={{
              padding: '8px 12px',
              borderTop: '1px solid var(--color-border)',
              display: 'flex',
              flexWrap: 'wrap',
              gap: '4px 10px',
            }}>
              <span style={{ width: '100%', fontSize: 10, color: 'var(--color-text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 2 }}>Node Prices</span>
              {Array.from({ length: Math.min(graphData?.num_nodes ?? 0, 12) }, (_, i) => {
                const price = isLive
                  ? (liveStep?.prices?.[String(i)] ?? currentPrices?.[String(i)] ?? 0)
                  : (replayTrajectoryStep?.prices?.[String(i)] ?? 0);
                const owner = graphData?.ownership?.[i] ?? -1;
                const c = owner >= 0 ? (agentColors[owner % agentColors.length] ?? AGENT_COLORS[owner % AGENT_COLORS.length]) : 'var(--color-text-dim)';
                return (
                  <div key={i} style={{ display: 'flex', gap: 3, alignItems: 'center' }}>
                    <span style={{ fontSize: 10, fontWeight: 600, color: c }}>N{i}</span>
                    <span style={{ fontSize: 10, color: 'var(--color-text)', fontFamily: 'var(--font-mono)' }}>${Number(price).toFixed(0)}</span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </>
  );
};
