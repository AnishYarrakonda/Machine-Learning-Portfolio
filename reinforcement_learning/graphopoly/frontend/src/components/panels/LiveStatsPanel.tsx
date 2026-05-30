import React from 'react';
import { useReplayStore } from '../../stores/replayStore';
import { useTrainingStore } from '../../stores/trainingStore';
import { useGraphStore } from '../../stores/graphStore';
import { useConfigStore } from '../../stores/configStore';
import { useUIStore } from '../../stores/uiStore';
import { AGENT_COLORS } from '../../lib/chartTheme';
import { Activity } from 'lucide-react';

const StatPill: React.FC<{ label: string; value: number | string; color?: string }> = ({ label, value, color }) => (
  <div style={{
    display: 'flex',
    flexDirection: 'column',
    gap: 4,
    padding: '8px 10px',
    background: 'rgba(255,255,255,0.02)',
    borderRadius: 'var(--radius-sm)',
  }}>
    <span className="text-label" style={{ fontSize: 'var(--text-xs)', letterSpacing: '0.05em' }}>{label}</span>
    <span style={{
      fontSize: 'var(--text-lg)',
      fontWeight: 700,
      color: color || 'var(--color-text)',
      fontFamily: 'var(--font-mono)',
      fontVariantNumeric: 'tabular-nums',
      lineHeight: 1.2,
    }}>
      {typeof value === 'number' ? (Number.isInteger(value) ? value : value.toFixed(1)) : value}
    </span>
  </div>
);

export const LiveStatsPanel: React.FC = () => {
  const { episodeData, currentStep: replayStep } = useReplayStore();
  const { isTraining, agentDetails, stepHistory, simAnimStep, currentPrices } = useTrainingStore();
  const graphData = useGraphStore(s => s.data);
  const config = useConfigStore(s => s.config);
  const agentColors = useUIStore(s => s.agentColors);

  const isLive = isTraining && stepHistory.length > 0;
  const liveStep = isLive ? stepHistory[simAnimStep] : null;
  const replayTrajectoryStep = !isLive ? episodeData?.trajectory?.[replayStep] : null;

  const displayStep = isLive ? (liveStep?.step ?? simAnimStep) : replayStep;
  const numAgentsDisplay = episodeData?.metadata?.num_agents ?? config?.agent?.num_agents ?? 2;

  if (!isLive && !replayTrajectoryStep && agentDetails.length === 0) {
    return (
      <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', opacity: 0.3 }}>
        <div style={{ textAlign: 'center' }}>
          <Activity size={28} style={{ marginBottom: 8, opacity: 0.5 }} />
          <div style={{ fontSize: 'var(--text-md)', fontWeight: 500, marginBottom: 4 }}>No Active Live Data</div>
          <div style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-dim)' }}>Start a simulation to see real-time agent metrics</div>
        </div>
      </div>
    );
  }

  const COL_HEADERS = ['Net Reward', 'Trips', 'Dest Rev', 'Tax Rev', 'Tax Paid', 'Avg/Step'];
  const COL_GRID = '110px 56px repeat(6, 1fr)';

  return (
    <div style={{ padding: '0', display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* ── AGENT TABLE ─────────────────────────────────────── */}
      <div>
        {/* Column headers */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: COL_GRID,
          gap: '0 12px',
          padding: '6px 20px',
          borderBottom: '1px solid var(--color-border)',
        }}>
          <span className="text-label">Agent</span>
          <span className="text-label"></span>
          {COL_HEADERS.map(h => (
            <span key={h} className="text-label" style={{ fontSize: 'var(--text-xs)', letterSpacing: '0.05em' }}>{h}</span>
          ))}
        </div>

        {Array.from({ length: numAgentsDisplay }, (_, i) => {
          const aid = String(i);
          const replayStats = replayTrajectoryStep?.agent_stats?.[aid];
          const detail = agentDetails.find(a => a.agent_id === i);
          const color = agentColors[i % agentColors.length] ?? AGENT_COLORS[i % AGENT_COLORS.length];

          const reward = detail?.cumulative_reward ?? replayStats?.total_profit ?? 0;
          const trips = detail?.trips_completed ?? replayStats?.trips_completed ?? 0;
          const taxRev = detail?.tax_revenue ?? replayStats?.tax_revenue ?? 0;
          const taxPaid = detail?.tax_paid ?? replayStats?.tax_paid ?? 0;
          const destRev = detail?.dest_revenue ?? replayStats?.dest_revenue ?? 0;
          const avgPerStep = displayStep > 0 ? reward / (displayStep + 1) : 0;
          const isProfitable = reward >= 0;

          const cellStyle: React.CSSProperties = {
            fontSize: 'var(--text-sm)',
            fontWeight: 600,
            color: 'var(--color-text)',
            fontFamily: 'var(--font-mono)',
            fontVariantNumeric: 'tabular-nums',
            display: 'flex',
            alignItems: 'center',
          };

          const fmt = (v: number) => Number.isInteger(v) ? String(v) : v.toFixed(1);

          return (
            <div key={i} style={{
              display: 'grid',
              gridTemplateColumns: COL_GRID,
              gap: '0 12px',
              padding: '9px 20px',
              borderBottom: '1px solid var(--color-border-subtle)',
              alignItems: 'center',
            }}>
              {/* Agent name */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                <div style={{ width: 8, height: 8, borderRadius: '50%', background: color, flexShrink: 0 }} />
                <span style={{ fontSize: 'var(--text-sm)', fontWeight: 700, color }}> Agent {i}</span>
              </div>

              {/* Profit/Loss badge */}
              <div style={{
                fontSize: 10,
                fontWeight: 700,
                padding: '2px 6px',
                background: isProfitable ? 'rgba(34,197,94,0.1)' : 'rgba(239,68,68,0.1)',
                borderRadius: 'var(--radius-sm)',
                color: isProfitable ? 'var(--color-success)' : 'var(--color-danger)',
                textAlign: 'center',
                letterSpacing: '0.02em',
                alignSelf: 'center',
              }}>
                {isProfitable ? 'Profit' : 'Loss'}
              </div>

              {/* Metrics */}
              <span style={{ ...cellStyle, color: isProfitable ? 'var(--color-success)' : 'var(--color-danger)' }}>{fmt(reward)}</span>
              <span style={cellStyle}>{fmt(trips)}</span>
              <span style={cellStyle}>{fmt(destRev)}</span>
              <span style={cellStyle}>{fmt(taxRev)}</span>
              <span style={cellStyle}>{fmt(taxPaid)}</span>
              <span style={{ ...cellStyle, color: 'var(--color-text-secondary)' }}>{fmt(avgPerStep)}</span>
            </div>
          );
        })}
      </div>

      {/* ── SYSTEM OVERVIEW ─────────────────────────────── */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 24,
        padding: '10px 20px',
        background: 'rgba(255,255,255,0.02)',
        borderTop: '1px solid var(--color-border)',
      }}>
        <div style={{ display: 'flex', gap: 20 }}>
          <StatPill label="Step" value={displayStep} />
          <StatPill label="Completions" value={isLive ? (liveStep?.dest_completions?.length ?? 0) : (replayTrajectoryStep?.dest_completions?.length ?? 0)} />
          <StatPill label="Agents" value={numAgentsDisplay} />
        </div>

        <div style={{ height: 24, width: 1, background: 'var(--color-border)' }} />

        <div style={{ flex: 1, display: 'flex', gap: 16, overflowX: 'auto', alignItems: 'center' }}>
          <span className="text-label" style={{ flexShrink: 0 }}>Prices</span>
          {Array.from({ length: Math.min(graphData?.num_nodes ?? 0, 10) }, (_, i) => {
            const price = isLive ? (liveStep?.prices?.[String(i)] ?? currentPrices?.[String(i)] ?? 0) : (replayTrajectoryStep?.prices?.[String(i)] ?? 0);
            const owner = graphData?.ownership?.[i] ?? -1;
            const color = owner >= 0 ? (agentColors[owner % agentColors.length] ?? AGENT_COLORS[owner % AGENT_COLORS.length]) : 'var(--color-text-dim)';
            return (
              <div key={i} style={{ display: 'flex', gap: 3, alignItems: 'center' }}>
                <span style={{ fontSize: 'var(--text-xs)', fontWeight: 600, color }}>N{i}</span>
                <span style={{ fontSize: 'var(--text-sm)', fontWeight: 500, color: 'var(--color-text)', fontFamily: 'var(--font-mono)', fontVariantNumeric: 'tabular-nums' }}>${Number(price).toFixed(1)}</span>
              </div>
            );
          })}
          {(graphData?.num_nodes ?? 0) > 10 && <span style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-dim)' }}>...</span>}
        </div>
      </div>
    </div>
  );
};
