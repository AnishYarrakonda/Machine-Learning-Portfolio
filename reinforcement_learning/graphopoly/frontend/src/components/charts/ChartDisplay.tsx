import React, { useMemo, useState } from 'react';
import { useAnalyzeStore } from '../../stores/analyzeStore';
import { useReplayStore } from '../../stores/replayStore';
import { useUIStore } from '../../stores/uiStore';
import { chartsByCategory, type BuildParams } from '../../lib/chartRegistry';
import { AGENT_COLORS } from '../../lib/chartTheme';
import { ChartWrapper } from './ChartWrapper';
import { Maximize2, Minimize2 } from 'lucide-react';

export const ChartDisplay: React.FC = () => {
  const { activeCategory, episodeData, timeline, selectedAgents, selectedNodes } = useAnalyzeStore();
  const currentStep = useReplayStore(s => s.currentStep);
  const agentColors = useUIStore(s => s.agentColors);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const params: BuildParams | null = useMemo(() => {
    if (!episodeData || timeline.length === 0) return null;
    return {
      timeline,
      currentStep: Math.min(currentStep, timeline.length - 1),
      episodeData,
      selectedAgents,
      selectedNodes,
      agentColors: agentColors.length > 0 ? agentColors : AGENT_COLORS,
    };
  }, [timeline, currentStep, episodeData, selectedAgents, selectedNodes, agentColors]);

  const charts = chartsByCategory(activeCategory);
  const expandedChart = expandedId ? charts.find(c => c.id === expandedId) ?? null : null;

  if (!params) {
    return (
      <div style={{ flex: 1, height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--color-text-dim)', fontSize: 'var(--text-base)', opacity: 0.5 }}>
        No data available
      </div>
    );
  }

  // ── Expanded single-chart view ─────────────────────────────────────────────
  if (expandedChart) {
    return (
      <div style={{ height: '100%', display: 'flex', flexDirection: 'column', padding: '14px 20px', gap: 10 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexShrink: 0 }}>
          <div>
            <div style={{ fontSize: 'var(--text-md)', fontWeight: 600, color: 'var(--color-text)' }}>
              {expandedChart.title}
            </div>
            <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-dim)', marginTop: 2 }}>
              {expandedChart.syncMode === 'atStep' ? `Step ${params.currentStep}` : `${timeline.length} steps`}
              {' · '}{expandedChart.chartType}
            </div>
          </div>
          <button
            onClick={() => setExpandedId(null)}
            style={{
              display: 'flex', alignItems: 'center', gap: 6,
              background: 'var(--color-bg-hover)',
              border: '1px solid var(--color-border)',
              color: 'var(--color-text-dim)',
              fontSize: 'var(--text-xs)', fontWeight: 500,
              padding: '6px 12px',
              borderRadius: 'var(--radius-btn)',
              cursor: 'pointer',
              transition: 'all var(--transition-fast)',
            }}
            onMouseEnter={e => { e.currentTarget.style.color = 'var(--color-text)'; e.currentTarget.style.borderColor = 'var(--color-border-active)'; }}
            onMouseLeave={e => { e.currentTarget.style.color = 'var(--color-text-dim)'; e.currentTarget.style.borderColor = 'var(--color-border)'; }}
          >
            <Minimize2 size={13} /> All charts
          </button>
        </div>
        <div style={{ flex: 1, position: 'relative', minHeight: 0 }}>
          <ChartWrapper chart={expandedChart} params={params} />
        </div>
      </div>
    );
  }

  // ── Grid view ──────────────────────────────────────────────────────────────
  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))',
      gap: 16,
      padding: '16px 20px',
      alignContent: 'start',
    }}>
      {charts.map(chart => (
        <div
          key={chart.id}
          onClick={() => setExpandedId(chart.id)}
          style={{
            background: 'var(--color-bg-elevated)',
            border: '1px solid var(--color-border)',
            borderRadius: 'var(--radius-card)',
            padding: '16px',
            display: 'flex',
            flexDirection: 'column',
            gap: 10,
            minHeight: 320,
            cursor: 'pointer',
            transition: 'border-color var(--transition-fast)',
          }}
          onMouseEnter={e => (e.currentTarget.style.borderColor = 'var(--color-border-active)')}
          onMouseLeave={e => (e.currentTarget.style.borderColor = 'var(--color-border)')}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexShrink: 0 }}>
            <div>
              <div style={{ fontSize: 'var(--text-md)', fontWeight: 600, color: 'var(--color-text)' }}>
                {chart.title}
              </div>
              <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-dim)', marginTop: 2 }}>
                {chart.syncMode === 'atStep' ? `Step ${params.currentStep}` : `${timeline.length} steps`}
                {' · '}{chart.chartType}
              </div>
            </div>
            <Maximize2 size={13} style={{ color: 'var(--color-text-muted)', flexShrink: 0, marginTop: 2 }} />
          </div>
          <div style={{ flex: 1, position: 'relative', minHeight: 240 }}>
            <ChartWrapper chart={chart} params={params} />
          </div>
        </div>
      ))}
    </div>
  );
};
