import React from 'react';
import { useAnalyzeStore } from '../../stores/analyzeStore';
import { CATEGORIES } from '../../lib/chartRegistry';
import { useUIStore } from '../../stores/uiStore';
import { AGENT_COLORS } from '../../lib/chartTheme';

export const ChartNavigator: React.FC = () => {
  const { activeCategory, setCategory,
    selectedAgents, selectedNodes, toggleAgent, toggleNode, episodeData } = useAnalyzeStore();
  const agentColors = useUIStore(s => s.agentColors);

  const numAgents = episodeData?.metadata?.num_agents ?? 0;
  const numNodes = episodeData?.metadata?.num_nodes ?? 0;

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: 24,
      padding: '12px 24px',
      borderBottom: '1px solid var(--color-border)',
      background: 'rgba(255,255,255,0.01)',
      flexWrap: 'wrap',
    }}>
      {/* ── CATEGORIES ─────────────────────────────────── */}
      <div style={{ display: 'flex', gap: 6, overflowX: 'auto', scrollbarWidth: 'none' }}>
        {CATEGORIES.map(cat => (
          <button
            key={cat.id}
            onClick={() => setCategory(cat.id)}
            style={{
              padding: '7px 16px',
              borderRadius: 'var(--radius-pill)',
              fontSize: 'var(--text-sm)',
              fontWeight: 600,
              letterSpacing: '0.01em',
              background: activeCategory === cat.id ? 'var(--color-accent)' : 'rgba(255,255,255,0.03)',
              color: activeCategory === cat.id ? '#fff' : 'var(--color-text-dim)',
              border: activeCategory === cat.id ? '1px solid var(--color-accent)' : '1px solid var(--color-border)',
              cursor: 'pointer',
              whiteSpace: 'nowrap',
              transition: 'all 0.2s',
            }}
          >
            {cat.label}
          </button>
        ))}
      </div>

      <div style={{ height: 20, width: 1, background: 'var(--color-border)', flexShrink: 0 }} />

      {/* ── FILTERS ────────────────────────────────────── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 20, flexWrap: 'wrap' }}>
        {numAgents > 0 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span className="text-label" style={{ fontSize: 'var(--text-xs)' }}>Agents</span>
            <div style={{ display: 'flex', gap: 4 }}>
              {Array.from({ length: numAgents }, (_, i) => {
                const id = String(i);
                const active = selectedAgents.length === 0 || selectedAgents.includes(id);
                const color = agentColors[i % agentColors.length] ?? AGENT_COLORS[i % AGENT_COLORS.length];
                return (
                  <button
                    key={id}
                    onClick={() => toggleAgent(id)}
                    style={{
                      padding: '3px 10px',
                      fontSize: 'var(--text-xs)',
                      fontWeight: 700,
                      borderRadius: 'var(--radius-sm)',
                      background: active ? color + '22' : 'transparent',
                      color: active ? color : 'var(--color-text-dim)',
                      border: `1px solid ${active ? color : 'var(--color-border)'}`,
                      cursor: 'pointer',
                      transition: 'all 0.15s',
                    }}
                  >
                    A{i}
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {numNodes > 0 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span className="text-label" style={{ fontSize: 'var(--text-xs)' }}>Nodes</span>
            <div style={{ display: 'flex', gap: 4, overflowX: 'auto', maxWidth: 400, scrollbarWidth: 'none' }}>
              {Array.from({ length: numNodes }, (_, i) => {
                const id = String(i);
                const active = selectedNodes.length === 0 || selectedNodes.includes(id);
                const owner = episodeData?.graph?.ownership?.[id] ?? -1;
                const color = owner >= 0 ? (agentColors[owner % agentColors.length] ?? AGENT_COLORS[owner % AGENT_COLORS.length]) : 'var(--color-text-dim)';
                return (
                  <button
                    key={id}
                    onClick={() => toggleNode(id)}
                    style={{
                      padding: '3px 10px',
                      fontSize: 'var(--text-xs)',
                      fontWeight: 700,
                      borderRadius: 'var(--radius-sm)',
                      background: active ? color + '22' : 'transparent',
                      color: active ? color : 'var(--color-text-dim)',
                      border: `1px solid ${active ? color : 'var(--color-border)'}`,
                      cursor: 'pointer',
                      whiteSpace: 'nowrap',
                      transition: 'all 0.15s',
                    }}
                  >
                    N{i}
                  </button>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
