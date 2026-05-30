import React from 'react';
import { useTrainingStore } from '../../stores/trainingStore';
import { useWebSocket } from '../../hooks/useWebSocket';
import { HelpCircle } from 'lucide-react';
import { AGENT_COLORS } from '../../lib/chartTheme';
import { useUIStore } from '../../stores/uiStore';

const LogoIcon: React.FC = () => (
  <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
    <circle cx="4" cy="14" r="2.5" fill="var(--color-accent)" opacity="0.9" />
    <circle cx="16" cy="14" r="2.5" fill="var(--color-accent)" opacity="0.9" />
    <circle cx="10" cy="4" r="2.5" fill="var(--color-accent)" opacity="0.9" />
    <line x1="4" y1="14" x2="10" y2="4" stroke="var(--color-accent)" strokeWidth="1.2" opacity="0.5" />
    <line x1="16" y1="14" x2="10" y2="4" stroke="var(--color-accent)" strokeWidth="1.2" opacity="0.5" />
    <line x1="4" y1="14" x2="16" y2="14" stroke="var(--color-accent)" strokeWidth="1.2" opacity="0.5" />
  </svg>
);

export const Header: React.FC = () => {
  const { isTraining, runName, currentEpisode, episodeRewards } = useTrainingStore();
  const agentColors = useUIStore(s => s.agentColors);
  const wsStatus = useWebSocket();

  const avgReward = episodeRewards.length > 0
    ? episodeRewards.reduce((a, b) => a + b, 0) / episodeRewards.length
    : null;

  const wsColor = wsStatus === 'connected'
    ? 'var(--color-success)'
    : wsStatus === 'connecting'
    ? 'var(--color-warning)'
    : 'var(--color-danger)';

  const handleHelpClick = () => {
    localStorage.removeItem('graphopoly_onboarded_v3');
    window.location.reload();
  };

  return (
    <header className="app-header">
      {/* ── LEFT: logo ── */}
      <div className="header-left">
        <div className="header-logo">
          <LogoIcon />
          <div className="header-logo-text">
            <span className="header-title">Graphopoly</span>
            <span className="header-subtitle">MARL Research Platform</span>
          </div>
        </div>

        {/* WS status dot */}
        <div className="header-status-pill">
          <div className="header-status-dot" style={{
            background: wsColor,
            boxShadow: wsStatus === 'connected' ? `0 0 6px ${wsColor}` : 'none',
            animation: wsStatus === 'connecting' ? 'pulse 1.5s infinite' : 'none',
          }} />
          {wsStatus === 'connected' ? 'Connected' : wsStatus === 'connecting' ? 'Connecting…' : 'Disconnected'}
        </div>
      </div>

      {/* ── CENTER: live run info ── */}
      {isTraining && (
        <div className="header-run-bar">
          {/* Pulsing indicator */}
          <div className="header-run-dot" />

          {/* Run name */}
          <span className="header-run-name">
            {runName || 'Running'}
          </span>

          <div className="header-run-divider" />

          {/* Episode count */}
          <div className="header-run-stat">
            <span className="header-run-stat-label">Episode</span>
            <span className="header-run-stat-value">{currentEpisode.toLocaleString()}</span>
          </div>

          <div className="header-run-divider" />

          {/* Per-agent rewards */}
          {episodeRewards.length > 0 && (
            <div className="header-run-rewards">
              {episodeRewards.map((r, i) => {
                const color = agentColors[i % agentColors.length] ?? AGENT_COLORS[i % AGENT_COLORS.length];
                return (
                  <div key={i} className="header-run-reward-chip" title={`Agent ${i}`}>
                    <div style={{ width: 6, height: 6, borderRadius: '50%', background: color, flexShrink: 0 }} />
                    <span style={{
                      color: r >= 0 ? 'var(--color-success)' : 'var(--color-danger)',
                      fontWeight: 700,
                    }}>
                      {r >= 0 ? '+' : ''}{r.toFixed(1)}
                    </span>
                  </div>
                );
              })}
            </div>
          )}

          {avgReward !== null && (
            <>
              <div className="header-run-divider" />
              <div className="header-run-stat">
                <span className="header-run-stat-label">Avg</span>
                <span className="header-run-stat-value" style={{
                  color: avgReward >= 0 ? 'var(--color-success)' : 'var(--color-danger)',
                }}>
                  {avgReward >= 0 ? '+' : ''}{avgReward.toFixed(1)}
                </span>
              </div>
            </>
          )}
        </div>
      )}

      {/* ── RIGHT: help ── */}
      <button className="header-help-btn" onClick={handleHelpClick} title="Show onboarding tour">
        <HelpCircle size={16} />
      </button>

      <style>{`
        .app-header {
          height: var(--header-h);
          display: flex;
          align-items: center;
          padding: 0 20px;
          justify-content: space-between;
          position: fixed;
          top: 0; left: 0; right: 0;
          z-index: 100;
          background: var(--color-bg-elevated);
          border-bottom: 1px solid var(--color-border-active);
        }
        .header-left {
          display: flex;
          align-items: center;
          gap: 16px;
        }
        .header-logo {
          display: flex;
          align-items: center;
          gap: 10px;
        }
        .header-logo-text {
          display: flex;
          flex-direction: column;
          gap: 1px;
        }
        .header-title {
          font-size: 15px;
          font-weight: 600;
          letter-spacing: 0.02em;
          background: linear-gradient(135deg, var(--color-text), var(--color-accent));
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
          line-height: 1.2;
        }
        .header-subtitle {
          font-size: 10px;
          font-weight: 500;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: var(--color-text-muted);
          line-height: 1;
        }
        .header-status-pill {
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 4px 10px;
          background: rgba(255,255,255,0.03);
          border-radius: var(--radius-pill);
          font-size: var(--text-sm);
          font-weight: 500;
          color: var(--color-text-dim);
          border: 1px solid var(--color-border);
        }
        .header-status-dot {
          width: 6px;
          height: 6px;
          border-radius: 50%;
          flex-shrink: 0;
        }

        /* ── Run bar ── */
        .header-run-bar {
          position: absolute;
          left: 50%;
          transform: translateX(-50%);
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 6px 16px;
          background: var(--color-bg-surface);
          border: 1px solid var(--color-border);
          border-radius: var(--radius-pill);
          font-size: var(--text-sm);
          white-space: nowrap;
        }
        .header-run-dot {
          width: 7px;
          height: 7px;
          border-radius: 50%;
          background: var(--color-accent);
          animation: pulse 1.5s infinite;
          flex-shrink: 0;
        }
        .header-run-name {
          font-weight: 600;
          color: var(--color-text);
          max-width: 160px;
          overflow: hidden;
          text-overflow: ellipsis;
        }
        .header-run-divider {
          width: 1px;
          height: 14px;
          background: var(--color-border);
          flex-shrink: 0;
        }
        .header-run-stat {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 1px;
        }
        .header-run-stat-label {
          font-size: 10px;
          font-weight: 500;
          letter-spacing: 0.06em;
          text-transform: uppercase;
          color: var(--color-text-muted);
          line-height: 1;
        }
        .header-run-stat-value {
          font-size: var(--text-sm);
          font-weight: 700;
          color: var(--color-text);
          font-family: var(--font-mono);
          font-variant-numeric: tabular-nums;
          line-height: 1;
        }
        .header-run-rewards {
          display: flex;
          gap: 8px;
          align-items: center;
        }
        .header-run-reward-chip {
          display: flex;
          align-items: center;
          gap: 4px;
          font-size: var(--text-sm);
          font-family: var(--font-mono);
          font-variant-numeric: tabular-nums;
        }

        .header-help-btn {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 32px;
          height: 32px;
          border-radius: var(--radius-btn);
          border: 1px solid var(--color-border);
          background: transparent;
          color: var(--color-text-dim);
          cursor: pointer;
          transition: all var(--transition-fast);
        }
        .header-help-btn:hover {
          color: var(--color-text);
          border-color: var(--color-border-active);
          background: rgba(255,255,255,0.03);
        }
      `}</style>
    </header>
  );
};
