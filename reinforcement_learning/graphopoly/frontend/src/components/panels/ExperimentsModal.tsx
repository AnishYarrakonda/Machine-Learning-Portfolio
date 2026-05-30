import React, { useEffect, useState, useCallback } from 'react';
import { api, ExperimentMeta } from '../../api/client';
import { useReplayStore } from '../../stores/replayStore';
import { adaptToEpisodeJSON } from '../../lib/episodeAdapter';
import { X, Trash2, FolderOpen, RefreshCw } from 'lucide-react';

interface Props {
  onClose: () => void;
}

function formatDate(iso: string) {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

function avgReward(rewards: number[]) {
  if (!rewards.length) return '—';
  const avg = rewards.reduce((a, b) => a + b, 0) / rewards.length;
  return avg.toFixed(1);
}

export const ExperimentsModal: React.FC<Props> = ({ onClose }) => {
  const [experiments, setExperiments] = useState<ExperimentMeta[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingId, setLoadingId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const loadEpisode = useReplayStore(s => s.loadEpisode);

  const fetchList = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.experiments.list();
      setExperiments(res.experiments);
    } catch (e) {
      setError('Failed to load experiments.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchList(); }, [fetchList]);

  const handleLoad = async (runId: string) => {
    setLoadingId(runId);
    setError(null);
    try {
      const res = await api.experiments.load(runId);
      if (res.status === 'ok' && res.data) {
        const episodeData = adaptToEpisodeJSON(res.data);
        loadEpisode(episodeData);
        onClose();
      }
    } catch (e) {
      setError(`Failed to load run: ${e instanceof Error ? e.message : 'Unknown error'}`);
    } finally {
      setLoadingId(null);
    }
  };

  const handleDelete = async (runId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!window.confirm('Delete this run? This cannot be undone.')) return;
    setDeletingId(runId);
    try {
      await api.experiments.delete(runId);
      setExperiments(prev => prev.filter(x => x.run_id !== runId));
    } catch {
      setError('Failed to delete run.');
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 1000,
      background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
    }} onClick={onClose}>
      <div
        onClick={e => e.stopPropagation()}
        style={{
          width: 560, maxHeight: '75vh',
          background: 'var(--color-bg-elevated)',
          border: '1px solid var(--color-border)',
          borderRadius: 12,
          display: 'flex', flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
        {/* Header */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '16px 20px', borderBottom: '1px solid var(--color-border)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <FolderOpen size={16} style={{ color: 'var(--color-accent)' }} />
            <span style={{ fontSize: 'var(--text-md)', fontWeight: 600, color: 'var(--color-text)' }}>
              Past Runs
            </span>
            <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-dim)', fontWeight: 500 }}>
              {experiments.length} saved
            </span>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button
              onClick={fetchList}
              style={{ background: 'none', border: 'none', color: 'var(--color-text-dim)', cursor: 'pointer', padding: 4, borderRadius: 6 }}
              title="Refresh"
            >
              <RefreshCw size={14} />
            </button>
            <button
              onClick={onClose}
              style={{ background: 'none', border: 'none', color: 'var(--color-text-dim)', cursor: 'pointer', padding: 4, borderRadius: 6 }}
            >
              <X size={16} />
            </button>
          </div>
        </div>

        {/* Body */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '8px 0' }}>
          {loading && (
            <div style={{ padding: 32, textAlign: 'center', color: 'var(--color-text-dim)', fontSize: 'var(--text-sm)' }}>
              Loading…
            </div>
          )}
          {!loading && experiments.length === 0 && (
            <div style={{ padding: 40, textAlign: 'center', color: 'var(--color-text-dim)', fontSize: 'var(--text-sm)' }}>
              No saved runs yet. Start a simulation to create one.
            </div>
          )}
          {!loading && experiments.map(exp => (
            <div
              key={exp.run_id}
              style={{
                display: 'flex', alignItems: 'center', gap: 12,
                padding: '10px 20px',
                cursor: 'pointer',
                transition: 'background var(--transition-fast)',
              }}
              onMouseEnter={e => { (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.03)'; }}
              onMouseLeave={e => { (e.currentTarget as HTMLElement).style.background = 'transparent'; }}
              onClick={() => handleLoad(exp.run_id)}
            >
              {/* Mode badge */}
              <span style={{
                fontSize: 10, fontWeight: 700, letterSpacing: '0.04em',
                padding: '2px 6px', borderRadius: 4,
                background: exp.mode === 'train' ? 'rgba(99,102,241,0.15)' : 'rgba(16,185,129,0.12)',
                color: exp.mode === 'train' ? '#818cf8' : '#34d399',
                flexShrink: 0, textTransform: 'uppercase',
              }}>
                {exp.mode}
              </span>

              {/* Name + meta */}
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: 'var(--text-sm)', fontWeight: 600, color: 'var(--color-text)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {exp.run_name || exp.run_id}
                </div>
                <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-dim)', marginTop: 2 }}>
                  {formatDate(exp.finished_at)} · {exp.num_nodes}n {exp.num_agents}a · ep {exp.num_episodes} · avg {avgReward(exp.final_rewards)}
                </div>
              </div>

              {/* Load button */}
              {loadingId === exp.run_id ? (
                <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-accent)' }}>Loading…</span>
              ) : (
                <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-accent)', fontWeight: 600, opacity: 0.7 }}>Load</span>
              )}

              {/* Delete */}
              <button
                onClick={(e) => handleDelete(exp.run_id, e)}
                disabled={deletingId === exp.run_id}
                style={{
                  background: 'none', border: 'none', padding: 4, borderRadius: 4,
                  color: 'var(--color-text-dim)', cursor: 'pointer', flexShrink: 0,
                  opacity: deletingId === exp.run_id ? 0.4 : 0.6,
                }}
                onMouseEnter={e => { (e.currentTarget as HTMLElement).style.color = 'var(--color-danger)'; }}
                onMouseLeave={e => { (e.currentTarget as HTMLElement).style.color = 'var(--color-text-dim)'; }}
              >
                <Trash2 size={13} />
              </button>
            </div>
          ))}
        </div>

        {error && (
          <div style={{ padding: '8px 20px', borderTop: '1px solid var(--color-border)', color: 'var(--color-danger)', fontSize: 'var(--text-sm)' }}>
            {error}
          </div>
        )}
      </div>
    </div>
  );
};
