import React, { useEffect, useCallback, useState } from 'react';
import { useReplayStore } from '../../stores/replayStore';
import { useTrainingStore } from '../../stores/trainingStore';
import { useAnalyzeStore } from '../../stores/analyzeStore';
import { useUIStore } from '../../stores/uiStore';
import { api } from '../../api/client';
import { Play, Pause, SkipBack, SkipForward, FastForward, Rewind, Download, FolderOpen } from 'lucide-react';
import { RangeSlider, Button } from '../shared';
import { ChartNavigator } from '../charts/ChartNavigator';
import { ChartDisplay } from '../charts/ChartDisplay';
import { ExperimentsModal } from './ExperimentsModal';
import { downloadAllCsvsAsZip } from '../../lib/csvExport';
import { AGENT_COLORS } from '../../lib/chartTheme';
import type { BuildParams } from '../../lib/chartRegistry';

const EmptyState = ({ message }: { message: string }) => (
  <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', opacity: 0.3 }}>
    <div style={{ fontSize: 'var(--text-sm)', fontWeight: 500, color: 'var(--color-text-dim)' }}>{message}</div>
  </div>
);

export const AnalysisReplayPanel: React.FC = () => {
  const { episodeData, currentStep, totalSteps, isPlaying, play, pause, setStep, stepBack, stepForward, jumpBack, jumpForward } = useReplayStore();
  const { isTraining } = useTrainingStore();
  const { setAnalysisData, episodeData: analyzeEpisodeData, timeline } = useAnalyzeStore();
  const agentColors = useUIStore(s => s.agentColors);
  const animSpeed = useUIStore(s => s.animSpeed);
  const setAnimSpeed = useUIStore(s => s.setAnimSpeed);

  const [isComputing, setIsComputing] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [showExperiments, setShowExperiments] = useState(false);

  // Auto-compute analysis timeline when episode data becomes available
  const computeTimeline = useCallback(async (data: typeof episodeData) => {
    if (!data) return;
    setIsComputing(true);
    try {
      const res = await api.analyze.compute(data);
      setAnalysisData(data, res.timeline);
    } catch (e) {
      console.error('Analysis computation failed:', e);
    } finally {
      setIsComputing(false);
    }
  }, [setAnalysisData]);

  useEffect(() => {
    if (episodeData && episodeData !== analyzeEpisodeData) {
      computeTimeline(episodeData);
    }
  }, [episodeData, analyzeEpisodeData, computeTimeline]);

  const handleDownloadAll = async () => {
    if (!analyzeEpisodeData || timeline.length === 0) return;
    setIsDownloading(true);
    try {
      const params: BuildParams = {
        timeline,
        currentStep,
        episodeData: analyzeEpisodeData,
        selectedAgents: [],
        selectedNodes: [],
        agentColors: agentColors.length > 0 ? agentColors : AGENT_COLORS,
      };
      await downloadAllCsvsAsZip(params);
    } catch (e) {
      console.error('Download failed:', e);
    } finally {
      setIsDownloading(false);
    }
  };

  const hasCharts = timeline.length > 0;

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', background: 'var(--color-bg-surface)' }}>

      {/* ── REPLAY CONTROLS BANNER ────────────────────────────────── */}
      <div style={{ 
        padding: '12px 24px', 
        borderBottom: '1px solid var(--color-border)',
        display: 'flex',
        alignItems: 'center',
        gap: 24,
        background: 'rgba(255,255,255,0.01)'
      }}>
        <div style={{ flex: 1, maxWidth: 600 }}>
          {episodeData && (
            <RangeSlider
              value={currentStep}
              min={0}
              max={totalSteps > 0 ? totalSteps - 1 : 0}
              onChange={setStep}
              label={`Step ${currentStep}`}
              formatValue={() => `/ ${totalSteps - 1}`}
            />
          )}
        </div>

        <div style={{ display: 'flex', gap: 6 }}>
          <Button onClick={() => jumpBack(10)} variant="secondary" style={{ padding: '8px 12px' }}><Rewind size={14} /></Button>
          <Button onClick={stepBack} variant="secondary" style={{ padding: '8px 12px' }}><SkipBack size={14} /></Button>
          
          {isPlaying ? (
            <Button onClick={pause} variant="primary" style={{ padding: '8px 20px', background: 'var(--color-text)', color: 'var(--color-bg)' }}>
              <Pause size={16} fill="currentColor" />
            </Button>
          ) : (
            <Button onClick={play} variant="primary" style={{ padding: '8px 20px' }}>
              <Play size={16} fill="currentColor" />
            </Button>
          )}

          <Button onClick={stepForward} variant="secondary" style={{ padding: '8px 12px' }}><SkipForward size={14} /></Button>
          <Button onClick={() => jumpForward(10)} variant="secondary" style={{ padding: '8px 12px' }}><FastForward size={14} /></Button>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          {/* Playback speed */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span className="text-label" style={{ whiteSpace: 'nowrap', fontSize: 'var(--text-xs)' }}>Speed</span>
            <input
              type="range"
              className="range-slider"
              min={0}
              max={1000}
              step={50}
              value={animSpeed}
              onChange={e => setAnimSpeed(Number(e.target.value))}
              style={{ width: 80 }}
            />
            <span style={{ fontSize: 'var(--text-xs)', fontWeight: 600, color: animSpeed === 0 ? 'var(--color-success)' : 'var(--color-text)', fontFamily: 'var(--font-mono)', minWidth: 36 }}>
              {animSpeed === 0 ? 'Max' : `${animSpeed}ms`}
            </span>
          </div>

          <div style={{ display: 'flex', gap: 8 }}>
            <Button onClick={() => setShowExperiments(true)} variant="secondary" style={{ fontSize: 10, padding: '8px 16px', gap: 8 }}>
              <FolderOpen size={14} /> Past Runs
            </Button>
            {hasCharts && (
              <Button onClick={handleDownloadAll} disabled={isDownloading} variant="secondary" style={{ fontSize: 10, padding: '8px 16px', gap: 8 }}>
                <Download size={14} /> {isDownloading ? 'Exporting...' : 'Export CSV'}
              </Button>
            )}
          </div>
        </div>
      </div>

      {showExperiments && <ExperimentsModal onClose={() => setShowExperiments(false)} />}

      {/* ── ANALYSIS CONTENT ─────────────────────────────────────── */}
      <div style={{ flex: 1, overflow: 'auto', display: 'flex', flexDirection: 'column' }}>
        {isTraining ? (
          <EmptyState message="Simulation active. Stop to analyze." />
        ) : isComputing ? (
          <EmptyState message="Computing analysis timeline..." />
        ) : hasCharts ? (
          <div style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
            <ChartNavigator />
            <div style={{ flex: 1, overflowY: 'auto' }}>
              <ChartDisplay />
            </div>
          </div>
        ) : (
          <EmptyState message="No analysis data. Complete a simulation first." />
        )}
      </div>
    </div>
  );
};
