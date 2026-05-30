import React, { useEffect, useState } from 'react';
import { AppShell } from './components/layout/AppShell';
import { GraphCanvas } from './components/graph/GraphCanvas';
import { SettingsPanel } from './components/panels/SettingsPanel';
import { LiveStatsOverlay } from './components/panels/LiveStatsOverlay';
import { AnalysisReplayPanel } from './components/panels/AnalysisReplayPanel';
import { OnboardingOverlay } from './components/onboarding/OnboardingOverlay';
import { useKeyboard } from './hooks/useKeyboard';
import { usePlayback } from './hooks/usePlayback';
import { useSimulationPlayback } from './hooks/useSimulationPlayback';
import { api } from './api/client';
import { useConfigStore } from './stores/configStore';
import { useUIStore } from './stores/uiStore';

export default function App() {
  useKeyboard();
  usePlayback();
  useSimulationPlayback();
  const loadConfig = useConfigStore(s => s.loadConfig);
  const isSidebarCollapsed = useUIStore(s => s.isSidebarCollapsed);
  const [bottomPanelHeight, setBottomPanelHeight] = useState(38);
  const [sidebarWidth, setSidebarWidth] = useState(300);

  useEffect(() => {
    api.config.get().then(config => loadConfig(config)).catch(e => console.error('Config fetch failed', e));
  }, [loadConfig]);

  const startResizingBottom = (mouseDownEvent: React.MouseEvent) => {
    const startY = mouseDownEvent.clientY;
    const startHeight = bottomPanelHeight;
    const onMouseMove = (e: MouseEvent) => {
      const delta = startY - e.clientY;
      setBottomPanelHeight(Math.min(Math.max(startHeight + (delta / window.innerHeight) * 100, 15), 85));
    };
    const onMouseUp = () => {
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    };
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  };

  const startResizingSidebar = (mouseDownEvent: React.MouseEvent) => {
    mouseDownEvent.preventDefault();
    const startX = mouseDownEvent.clientX;
    const startWidth = sidebarWidth;
    const onMouseMove = (e: MouseEvent) => {
      setSidebarWidth(Math.min(Math.max(startWidth + (e.clientX - startX), 180), 600));
    };
    const onMouseUp = () => {
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    };
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  };

  const effectiveSidebarWidth = isSidebarCollapsed ? 56 : sidebarWidth;

  return (
    <AppShell>
      <OnboardingOverlay />

      {/* ── SIDEBAR ─────────────────────────────────── */}
      <aside style={{
        width: effectiveSidebarWidth,
        minWidth: effectiveSidebarWidth,
        height: '100%',
        borderRight: '1px solid var(--color-border)',
        background: 'var(--color-bg-elevated)',
        display: 'flex',
        flexDirection: 'column',
        zIndex: 10,
        overflowY: 'auto',
        position: 'relative',
        flexShrink: 0,
        transition: isSidebarCollapsed ? 'width var(--transition-slow)' : 'none',
      }}>
        <SettingsPanel />
        {!isSidebarCollapsed && (
          <div
            onMouseDown={startResizingSidebar}
            style={{
              position: 'absolute', top: 0, right: 0,
              width: 5, height: '100%', cursor: 'ew-resize', zIndex: 20,
            }}
            onMouseEnter={e => { e.currentTarget.style.background = 'var(--color-accent)'; e.currentTarget.style.opacity = '0.4'; }}
            onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.opacity = '1'; }}
          />
        )}
      </aside>

      {/* ── MAIN CONTENT ────────────────────────────── */}
      <main style={{
        flex: 1, height: '100%', display: 'flex', flexDirection: 'column',
        position: 'relative', background: 'var(--color-bg)',
      }}>
        {/* GRAPH CANVAS — fills remaining space, overlay mounted inside */}
        <section style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
          <GraphCanvas />
          <LiveStatsOverlay />
        </section>

        {/* RESIZE HANDLE */}
        <div
          onMouseDown={startResizingBottom}
          style={{ height: 10, cursor: 'ns-resize', flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
          onMouseEnter={e => { const b = e.currentTarget.querySelector('.drag-indicator') as HTMLElement; if (b) b.style.background = 'var(--color-accent)'; }}
          onMouseLeave={e => { const b = e.currentTarget.querySelector('.drag-indicator') as HTMLElement; if (b) b.style.background = 'var(--color-text-muted)'; }}
        >
          <div className="drag-indicator" style={{ width: 48, height: 4, borderRadius: 2, background: 'var(--color-text-muted)', transition: 'background var(--transition-fast)' }} />
        </div>

        {/* BOTTOM PANEL — Analysis only */}
        <section style={{
          height: `${bottomPanelHeight}vh`,
          background: 'var(--color-bg-surface)',
          borderTop: '1px solid var(--color-border)',
          display: 'flex',
          flexDirection: 'column',
          zIndex: 10,
          flexShrink: 0,
        }}>
          <AnalysisReplayPanel />
        </section>
      </main>
    </AppShell>
  );
}
