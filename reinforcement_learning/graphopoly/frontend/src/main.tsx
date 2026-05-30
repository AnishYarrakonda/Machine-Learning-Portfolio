import React, { Component, StrictMode, type ReactNode } from 'react'
import { createRoot } from 'react-dom/client'
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement,
  BarElement, ArcElement, RadialLinearScale, Filler, Tooltip, Legend,
} from 'chart.js';
import annotationPlugin from 'chartjs-plugin-annotation';
import './styles/reset.css'
import './styles/tokens.css'
import './styles/globals.css'
import App from './App.tsx'

ChartJS.register(
  CategoryScale, LinearScale, PointElement, LineElement,
  BarElement, ArcElement, RadialLinearScale, Filler, Tooltip, Legend,
  annotationPlugin,
);

ChartJS.defaults.color = 'rgba(255,255,255,0.5)';
ChartJS.defaults.font.family = "'Inter', 'SF Mono', monospace";
ChartJS.defaults.font.size = 11;

// ── Global error boundary — prevents blank screen on render errors ────────────
class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error('[ErrorBoundary] Caught render error:', error, info);
  }

  render() {
    if (this.state.error) {
      return (
        <div style={{
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
          height: '100vh', background: '#111', color: '#fff', fontFamily: "'Inter', sans-serif",
          gap: 16, padding: 40,
        }}>
          <div style={{ fontSize: 28, fontWeight: 300, letterSpacing: '0.05em', color: 'rgba(255,255,255,0.8)' }}>
            Something went wrong
          </div>
          <div style={{ fontSize: 13, color: 'rgba(255,255,255,0.4)', maxWidth: 500, textAlign: 'center', lineHeight: 1.6 }}>
            {(this.state.error as Error).message}
          </div>
          <button
            onClick={() => window.location.reload()}
            style={{
              marginTop: 8, padding: '10px 28px', background: 'none',
              border: '1px solid rgba(255,255,255,0.2)', color: 'rgba(255,255,255,0.6)',
              fontFamily: 'inherit', fontSize: 13, cursor: 'pointer',
              letterSpacing: '0.1em', textTransform: 'uppercase',
            }}
          >
            Reload
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

const el = document.getElementById('root');
if (el) {
  createRoot(el).render(
    <StrictMode>
      <ErrorBoundary>
        <App />
      </ErrorBoundary>
    </StrictMode>,
  );
} else {
  console.error('[main] No #root element found');
}
