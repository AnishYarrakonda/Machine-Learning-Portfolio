import React, { useState, useEffect, useCallback, useRef } from 'react';

// ── Step definitions ─────────────────────────────────────────────────────────
// `anchor` is a CSS selector for the element the card should point to.
// `placement` controls which side of the anchor the card appears on.
// `scrollTo` if true, scrolls the anchor into view before positioning.

type Placement = 'right' | 'left' | 'bottom' | 'top' | 'center';

interface TourStep {
  title: string;
  body: string;
  anchor: string | null;      // CSS selector, null = centered on screen
  placement: Placement;
  scrollTo?: boolean;
}

const STEPS: TourStep[] = [
  {
    title: 'Welcome to Graphopoly',
    body: 'A multi-agent reinforcement learning playground where agents navigate graphs, own nodes, and set prices to maximize reward.',
    anchor: null,
    placement: 'center',
  },
  {
    title: 'Generate a Graph',
    body: 'Set the number of nodes, agents, and destinations, then hit this button to build a random graph topology.',
    anchor: '[data-tour="generate-btn"]',
    placement: 'right',
    scrollTo: true,
  },
  {
    title: 'Build Custom Graphs',
    body: 'Use the toolbar to manually place nodes, connect them with edges, assign owners, and mark destinations for each agent.',
    anchor: '[data-tour="toolbar"]',
    placement: 'bottom',
  },
  {
    title: 'The Playground',
    body: 'Drag nodes to rearrange the layout at any time. Zoom with scroll, pan by dragging the background. Your layout persists across simulations.',
    anchor: '[data-tour="playground"]',
    placement: 'bottom',
  },
  {
    title: 'Start a Simulation',
    body: 'Hit Start Simulation to watch agents move across the graph in real time. They\'ll set prices, collect taxes, and race to destinations.',
    anchor: '[data-tour="start-sim"]',
    placement: 'right',
    scrollTo: true,
  },
  {
    title: 'Live Stats',
    body: 'Switch to the Live Stats tab during a simulation to see real-time per-agent rewards, node prices, visit counts, and system metrics.',
    anchor: '[data-tour="tab-livestats"]',
    placement: 'bottom',
    scrollTo: true,
  },
  {
    title: 'Analysis & Replay',
    body: 'After stopping a simulation, switch to Analysis. Use the replay slider to scrub through each step, and explore 26 charts across agents, nodes, economy, and system categories.',
    anchor: '[data-tour="tab-analysis"]',
    placement: 'bottom',
    scrollTo: true,
  },
];

// ── Positioning logic ────────────────────────────────────────────────────────

const CARD_WIDTH = 380;
const CARD_GAP = 16;    // gap between anchor edge and card
const SPOTLIGHT_PAD = 8; // padding around the spotlight cutout

interface CardPos {
  top: number;
  left: number;
}

interface SpotlightRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

function computePosition(
  anchorRect: DOMRect | null,
  placement: Placement,
): { card: CardPos; spotlight: SpotlightRect | null } {
  // Center on screen if no anchor
  if (!anchorRect) {
    return {
      card: {
        top: window.innerHeight * 0.35,
        left: (window.innerWidth - CARD_WIDTH) / 2,
      },
      spotlight: null,
    };
  }

  const spotlight: SpotlightRect = {
    x: anchorRect.left - SPOTLIGHT_PAD,
    y: anchorRect.top - SPOTLIGHT_PAD,
    w: anchorRect.width + SPOTLIGHT_PAD * 2,
    h: anchorRect.height + SPOTLIGHT_PAD * 2,
  };

  let top: number;
  let left: number;

  switch (placement) {
    case 'right':
      top = anchorRect.top;
      left = anchorRect.right + CARD_GAP;
      // If card would overflow right edge, flip to left
      if (left + CARD_WIDTH > window.innerWidth - 20) {
        left = anchorRect.left - CARD_WIDTH - CARD_GAP;
      }
      break;
    case 'left':
      top = anchorRect.top;
      left = anchorRect.left - CARD_WIDTH - CARD_GAP;
      if (left < 20) {
        left = anchorRect.right + CARD_GAP;
      }
      break;
    case 'bottom':
      top = anchorRect.bottom + CARD_GAP;
      left = anchorRect.left + (anchorRect.width - CARD_WIDTH) / 2;
      // Clamp horizontal
      left = Math.max(20, Math.min(left, window.innerWidth - CARD_WIDTH - 20));
      // If card would overflow bottom, flip to top
      if (top + 200 > window.innerHeight) {
        top = anchorRect.top - 200 - CARD_GAP;
      }
      break;
    case 'top':
      top = anchorRect.top - 200 - CARD_GAP;
      left = anchorRect.left + (anchorRect.width - CARD_WIDTH) / 2;
      left = Math.max(20, Math.min(left, window.innerWidth - CARD_WIDTH - 20));
      if (top < 20) {
        top = anchorRect.bottom + CARD_GAP;
      }
      break;
    default: // center
      top = window.innerHeight * 0.35;
      left = (window.innerWidth - CARD_WIDTH) / 2;
      break;
  }

  // Clamp vertical
  top = Math.max(20, Math.min(top, window.innerHeight - 300));

  return { card: { top, left }, spotlight };
}

// ── SVG spotlight overlay ────────────────────────────────────────────────────

const SpotlightOverlay: React.FC<{ rect: SpotlightRect | null; onClick: () => void }> = ({ rect, onClick }) => {
  const W = window.innerWidth;
  const H = window.innerHeight;

  if (!rect) {
    // Simple dim overlay, no cutout
    return (
      <div
        onClick={onClick}
        style={{
          position: 'fixed', inset: 0, zIndex: 9998,
          background: 'rgba(0,0,0,0.6)',
          pointerEvents: 'auto',
        }}
      />
    );
  }

  // SVG overlay with a rectangular cutout
  const r = 6; // border-radius of cutout
  return (
    <svg
      onClick={onClick}
      style={{ position: 'fixed', inset: 0, zIndex: 9998, pointerEvents: 'auto', cursor: 'pointer' }}
      width={W}
      height={H}
      viewBox={`0 0 ${W} ${H}`}
    >
      <defs>
        <mask id="tour-spotlight-mask">
          <rect x={0} y={0} width={W} height={H} fill="white" />
          <rect x={rect.x} y={rect.y} width={rect.w} height={rect.h} rx={r} ry={r} fill="black" />
        </mask>
      </defs>
      <rect
        x={0} y={0} width={W} height={H}
        fill="rgba(0,0,0,0.6)"
        mask="url(#tour-spotlight-mask)"
      />
      {/* Subtle glow ring around the cutout */}
      <rect
        x={rect.x} y={rect.y} width={rect.w} height={rect.h}
        rx={r} ry={r}
        fill="none"
        stroke="rgba(255,255,255,0.25)"
        strokeWidth={2}
      />
    </svg>
  );
};

// ── Main overlay component ───────────────────────────────────────────────────

export const OnboardingOverlay: React.FC = () => {
  const [step, setStep] = useState(0);
  const [done, setDone] = useState(() => !!localStorage.getItem('graphopoly_onboarded_v3'));
  const [cardPos, setCardPos] = useState<CardPos>({ top: 0, left: 0 });
  const [spotlight, setSpotlight] = useState<SpotlightRect | null>(null);
  const [visible, setVisible] = useState(false); // for entrance animation
  const cardRef = useRef<HTMLDivElement>(null);

  const current = STEPS[step];
  const isLast = step === STEPS.length - 1;

  // Recompute position whenever step changes or window resizes
  const updatePosition = useCallback(() => {
    const cur = STEPS[step];
    let anchorRect: DOMRect | null = null;

    if (cur.anchor) {
      const el = document.querySelector(cur.anchor);
      if (el) {
        if (cur.scrollTo) {
          el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
        anchorRect = el.getBoundingClientRect();
      }
    }

    const { card, spotlight: sp } = computePosition(anchorRect, cur.placement);
    setCardPos(card);
    setSpotlight(sp);
  }, [step]);

  // Position on mount and step change
  useEffect(() => {
    if (done) return;
    setVisible(false);
    // Small delay to allow scrolling to complete before measuring
    const scrollDelay = STEPS[step].scrollTo ? 350 : 50;
    const timer = setTimeout(() => {
      updatePosition();
      setVisible(true);
    }, scrollDelay);
    return () => clearTimeout(timer);
  }, [step, done, updatePosition]);

  // Reposition on window resize
  useEffect(() => {
    if (done) return;
    const handleResize = () => updatePosition();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [done, updatePosition]);

  if (done) return null;

  const next = () => {
    if (isLast) {
      localStorage.setItem('graphopoly_onboarded_v3', '1');
      setDone(true);
    } else {
      setStep(s => s + 1);
    }
  };

  const prev = () => setStep(s => Math.max(0, s - 1));

  const skip = () => {
    localStorage.setItem('graphopoly_onboarded_v3', '1');
    setDone(true);
  };

  return (
    <>
      {/* Spotlight / dim overlay */}
      <SpotlightOverlay rect={spotlight} onClick={next} />

      {/* Tutorial card */}
      <div
        ref={cardRef}
        style={{
          position: 'fixed',
          top: cardPos.top,
          left: cardPos.left,
          width: CARD_WIDTH,
          zIndex: 9999,
          background: 'var(--color-bg-elevated)',
          border: '1px solid var(--color-border-active)',
          borderLeft: '3px solid var(--color-accent)',
          boxShadow: '0 12px 48px rgba(0,0,0,0.7)',
          padding: '28px 28px 24px',
          fontFamily: "'Inter', sans-serif",
          pointerEvents: 'auto',
          opacity: visible ? 1 : 0,
          transform: visible ? 'translateY(0)' : 'translateY(8px)',
          transition: 'opacity 0.3s ease, transform 0.3s ease, top 0.35s ease, left 0.35s ease',
        }}
      >
        {/* Step indicator dots */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: 20,
        }}>
          <div style={{ display: 'flex', gap: 6 }}>
            {STEPS.map((_, i) => (
              <div
                key={i}
                onClick={() => setStep(i)}
                style={{
                  width: i === step ? 20 : 6,
                  height: 6,
                  borderRadius: 3,
                  background: i === step ? 'rgba(255,255,255,0.85)' : i < step ? 'rgba(255,255,255,0.35)' : 'rgba(255,255,255,0.12)',
                  cursor: 'pointer',
                  transition: 'all 0.25s ease',
                }}
              />
            ))}
          </div>
          <span style={{
            fontSize: 11,
            color: 'rgba(255,255,255,0.3)',
            letterSpacing: '0.1em',
            fontWeight: 500,
          }}>
            {step + 1} / {STEPS.length}
          </span>
        </div>

        {/* Content */}
        <h3 style={{
          fontSize: 17,
          fontWeight: 600,
          color: 'var(--color-text)',
          marginBottom: 12,
          lineHeight: 1.3,
        }}>
          {current.title}
        </h3>
        <p style={{
          fontSize: 13,
          color: 'var(--color-text-secondary)',
          lineHeight: 1.7,
          marginBottom: 24,
        }}>
          {current.body}
        </p>

        {/* Actions */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <button
            onClick={skip}
            style={{
              background: 'none',
              border: 'none',
              color: 'rgba(255,255,255,0.2)',
              fontSize: 12,
              cursor: 'pointer',
              padding: 0,
              fontFamily: "'Inter', sans-serif",
              letterSpacing: '0.05em',
              transition: 'color 0.2s',
            }}
            onMouseEnter={e => (e.currentTarget.style.color = 'rgba(255,255,255,0.45)')}
            onMouseLeave={e => (e.currentTarget.style.color = 'rgba(255,255,255,0.2)')}
          >
            Skip tutorial
          </button>

          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            {step > 0 && (
              <button
                onClick={prev}
                style={{
                  background: 'none',
                  border: '1px solid rgba(255,255,255,0.12)',
                  color: 'rgba(255,255,255,0.5)',
                  fontSize: 12,
                  padding: '8px 18px',
                  cursor: 'pointer',
                  fontFamily: "'Inter', sans-serif",
                  transition: 'all 0.2s',
                }}
                onMouseEnter={e => { e.currentTarget.style.borderColor = 'rgba(255,255,255,0.25)'; e.currentTarget.style.color = 'rgba(255,255,255,0.75)'; }}
                onMouseLeave={e => { e.currentTarget.style.borderColor = 'rgba(255,255,255,0.12)'; e.currentTarget.style.color = 'rgba(255,255,255,0.5)'; }}
              >
                Back
              </button>
            )}
            <button
              onClick={next}
              style={{
                background: 'var(--color-accent)',
                border: 'none',
                color: '#fff',
                fontSize: 12,
                fontWeight: 600,
                padding: '8px 22px',
                cursor: 'pointer',
                fontFamily: "'Inter', sans-serif",
                letterSpacing: '0.02em',
                borderRadius: 4,
                transition: 'background 0.15s',
              }}
              onMouseEnter={e => (e.currentTarget.style.background = 'var(--color-accent-dim)')}
              onMouseLeave={e => (e.currentTarget.style.background = 'var(--color-accent)')}
            >
              {isLast ? 'Get started' : 'Next'}
            </button>
          </div>
        </div>
      </div>
    </>
  );
};
