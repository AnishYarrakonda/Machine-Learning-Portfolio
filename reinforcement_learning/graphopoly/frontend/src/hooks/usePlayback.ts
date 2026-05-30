import { useEffect } from 'react';
import { useReplayStore } from '../stores/replayStore';
import { useUIStore } from '../stores/uiStore';

/**
 * Drives replay playback.
 *
 * Normal mode (animSpeed > 0): setInterval fires every animSpeed ms, advancing one step.
 * Turbo mode (animSpeed === 0): requestAnimationFrame advances steps in batches of 20
 *   per frame — plays through the episode as fast as possible with no animation delay.
 *   Agent dots are hidden (GraphRenderer checks animSpeed).
 */
export function usePlayback() {
  const isPlaying = useReplayStore(s => s.isPlaying);
  const totalSteps = useReplayStore(s => s.totalSteps);
  const stepForward = useReplayStore(s => s.stepForward);
  const pause = useReplayStore(s => s.pause);
  const animSpeed = useUIStore(s => s.animSpeed);

  useEffect(() => {
    if (!isPlaying) return;

    if (animSpeed === 0) {
      // ── Turbo mode: advance as many steps as possible per animation frame ──
      let rafId: number;
      const STEPS_PER_FRAME = 20; // batch size — high enough to be "instant"

      const runFrame = () => {
        for (let i = 0; i < STEPS_PER_FRAME; i++) {
          const { currentStep, totalSteps: total } = useReplayStore.getState();
          if (currentStep >= total - 1) {
            pause();
            return;
          }
          stepForward();
        }
        rafId = requestAnimationFrame(runFrame);
      };

      rafId = requestAnimationFrame(runFrame);
      return () => cancelAnimationFrame(rafId);
    }

    // ── Normal mode: one step per interval ──────────────────────────────────
    const id = setInterval(() => {
      const current = useReplayStore.getState().currentStep;
      if (current >= totalSteps - 1) {
        pause();
      } else {
        stepForward();
      }
    }, animSpeed);

    return () => clearInterval(id);
  }, [isPlaying, totalSteps, animSpeed, stepForward, pause]);
}
