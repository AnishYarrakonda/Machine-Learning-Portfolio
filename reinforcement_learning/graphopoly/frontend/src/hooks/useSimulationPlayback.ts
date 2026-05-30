import { useEffect } from 'react';
import { useTrainingStore } from '../stores/trainingStore';
import { useUIStore } from '../stores/uiStore';

/**
 * Drives step-by-step animation during live simulation.
 *
 * Each episode arrives from the backend as a complete step_history (100 steps).
 * This hook advances through the steps one at a time at animSpeed, so agents
 * move smoothly instead of teleporting to the final position.
 *
 * When the current episode finishes animating, it loads the next queued episode.
 * Turbo mode (animSpeed === 0) skips animation entirely — jumps to the last step.
 */
export function useSimulationPlayback() {
  const isTraining = useTrainingStore(s => s.isTraining);
  const isPaused = useTrainingStore(s => s.isPaused);
  const stepHistory = useTrainingStore(s => s.stepHistory);
  const simAnimStep = useTrainingStore(s => s.simAnimStep);
  const advanceSimStep = useTrainingStore(s => s.advanceSimStep);
  const loadNextQueuedEpisode = useTrainingStore(s => s.loadNextQueuedEpisode);
  const animSpeed = useUIStore(s => s.animSpeed);

  useEffect(() => {
    if (!isTraining || isPaused || stepHistory.length === 0) return;

    // Turbo mode: jump to last step immediately
    if (animSpeed === 0) {
      // Skip to end of this episode, then load next queued
      const state = useTrainingStore.getState();
      if (state.simAnimStep < state.stepHistory.length - 1) {
        useTrainingStore.setState({ simAnimStep: state.stepHistory.length - 1 });
      }
      return;
    }

    // Normal mode: advance one step at a time
    const id = setInterval(() => {
      const advanced = advanceSimStep();
      if (!advanced) {
        // Current episode animation is done — try loading the next queued one
        const loaded = loadNextQueuedEpisode();
        if (!loaded) {
          // No more queued episodes — we'll wait for the next WS message
          // (the interval keeps running, but advanceSimStep is a no-op at end)
        }
      }
    }, animSpeed);

    return () => clearInterval(id);
  }, [isTraining, isPaused, stepHistory, animSpeed, simAnimStep, advanceSimStep, loadNextQueuedEpisode]);
}
