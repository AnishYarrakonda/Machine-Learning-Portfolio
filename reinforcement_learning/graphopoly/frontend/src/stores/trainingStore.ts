import { create } from 'zustand';
import { WSMessage, StepHistoryEntry, AgentDict } from '../types/websocket';
import { GraphData } from '../types/graph';
import { EpisodeJSON } from '../types/episode';
import { adaptToEpisodeJSON } from '../lib/episodeAdapter';

interface TrainingState {
  isTraining: boolean;
  isPaused: boolean;
  runName: string;
  currentEpisode: number;
  totalEpisodes: number;
  episodeRewards: number[];
  episodeTrips: number[];
  losses: Record<number, { policy_loss: number; value_loss: number; entropy: number }>;
  agentDetails: AgentDict[];
  stepHistory: StepHistoryEntry[];
  currentPrices: Record<string, number>;
  latestEpisodeData: EpisodeJSON | null;
  latestGraphData: GraphData | null;
  latestLayout: Record<number, [number, number]> | null;

  // Step-by-step animation state
  simAnimStep: number;
  /** Queue of step-histories waiting to be animated. */
  stepHistoryQueue: StepHistoryEntry[][];

  startTraining: (runName?: string) => void;
  stopTraining: () => void;
  pauseTraining: () => void;
  resumeTraining: () => void;
  resetEpisodeData: () => void;
  handleEpisodeUpdate: (msg: Extract<WSMessage, { type: 'episode_update' }>) => void;
  handleTrainingComplete: () => void;
  setLatestEpisodeData: (data: EpisodeJSON) => void;

  // Animation controls
  advanceSimStep: () => boolean; // returns false if at end
  loadNextQueuedEpisode: () => boolean; // returns false if queue empty
}

export const useTrainingStore = create<TrainingState>((set, get) => ({
  isTraining: false,
  isPaused: false,
  runName: '',
  currentEpisode: 0,
  totalEpisodes: 0,
  episodeRewards: [],
  episodeTrips: [],
  losses: {},
  agentDetails: [],
  stepHistory: [],
  currentPrices: {},
  latestEpisodeData: null,
  latestGraphData: null,
  latestLayout: null,
  simAnimStep: 0,
  stepHistoryQueue: [],

  startTraining: (runName = '') => set({
    isTraining: true, isPaused: false, runName, currentEpisode: 0, totalEpisodes: 0,
    episodeRewards: [], episodeTrips: [], losses: {}, agentDetails: [],
    stepHistory: [], currentPrices: {}, simAnimStep: 0, stepHistoryQueue: [],
    latestEpisodeData: null, latestGraphData: null, latestLayout: null,
  }),
  stopTraining: () => set({ isTraining: false, isPaused: false }),
  pauseTraining: () => set({ isPaused: true }),
  resumeTraining: () => set({ isPaused: false }),
  resetEpisodeData: () => set({
    stepHistory: [], currentPrices: {}, agentDetails: [],
    simAnimStep: 0, stepHistoryQueue: [], latestEpisodeData: null,
  }),

  handleEpisodeUpdate: (msg) => {
    const d = msg.data;
    const state = get();

    // If we're currently animating through a step history, queue this one
    if (state.stepHistory.length > 0 && state.simAnimStep < state.stepHistory.length - 1) {
      set({
        isTraining: true,
        currentEpisode: d.episode,
        totalEpisodes: d.total_episodes,
        episodeRewards: d.episode_rewards,
        episodeTrips: d.episode_trips,
        losses: d.losses,
        agentDetails: d.agent_details,
        currentPrices: d.env_snapshot.prices,
        latestGraphData: d.graph_data,
        latestLayout: d.layout,
        stepHistoryQueue: [...state.stepHistoryQueue, d.step_history].slice(-3), // keep max 3 queued
      });
    } else {
      // Not currently animating — load this episode directly
      set({
        isTraining: true,
        currentEpisode: d.episode,
        totalEpisodes: d.total_episodes,
        episodeRewards: d.episode_rewards,
        episodeTrips: d.episode_trips,
        losses: d.losses,
        agentDetails: d.agent_details,
        stepHistory: d.step_history,
        currentPrices: d.env_snapshot.prices,
        latestGraphData: d.graph_data,
        latestLayout: d.layout,
        simAnimStep: 0,
      });
    }
  },

  handleTrainingComplete: () => {
    try {
      set({ isTraining: false, isPaused: false });
    } catch (e) {
      console.error('[training] setState failed:', e);
    }
    fetch('/api/episode/latest')
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(async (res) => {
        if (res?.status === 'ok' && res.data) {
          try {
            const episodeData = adaptToEpisodeJSON(res.data);
            const { useReplayStore } = await import('./replayStore');
            useReplayStore.getState().loadEpisode(episodeData);
          } catch (err) {
            console.error('[training] Failed to load episode into replay:', err);
          }
        }
      })
      .catch((err) => {
        console.warn('[training] Could not fetch latest episode:', err);
      });
  },

  setLatestEpisodeData: (data) => set({ latestEpisodeData: data }),

  advanceSimStep: () => {
    const state = get();
    if (state.simAnimStep >= state.stepHistory.length - 1) return false;
    set({ simAnimStep: state.simAnimStep + 1 });
    return true;
  },

  loadNextQueuedEpisode: () => {
    const state = get();
    if (state.stepHistoryQueue.length === 0) return false;
    const [next, ...rest] = state.stepHistoryQueue;
    set({ stepHistory: next, simAnimStep: 0, stepHistoryQueue: rest });
    return true;
  },
}));
