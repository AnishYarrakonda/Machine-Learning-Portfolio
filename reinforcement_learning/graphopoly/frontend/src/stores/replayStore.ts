import { create } from 'zustand';
import { EpisodeJSON } from '../types/episode';
import { useGraphStore } from './graphStore';

interface ReplayState {
  episodeData: EpisodeJSON | null;
  currentStep: number;
  totalSteps: number;
  isPlaying: boolean;
  playbackSpeed: number;

  loadEpisode: (data: EpisodeJSON) => void;
  setStep: (step: number) => void;
  stepForward: () => void;
  stepBack: () => void;
  jumpForward: (n: number) => void;
  jumpBack: (n: number) => void;
  play: () => void;
  pause: () => void;
  reset: () => void;
}

export const useReplayStore = create<ReplayState>((set) => ({
  episodeData: null,
  currentStep: 0,
  totalSteps: 0,
  isPlaying: false,
  playbackSpeed: 1,

  loadEpisode: (data) => {
    if (!data) return;
    try {
      const graphSection = data.graph;
      if (graphSection) {
        // Build graph data from episode
        const graphData = {
          num_nodes: graphSection.nodes?.length ?? 0,
          edges: graphSection.edges ?? [],
          ownership: graphSection.ownership ?? {},
          destinations: graphSection.destinations ?? {},
          starting_positions: graphSection.starting_positions ?? {},
        };

        // Prefer to keep the user's current layout (preserves their arrangement)
        // Only fall back to episode's saved positions if no current layout
        const currentLayout = useGraphStore.getState().layout;
        let layout: Record<number, [number, number]>;

        if (currentLayout && Object.keys(currentLayout).length > 0) {
          layout = currentLayout;
        } else if (graphSection.nodes && graphSection.nodes.length > 0) {
          layout = {};
          for (const node of graphSection.nodes) {
            if (node.position && node.position.length === 2) {
              layout[node.id] = node.position;
            }
          }
        } else {
          layout = {};
        }

        if (graphData.num_nodes > 0) {
          useGraphStore.getState().loadGraph(graphData, layout);
        }
      }
    } catch (err) {
      console.error('[replay] loadEpisode graph load failed:', err);
    }

    const trajectory = data.trajectory ?? [];
    set({
      episodeData: data,
      currentStep: 0,
      totalSteps: trajectory.length,
      isPlaying: false,
    });
  },

  setStep: (step) => set((state) => ({
    currentStep: Math.max(0, Math.min(step, state.totalSteps - 1))
  })),

  stepForward: () => set((state) => ({
    currentStep: Math.min(state.currentStep + 1, Math.max(0, state.totalSteps - 1))
  })),

  stepBack: () => set((state) => ({
    currentStep: Math.max(state.currentStep - 1, 0)
  })),

  jumpForward: (n) => set((state) => ({
    currentStep: Math.min(state.currentStep + n, Math.max(0, state.totalSteps - 1))
  })),

  jumpBack: (n) => set((state) => ({
    currentStep: Math.max(state.currentStep - n, 0)
  })),

  play: () => set((state) => {
    if (state.totalSteps === 0) return state;
    if (state.currentStep >= state.totalSteps - 1) {
      return { isPlaying: true, currentStep: 0 };
    }
    return { isPlaying: true };
  }),

  pause: () => set({ isPlaying: false }),

  reset: () => set({ currentStep: 0, isPlaying: false })
}));
