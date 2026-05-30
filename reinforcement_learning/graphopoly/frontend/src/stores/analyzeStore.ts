import { create } from 'zustand';
import { TimelineEntry, EpisodeJSON } from '../types/episode';
import type { ChartCategory } from '../lib/chartRegistry';

interface AnalyzeState {
  episodeData: EpisodeJSON | null;
  timeline: TimelineEntry[];
  activeCategory: ChartCategory;
  activeChartId: string;
  selectedAgents: string[];
  selectedNodes: string[];

  setAnalysisData: (data: EpisodeJSON, timeline: TimelineEntry[]) => void;
  clearAnalysis: () => void;
  setCategory: (cat: ChartCategory) => void;
  setChart: (chartId: string) => void;
  toggleAgent: (agentId: string) => void;
  toggleNode: (nodeId: string) => void;
}

export const useAnalyzeStore = create<AnalyzeState>((set) => ({
  episodeData: null,
  timeline: [],
  activeCategory: 'agents',
  activeChartId: 'agent-cumulative-reward',
  selectedAgents: [],
  selectedNodes: [],

  setAnalysisData: (data, timeline) => set({
    episodeData: data,
    timeline,
    selectedAgents: [],
    selectedNodes: [],
  }),

  clearAnalysis: () => set({
    episodeData: null,
    timeline: [],
    selectedAgents: [],
    selectedNodes: [],
  }),

  setCategory: (cat) => set({ activeCategory: cat }),

  setChart: (chartId) => set({ activeChartId: chartId }),

  toggleAgent: (agentId) => set((state) => ({
    selectedAgents: state.selectedAgents.includes(agentId)
      ? state.selectedAgents.filter(id => id !== agentId)
      : [...state.selectedAgents, agentId],
  })),

  toggleNode: (nodeId) => set((state) => ({
    selectedNodes: state.selectedNodes.includes(nodeId)
      ? state.selectedNodes.filter(id => id !== nodeId)
      : [...state.selectedNodes, nodeId],
  })),
}));
