import { create } from 'zustand';
import { AGENT_COLORS } from '../lib/chartTheme';

export type UIMode = 'view' | 'build_node' | 'build_edge' | 'build_owner' | 'build_dest';

export const DEFAULT_AGENT_COLORS = AGENT_COLORS;

interface UIState {
  // Visuals (toggles)
  showIds: boolean;
  showPrices: boolean;
  showDests: boolean;
  showAgents: boolean;

  // Node size
  nodeSize: number;

  // Agent colors (index = agent id)
  agentColors: string[];

  // Animation
  animSpeed: number;

  // Interaction state
  mode: UIMode;
  selectedAgent: string | null;
  selectedNode: string | null;

  // Sidebar
  isSidebarCollapsed: boolean;

  // Actions
  toggleShowIds: () => void;
  toggleShowPrices: () => void;
  toggleShowDests: () => void;
  toggleShowAgents: () => void;
  setNodeSize: (size: number) => void;
  setAgentColor: (agentIdx: number, color: string) => void;
  resetAgentColors: () => void;
  setAnimSpeed: (val: number) => void;
  setMode: (mode: UIMode) => void;
  setSelectedAgent: (agentId: string | null) => void;
  setSelectedNode: (nodeId: string | null) => void;
  toggleSidebar: () => void;
}

export const useUIStore = create<UIState>((set) => ({
  showIds: true,
  showPrices: true,
  showDests: true,
  showAgents: true,
  nodeSize: 30,
  agentColors: [...DEFAULT_AGENT_COLORS],
  animSpeed: 250,
  mode: 'view',
  selectedAgent: null,
  selectedNode: null,
  isSidebarCollapsed: false,

  toggleShowIds: () => set((state) => ({ showIds: !state.showIds })),
  toggleShowPrices: () => set((state) => ({ showPrices: !state.showPrices })),
  toggleShowDests: () => set((state) => ({ showDests: !state.showDests })),
  toggleShowAgents: () => set((state) => ({ showAgents: !state.showAgents })),
  setNodeSize: (size) => set({ nodeSize: size }),
  setAgentColor: (agentIdx, color) => set((state) => {
    const next = [...state.agentColors];
    while (next.length <= agentIdx) next.push(DEFAULT_AGENT_COLORS[next.length % DEFAULT_AGENT_COLORS.length]);
    next[agentIdx] = color;
    return { agentColors: next };
  }),
  resetAgentColors: () => set({ agentColors: [...DEFAULT_AGENT_COLORS] }),
  setAnimSpeed: (val) => set({ animSpeed: val }),
  setMode: (mode) => set({ mode }),
  setSelectedAgent: (agentId) => set({ selectedAgent: agentId }),
  setSelectedNode: (nodeId) => set({ selectedNode: nodeId }),
  toggleSidebar: () => set((state) => ({ isSidebarCollapsed: !state.isSidebarCollapsed })),
}));
