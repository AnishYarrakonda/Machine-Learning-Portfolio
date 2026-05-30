import { create } from 'zustand';
import { GraphData } from '../types/graph';

interface GraphState {
  data: GraphData | null;
  layout: Record<number, [number, number]> | null;

  // Actions
  loadGraph: (data: GraphData, layout: Record<number, [number, number]>) => void;
  updateOwnership: (data: GraphData) => void; // update ownership/edges without touching layout
  clearAll: () => void;
  addNode: (id: number, position: [number, number]) => void;
  addEdge: (source: number, target: number) => void;
  setOwner: (nodeId: number, agentId: number) => void;
  toggleDestination: (agentId: string, nodeId: number) => void;
  updateNodePosition: (nodeId: number, position: [number, number]) => void;
  setLayout: (layout: Record<number, [number, number]>) => void;
}

export const useGraphStore = create<GraphState>((set) => ({
  data: null,
  layout: null,

  loadGraph: (data, layout) => {
    const normalizedLayout: Record<number, [number, number]> = {};
    for (const [key, val] of Object.entries(layout)) {
      normalizedLayout[Number(key)] = val;
    }
    return set({ data, layout: normalizedLayout });
  },

  clearAll: () => set({ data: null, layout: null }),

  addNode: (id, position) => set((state) => {
    if (!state.data || !state.layout) {
      return {
        data: {
          num_nodes: 1,
          edges: [],
          ownership: { [id]: 0 }, // all nodes must be owned — default to agent 0
          destinations: {},
          starting_positions: {},
        },
        layout: { [id]: position }
      };
    }
    return {
      data: {
        ...state.data,
        num_nodes: state.data.num_nodes + 1,
        ownership: { ...state.data.ownership, [id]: 0 }, // default owner
      },
      layout: {
        ...state.layout,
        [id]: position
      }
    };
  }),

  addEdge: (source, target) => set((state) => {
    if (!state.data) return state;
    const exists = state.data.edges.some(e =>
      (e[0] === source && e[1] === target) || (e[0] === target && e[1] === source)
    );
    if (exists) return state;
    return {
      data: {
        ...state.data,
        edges: [...state.data.edges, [source, target]]
      }
    };
  }),

  setOwner: (nodeId, agentId) => set((state) => {
    if (!state.data) return state;
    return {
      data: {
        ...state.data,
        ownership: { ...state.data.ownership, [nodeId]: agentId }
      }
    };
  }),

  toggleDestination: (agentId, nodeId) => set((state) => {
    if (!state.data) return state;
    const currentDests = state.data.destinations[agentId] || [];
    const isDest = currentDests.includes(nodeId);

    return {
      data: {
        ...state.data,
        destinations: {
          ...state.data.destinations,
          [agentId]: isDest ? currentDests.filter(id => id !== nodeId) : [...currentDests, nodeId]
        }
      }
    };
  }),

  updateOwnership: (data) => set((state) => {
    if (!state.data) return state;
    // Update ownership/edges/destinations but NEVER touch layout (preserves user-dragged positions)
    return {
      data: {
        ...state.data,
        ownership: data.ownership,
        edges: data.edges,
        destinations: data.destinations,
      }
    };
  }),

  updateNodePosition: (nodeId, position) => set((state) => {
    if (!state.layout) return state;
    return {
      layout: {
        ...state.layout,
        [nodeId]: position
      }
    };
  }),

  setLayout: (layout) => set({ layout })
}));
