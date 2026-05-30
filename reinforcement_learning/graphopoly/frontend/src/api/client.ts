import { FullConfig } from '../types/config';
import { GraphResponse, AnalysisResponse } from '../types/api';

const API_BASE = '/api';

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...options?.headers },
  });
  if (!res.ok) {
    try {
      const body = await res.json();
      if (body?.message) throw new Error(body.message);
    } catch (e) {
      if (e instanceof Error && e.message && e.message !== 'Unexpected end of JSON input') throw e;
    }
    throw new Error(`API Error: ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export interface ExperimentMeta {
  run_id: string;
  run_name: string;
  mode: 'train' | 'simulate';
  started_at: string;
  finished_at: string;
  num_episodes: number;
  num_agents: number;
  num_nodes: number;
  final_rewards: number[];
  final_trips: number[];
}

export interface SavedGraphMeta {
  graph_id: string;
  name: string;
  saved_at: string;
  num_nodes: number;
  num_agents: number;
}

export const api = {
  graph: {
    random: (data: { num_nodes: number; num_edges: number | null; num_agents: number; num_destinations: number }) =>
      fetchJson<GraphResponse>('/graph/random', { method: 'POST', body: JSON.stringify(data) }),
    build: (data: { num_nodes: number; edges: [number, number][]; ownership: Record<string, number>; destinations: Record<string, number[]>; starting_positions: Record<string, number> }) =>
      fetchJson<GraphResponse>('/graph/build', { method: 'POST', body: JSON.stringify(data) }),
    syncLayout: (layout: Record<string, [number, number]>) =>
      fetchJson<{ status: "ok" }>('/graph/sync-layout', { method: 'POST', body: JSON.stringify({ layout }) }),
  },

  config: {
    get: () => fetchJson<FullConfig>('/config'),
    update: (data: { agent?: Partial<FullConfig['agent']>; train?: Partial<FullConfig['train']>; network?: Partial<FullConfig['network']>; log?: Partial<FullConfig['log']> }) =>
      fetchJson<{ status: "ok"; config: FullConfig }>('/config', { method: 'POST', body: JSON.stringify(data) }),
  },

  train: {
    start: (runName = "") => fetchJson<{ status: "ok"; message: string; run_id: string }>('/train/start', { method: 'POST', body: JSON.stringify({ run_name: runName }) }),
    stop: () => fetchJson<{ status: "ok" }>('/train/stop', { method: 'POST' }),
    pause: () => fetchJson<{ status: "ok"; paused: true }>('/train/pause', { method: 'POST' }),
    resume: () => fetchJson<{ status: "ok"; paused: false }>('/train/resume', { method: 'POST' }),
  },

  simulate: {
    start: (runName = "") => fetchJson<{ status: "ok"; message: string; run_id: string }>('/simulate/start', { method: 'POST', body: JSON.stringify({ run_name: runName }) }),
  },

  analyze: {
    compute: (data: any) =>
      fetchJson<AnalysisResponse>('/analyze/compute', { method: 'POST', body: JSON.stringify(data) }),
  },

  experiments: {
    list: () => fetchJson<{ experiments: ExperimentMeta[] }>('/experiments'),
    load: (runId: string) => fetchJson<{ status: "ok"; data: any }>(`/experiments/${runId}`),
    delete: (runId: string) => fetchJson<{ status: "ok" }>(`/experiments/${runId}`, { method: 'DELETE' }),
  },

  graphs: {
    save: (name: string, graph: Record<string, any>, layout: Record<string, [number, number]>) =>
      fetchJson<{ status: "ok"; graph_id: string }>('/graphs/save', { method: 'POST', body: JSON.stringify({ name, graph, layout }) }),
    list: () => fetchJson<{ graphs: SavedGraphMeta[] }>('/graphs'),
    load: (graphId: string) => fetchJson<{ status: "ok"; data: any }>(`/graphs/${graphId}`),
    delete: (graphId: string) => fetchJson<{ status: "ok" }>(`/graphs/${graphId}`, { method: 'DELETE' }),
  },

  status: () => fetchJson<{ training: boolean; paused: boolean; has_graph: boolean; run_id: string | null; run_name: string; run_mode: string | null }>('/status'),

  export: {
    downloadData: async () => {
      const res = await fetch('/api/export/data');
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.message ?? 'Export failed');
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'graphopoly_data.zip';
      a.click();
      URL.revokeObjectURL(url);
    },
  },
};
