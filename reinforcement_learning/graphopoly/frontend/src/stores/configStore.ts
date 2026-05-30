import { create } from 'zustand';
import { FullConfig } from '../types/config';

interface ConfigState {
  config: FullConfig | null;
  loadConfig: (config: FullConfig) => void;
  updateConfig: (patch: Partial<FullConfig>) => void;
}

export const useConfigStore = create<ConfigState>((set) => ({
  config: null,
  loadConfig: (config) => set({ config }),
  updateConfig: (patch) => set((state) => ({
    config: state.config ? { ...state.config, ...patch } : null
  }))
}));
