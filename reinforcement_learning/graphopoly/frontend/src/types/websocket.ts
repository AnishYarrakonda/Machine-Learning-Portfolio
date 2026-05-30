import { GraphData } from './graph';
import { FullConfig } from './config';

export interface AgentDict {
  agent_id: number;
  position: number;
  cumulative_reward: number;
  tax_revenue: number;
  tax_paid: number;
  dest_revenue: number;
  trips_completed: number;
  owned_nodes: number[];
  destinations: number[];
  prices: Record<string, number>;
  [key: string]: unknown;
}

export interface StepHistoryEntry {
  step: number;
  positions: number[];
  prices: Record<string, number>;
  actions: { move: number; price_changes: Record<string, number> }[];
  rewards: number[];
  taxes: Record<string, Record<string, number>>;
  dest_completions: { agent: number; node: number }[];
}

export type WSMessage =
  | {
      type: "episode_update";
      data: {
        episode: number;
        total_episodes: number;
        env_snapshot: { step: number; agents: AgentDict[]; prices: Record<string, number>; positions: number[] };
        graph_data: GraphData;
        layout: Record<number, [number, number]>;
        episode_rewards: number[];
        episode_trips: number[];
        losses: Record<number, { policy_loss: number; value_loss: number; entropy: number }>;
        agent_details: AgentDict[];
        step_history: StepHistoryEntry[];
        graph_embedded: GraphData;
        config_snapshot: FullConfig;
        stopped_early: boolean;
      };
    }
  | {
      type: "training_complete" | "training_stopped";
      data: { run_file: string; run_dir: string; final_rewards: number[]; final_trips: number[]; stopped_early: boolean };
    }
  | {
      type: "training_error";
      data: { error: string; trace: string };
    };
