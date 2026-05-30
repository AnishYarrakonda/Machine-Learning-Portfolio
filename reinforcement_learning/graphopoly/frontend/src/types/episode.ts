export interface EpisodeMetadata {
  episode_id: string;
  timestamp: string;
  finished_at: string;
  num_steps: number;
  num_episodes: number;
  num_agents: number;
  num_nodes: number;
  description: string;
}

export interface TrajectoryStep {
  step: number;
  agent_positions: Record<string, number>;
  actions: Record<string, { move: number; price_changes: Record<string, number> }>;
  prices: Record<string, number>;
  rewards: Record<string, number>;
  taxes: Record<string, Record<string, number>>;
  dest_completions: { agent: number; node: number }[];
  node_stats: Record<string, { visits: number; revenue_collected: number }>;
  agent_stats: Record<string, {
    trips_completed: number;
    total_profit: number;
    tax_revenue: number;
    tax_paid: number;
    dest_revenue: number;
  }>;
}

export interface EpisodeJSON {
  metadata: EpisodeMetadata;
  graph: {
    nodes: { id: number; owner: number; position: [number, number] }[];
    edges: [number, number][];
    ownership: Record<string, number>;
    destinations: Record<string, number[]>;
    starting_positions: Record<string, number>;
  };
  config: Record<string, unknown>;
  initial_state: {
    agent_positions: Record<string, number>;
    agent_destinations: Record<string, number[]>;
    agent_owned_nodes: Record<string, number[]>;
    prices: Record<string, number>;
    agent_stats: Record<string, unknown>;
  };
  trajectory: TrajectoryStep[];
  training_metrics: {
    episode_rewards: Record<string, number>[];
    episode_trips: Record<string, number>[];
    losses: { policy_loss: number[]; value_loss: number[]; entropy_bonus: number[] };
    num_episodes_trained: number;
  };
  aggregate_stats: {
    agents: Record<string, unknown>;
    nodes: Record<string, unknown>;
    system: Record<string, unknown>;
  };
}

export interface TimelineEntry {
  timestep: number;
  agents: Record<string, {
    cumulative_reward: number;
    net_profit: number;
    trips_completed: number;
    dest_revenue: number;
    tax_revenue: number;
    tax_paid: number;
    step_reward: number;
    total_visits: number;
  }>;
  nodes: Record<string, {
    current_price: number;
    total_visits: number;
    revenue_collected: number;
    avg_visits_per_step: number;
    owner?: number;
  }>;
  system: {
    total_system_reward: number;
    avg_node_price: number;
    revenue_distribution: Record<string, number>;
  };
}
