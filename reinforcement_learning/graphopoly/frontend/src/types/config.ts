export interface FullConfig {
  graph: { num_nodes: number; num_edges: number | null; [key: string]: unknown };
  agent: { num_agents: number; num_destinations: number; trip_reward: number; price_budget: number };
  train: { steps_per_episode: number; num_episodes: number; lr: number; gamma: number; gae_lambda: number; clip_epsilon: number; entropy_coef: number; value_coef: number; max_grad_norm: number; ppo_epochs: number; batch_size: number };
  network: { hidden_dim: number; num_gnn_layers: number; gat_heads: number; move_mlp_hidden: number; dropout: number };
  log: { log_dir: string; log_every: number; save_full_history: boolean };
  seed: number; device: string;
}
