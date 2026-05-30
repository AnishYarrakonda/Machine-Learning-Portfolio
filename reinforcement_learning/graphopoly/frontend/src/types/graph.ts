export interface GraphData {
  num_nodes: number;
  edges: [number, number][];
  ownership: Record<string, number>;          // nodeId → agentId
  destinations: Record<string, number[]>;     // agentId → nodeIds
  starting_positions: Record<string, number>; // agentId → nodeId
}
