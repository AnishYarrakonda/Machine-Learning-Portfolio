import { GraphData } from './graph';
import { TimelineEntry } from './episode';

export interface GraphResponse {
  status: "ok";
  graph: GraphData;
  layout: Record<number, [number, number]>;
}

export interface AnalysisResponse {
  status: "ok";
  timeline: TimelineEntry[];
  graph_data: any;
  config: any;
  num_steps: number;
  num_agents: number;
  num_nodes: number;
}
