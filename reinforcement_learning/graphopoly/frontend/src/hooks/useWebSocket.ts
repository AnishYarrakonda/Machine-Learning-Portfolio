import { useEffect, useState } from 'react';
import { useTrainingStore } from '../stores/trainingStore';
import { useGraphStore } from '../stores/graphStore';
import { GraphData } from '../types/graph';
import { WSMessage } from '../types/websocket';

const WS_URL = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`;

let sharedWs: WebSocket | null = null;
let sharedStatus: 'connected' | 'connecting' | 'disconnected' = 'disconnected';
const statusListeners: Set<(status: typeof sharedStatus) => void> = new Set();

function notifyStatus(status: typeof sharedStatus) {
  sharedStatus = status;
  statusListeners.forEach(fn => fn(status));
}

let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

function connectShared(
  handleEpisodeUpdate: (msg: Extract<WSMessage, { type: 'episode_update' }>) => void,
  handleTrainingComplete: () => void,
  updateOwnership: (data: GraphData) => void,
) {
  if (sharedWs && sharedWs.readyState <= 1) return;

  notifyStatus('connecting');
  const ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    notifyStatus('connected');
    if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data) as WSMessage;
      if (msg.type === 'episode_update') {
        handleEpisodeUpdate(msg);
        // Update graph data (ownership/edges) but PRESERVE the user's layout positions
        if (msg.data.graph_data) {
          updateOwnership(msg.data.graph_data);
        }
      } else if (msg.type === 'training_complete' || msg.type === 'training_stopped') {
        handleTrainingComplete();
      } else if (msg.type === 'training_error') {
        handleTrainingComplete();
        console.error('[WS] Training Error:', (msg.data as { error: string }).error);
      }
    } catch (err) {
      console.error('[WS] Message parse error:', err);
    }
  };

  ws.onclose = () => {
    sharedWs = null;
    notifyStatus('disconnected');
    reconnectTimer = setTimeout(() => {
      connectShared(handleEpisodeUpdate, handleTrainingComplete, updateOwnership);
    }, 3000);
  };

  ws.onerror = () => ws.close();
  sharedWs = ws;
}

export function useWebSocket(): 'connected' | 'connecting' | 'disconnected' {
  const [status, setStatus] = useState<'connected' | 'connecting' | 'disconnected'>(sharedStatus);

  const handleEpisodeUpdate = useTrainingStore(s => s.handleEpisodeUpdate);
  const handleTrainingComplete = useTrainingStore(s => s.handleTrainingComplete);
  // Only update ownership/edges from WS, not layout (preserves user-dragged positions)
  const updateOwnership = useGraphStore(s => s.updateOwnership);

  useEffect(() => {
    statusListeners.add(setStatus);
    connectShared(handleEpisodeUpdate, handleTrainingComplete, updateOwnership);
    return () => { statusListeners.delete(setStatus); };
  }, [handleEpisodeUpdate, handleTrainingComplete, updateOwnership]);

  return status;
}
