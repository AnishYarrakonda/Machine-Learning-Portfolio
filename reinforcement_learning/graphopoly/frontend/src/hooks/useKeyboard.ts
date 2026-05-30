import { useEffect } from 'react';
import { useReplayStore } from '../stores/replayStore';
import { useUIStore } from '../stores/uiStore';
import { useGraphStore } from '../stores/graphStore';

export function useKeyboard() {
  const { play, pause, isPlaying, stepForward, stepBack, jumpForward, jumpBack } = useReplayStore();
  const { mode } = useUIStore();
  const { clearAll } = useGraphStore();

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      switch(e.key) {
        case ' ':
          e.preventDefault();
          if (isPlaying) pause();
          else play();
          break;
        case 'ArrowRight':
          e.preventDefault();
          if (e.shiftKey) jumpForward(10); else stepForward();
          break;
        case 'ArrowLeft':
          e.preventDefault();
          if (e.shiftKey) jumpBack(10); else stepBack();
          break;
        case 'Escape':
          break;
        case 'Delete':
        case 'Backspace':
          break;
      }
    };

    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [play, pause, isPlaying, stepForward, stepBack, jumpForward, jumpBack, mode, clearAll]);
}
