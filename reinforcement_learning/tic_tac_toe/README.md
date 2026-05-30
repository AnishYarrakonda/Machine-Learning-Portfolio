# Tic-Tac-Toe Reinforcement Learning

Simple tabular Q-learning implementation with both GUI training/playback and CLI experimentation modes.

## Requirements
- Python 3.10+ (standard library only, though `tkinter` is required for the GUI).

## Usage
- `python game.py` (default): launches `tkinter` via `gui.py`, letting you train, save/load Q-values, and play human vs CPU.
  - Controls: set generation/test counts, press `Train`/`Test`, then choose `Play First (X)` or `Play Second (O)` to start a new match.
  - `Save`/`Load` operate on `.pkl` files; the default `tictactoe_qvalues.pkl` lives in this folder.
- `python game.py --cli`: runs the CLI menu that supports player vs CPU, CPU vs CPU, or self-play testing.
- `python gui.py`: bypasses the launcher and opens the GUI directly.

## Files
- `game.py`: launcher that handles CLI flags and redirects to `gui.py` or CLI menus defined in `game_logic.py`.
- `gui.py`: Tkinter app that exposes training counters, play buttons, and a board display.
- `game_logic.py`: board representation, Q-learning loops, serialization helpers, and CLI menus.
- `tictactoe_qvalues.pkl`: precomputed Q-values used by default when launching the GUI.

## Notes
- Training parameters (learning rate, epsilon, episodes) live inside `game_logic.py`; tweak them to observe how the agent learns.
- Use the CLI mode to snapshot different policies before loading them into the GUI.
