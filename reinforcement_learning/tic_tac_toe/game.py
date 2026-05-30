"""Compatibility launcher for Tic-Tac-Toe.

- `python game.py` opens the Tkinter GUI.
- `python game.py --cli` runs the terminal version.
"""

from __future__ import annotations

import sys

try:
    from .game_logic import cli_main
    from .gui import main as gui_main
except ImportError:
    from game_logic import cli_main
    from gui import main as gui_main


def main() -> None:
    if "--cli" in sys.argv:
        cli_main()
        return
    gui_main()


if __name__ == "__main__":
    main()
