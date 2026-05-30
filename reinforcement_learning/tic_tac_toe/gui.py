from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox

try:
    from .game_logic import Board, CPU, MODEL_PATH, load_model, save_model, train_cpus
except ImportError:
    from game_logic import Board, CPU, MODEL_PATH, load_model, save_model, train_cpus


class TicTacToeApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Tic-Tac-Toe RL")
        self.root.configure(bg="#f2f5f8")
        self.root.resizable(False, False)

        self.q_values: dict = {}
        self.model_path = MODEL_PATH

        self.board = Board()
        self.cpu = CPU("CPU", 2)
        self.human_marker = 1
        self.cpu_marker = 2
        self.game_over = False

        self._build_ui()
        self._new_game()

    def _build_ui(self) -> None:
        top = tk.Frame(self.root, bg="#f2f5f8")
        top.pack(fill="x", padx=12, pady=(12, 6))

        self.first_gen_var = tk.StringVar(value="30000")
        self.second_gen_var = tk.StringVar(value="15000")
        self.test_var = tk.StringVar(value="2000")
        self.play_as_x = tk.BooleanVar(value=True)

        self._labeled_entry(top, "Gen1", self.first_gen_var)
        self._labeled_entry(top, "Gen2", self.second_gen_var)
        self._labeled_entry(top, "Test", self.test_var)

        tk.Button(top, text="Train", command=self._train_model, width=10).pack(side="left", padx=3)
        tk.Button(top, text="Load", command=self._load_model, width=8).pack(side="left", padx=3)
        tk.Button(top, text="Save", command=self._save_model, width=8).pack(side="left", padx=3)
        tk.Button(top, text="Play First (X)", command=self._set_play_first, width=12).pack(side="left", padx=3)
        tk.Button(top, text="Play Second (O)", command=self._set_play_second, width=13).pack(side="left", padx=3)
        tk.Button(top, text="New Game", command=self._new_game, width=10).pack(side="left", padx=3)

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(
            self.root,
            textvariable=self.status_var,
            bg="#f2f5f8",
            fg="#243242",
            font=("Helvetica", 11, "bold"),
        ).pack(anchor="w", padx=12, pady=(2, 8))

        board_frame = tk.Frame(self.root, bg="#dbe2ea", bd=0)
        board_frame.pack(padx=12, pady=(0, 12))

        self.cells: list[tk.Button] = []
        for r in range(3):
            for c in range(3):
                idx = r * 3 + c
                btn = tk.Button(
                    board_frame,
                    text="",
                    width=5,
                    height=2,
                    font=("Helvetica", 26, "bold"),
                    bg="white",
                    fg="#18232f",
                    command=lambda i=idx: self._human_move(i),
                )
                btn.grid(row=r, column=c, padx=4, pady=4)
                self.cells.append(btn)

    def _labeled_entry(self, parent: tk.Widget, label: str, var: tk.StringVar) -> None:
        wrapper = tk.Frame(parent, bg="#f2f5f8")
        wrapper.pack(side="left", padx=4)
        tk.Label(wrapper, text=label, bg="#f2f5f8", font=("Helvetica", 10)).pack(side="left")
        tk.Entry(wrapper, textvariable=var, width=6, justify="center").pack(side="left", padx=(4, 0))

    def _parse_positive_int(self, raw: str, name: str) -> int:
        try:
            value = int(raw)
        except ValueError:
            raise ValueError(f"{name} must be an integer.")
        if value < 0:
            raise ValueError(f"{name} cannot be negative.")
        return value

    def _train_model(self) -> None:
        try:
            g1 = self._parse_positive_int(self.first_gen_var.get(), "Gen1")
            g2 = self._parse_positive_int(self.second_gen_var.get(), "Gen2")
            gt = self._parse_positive_int(self.test_var.get(), "Test")
        except ValueError as exc:
            messagebox.showerror("Invalid Input", str(exc))
            return

        self.status_var.set("Training... please wait")
        self.root.update_idletasks()
        self.q_values, _ = train_cpus(g1, g2, gt, q_values=self.q_values)
        self.status_var.set(f"Training complete. Entries: {len(self.q_values)}")
        self._new_game()

    def _load_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Load model",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.q_values = load_model(path)
            self.model_path = path
            self.status_var.set(f"Loaded model: {path}")
            self._new_game()
        except Exception as exc:
            messagebox.showerror("Load Failed", str(exc))

    def _save_model(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save model",
            defaultextension=".pkl",
            initialfile="tictactoe_qvalues.pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            save_model(self.q_values, path)
            self.model_path = path
            self.status_var.set(f"Saved model: {path}")
        except Exception as exc:
            messagebox.showerror("Save Failed", str(exc))

    def _new_game(self) -> None:
        self.board = Board()
        self.game_over = False

        self.human_marker = 1 if self.play_as_x.get() else 2
        self.cpu_marker = 2 if self.human_marker == 1 else 1
        self.cpu = CPU("CPU", self.cpu_marker)

        for btn in self.cells:
            btn.configure(text="", state="normal")

        self.status_var.set("Your turn" if self.human_marker == 1 else "CPU thinking...")
        if self.cpu_marker == 1:
            self.root.after(250, self._cpu_move)

    def _set_play_first(self) -> None:
        self.play_as_x.set(True)
        self._new_game()

    def _set_play_second(self) -> None:
        self.play_as_x.set(False)
        self._new_game()

    def _render_board(self) -> None:
        symbols = {0: "", 1: "X", 2: "O"}
        for i, btn in enumerate(self.cells):
            r, c = divmod(i, 3)
            val = self.board._get_cell(r, c)
            btn.configure(text=symbols[val])

    def _finish_game_if_needed(self) -> bool:
        result = self.board.check_winner()
        if result == 0:
            return False

        self.game_over = True
        for btn in self.cells:
            btn.configure(state="disabled")

        if result == -1:
            self.status_var.set("Draw")
        elif result == self.human_marker:
            self.status_var.set("You win")
        else:
            self.status_var.set("CPU wins")
        return True

    def _human_move(self, index: int) -> None:
        if self.game_over:
            return

        row, col = divmod(index, 3)
        if not self.board.is_valid_move(row, col):
            return

        self.board.make_move_cpu(self.human_marker, index)
        self._render_board()
        if self._finish_game_if_needed():
            return

        self.status_var.set("CPU thinking...")
        self.root.after(200, self._cpu_move)

    def _cpu_move(self) -> None:
        if self.game_over:
            return

        move = self.cpu.get_and_save_move(self.board, gen=3, q_values=self.q_values, epsilon=0)
        self.board.make_move_cpu(self.cpu_marker, move)
        self._render_board()

        if self._finish_game_if_needed():
            return
        self.status_var.set("Your turn")


def main() -> None:
    root = tk.Tk()
    TicTacToeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
