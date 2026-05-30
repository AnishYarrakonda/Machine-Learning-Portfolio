import os
import pickle
import random
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "tictactoe_qvalues.pkl")

# D4 symmetry group for a square board:
# 4 rotations + 4 reflections.
_TRANSFORMS = (
    lambda r, c: (r, c),           # identity
    lambda r, c: (c, 2 - r),       # rotate 90
    lambda r, c: (2 - r, 2 - c),   # rotate 180
    lambda r, c: (2 - c, r),       # rotate 270
    lambda r, c: (r, 2 - c),       # reflect vertical axis
    lambda r, c: (2 - r, c),       # reflect horizontal axis
    lambda r, c: (c, r),           # reflect main diagonal
    lambda r, c: (2 - c, 2 - r),   # reflect anti-diagonal
)


def _transform_index(index, transform_id):
    """Apply one symmetry transform to a flat tile index [0..8]."""
    row, col = divmod(index, 3)
    new_row, new_col = _TRANSFORMS[transform_id](row, col)
    return new_row * 3 + new_col


def _transform_state(state, transform_id):
    """Transform a base-3 encoded board state under one symmetry."""
    new_state = 0
    for i in range(9):
        cell_value = (state // (3 ** i)) % 3
        if cell_value != 0:
            new_index = _transform_index(i, transform_id)
            new_state += cell_value * (3 ** new_index)
    return new_state


def canonicalize_state_action(state, action):
    """
    Map (state, action) to a canonical representative across all 8 symmetries.
    This collapses equivalent board positions into one shared Q-table key.
    """
    best_key = None
    for transform_id in range(8):
        transformed_state = _transform_state(state, transform_id)
        transformed_action = _transform_index(action, transform_id)
        key = (transformed_state, transformed_action)
        if best_key is None or key < best_key:
            best_key = key
    return best_key


class Board:
    """Compact Tic-Tac-Toe board using a base-3 integer state."""
    def __init__(self):
        self.state = 0
    
    def _get_cell(self, row, col):
        """Return cell value: 0 empty, 1 X, 2 O."""
        index = row * 3 + col
        return (self.state // (3 ** index)) % 3

    def _set_cell(self, row, col, value):
        """Set one cell by rewriting its base-3 digit in the packed state."""
        index = row * 3 + col
        current_value = self._get_cell(row, col)
        self.state -= current_value * (3 ** index)
        self.state += value * (3 ** index)

    def display(self):
        """Pretty-print the board for interactive play."""
        symbols = {0: ' ', 1: 'X', 2: 'O'}
        print("   0   1   2")
        for r in range(3):
            row = [symbols[self._get_cell(r, c)] for c in range(3)]
            print(f"{r}  " + " | ".join(row))
            if r < 2:
                print("  " + "-" * 11)

    def is_valid_move(self, row, col):
        return 0 <= row < 3 and 0 <= col < 3 and self._get_cell(row, col) == 0

    def get_valid_moves(self):
        valid_moves = []
        for i in range(9):
            row, col = divmod(i, 3)
            if self.is_valid_move(row, col):
                valid_moves.append(i)
        return valid_moves

    def make_move_cpu(self, player_num, move):
        """Apply move index [0..8] for player 1/2 if legal."""
        if 0 <= move < 9:
            row, col = divmod(move, 3)
            if self.is_valid_move(row, col):
                self._set_cell(row, col, player_num)
                return True
        return False

    def check_winner(self):
        """Return -1 draw, 0 in-progress, 1 X won, 2 O won."""
        lines = []
        for i in range(3):
            lines.append([self._get_cell(i, j) for j in range(3)])
            lines.append([self._get_cell(j, i) for j in range(3)])
        lines.append([self._get_cell(i, i) for i in range(3)])
        lines.append([self._get_cell(i, 2 - i) for i in range(3)])
        for line in lines:
            if line.count(line[0]) == 3 and line[0] != 0:
                return line[0]
        if all(self._get_cell(r, c) != 0 for r in range(3) for c in range(3)):
            return -1
        return 0

class Player:
    def __init__(self, name, marker):
        self.name = name
        self.marker = marker
    
    def get_move(self, board):
        """Read and validate a human row/col move from stdin."""
        while True:
            try:
                row = int(input(f"\n{self.name}, enter row (0-2): "))
                col = int(input(f"{self.name}, enter col (0-2): "))
                print()
                if board.is_valid_move(row, col):
                    return row, col
            except:
                pass

class CPU(Player):
    """Q-learning-style policy player backed by a shared Q table."""
    def __init__(self, name, marker):
        super().__init__(name, marker)
        self.q_values = {}
        self.moves = []

    def get_and_save_move(self, board: Board, gen=1, q_values=None, epsilon=0.1):
        """
        Choose and record a move.
        - gen1 / epsilon branch: exploration
        - otherwise: exploit highest average reward from Q values
        """
        moves = board.get_valid_moves()
        if gen == 1 or q_values is None or random.random() < epsilon:
            move = random.choice(moves)
        else:
            q_avgs = []
            for m in moves:
                # Symmetry-canonical keys let equivalent board patterns share data.
                key = canonicalize_state_action(board.state, m)
                total, count = q_values.get(key, (0, 0))
                avg = total / count if count > 0 else 0
                q_avgs.append((avg, count, m))
            max_avg = max(q_avgs, key=lambda x: x[0])[0]
            best_moves = [(count, m) for avg, count, m in q_avgs if avg == max_avg]
            max_count = max(best_moves, key=lambda x: x[0])[0]
            best_moves = [m for count, m in best_moves if count == max_count]
            # Deterministic tie-break in pure greedy mode avoids random test instability.
            move = min(best_moves) if gen == 3 and epsilon == 0 else random.choice(best_moves)
        # Record canonical key for reward backprop at game end.
        self.moves.append(canonicalize_state_action(board.state, move))
        return move

def simulate_game(cpu1: CPU, cpu2: CPU, q_values: dict, stats: dict, gen=1, epsilon=0.1, learn=True):
    """Run one self-play game and optionally update Q values from final outcome."""
    board = Board()
    cpu1.moves = []
    cpu2.moves = []
    current_player, other_player = cpu1, cpu2

    while True:
        move = current_player.get_and_save_move(board, gen=gen, q_values=q_values, epsilon=epsilon)
        board.make_move_cpu(current_player.marker, move)
        result = board.check_winner()
        if result != 0:
            if result == -1:
                reward1 = 0
                reward2 = 0
                stats['draws'] += 1

            elif result == cpu1.marker:
                reward1 = 1
                reward2 = -1
                stats['win_loss'] += 1
            else:
                reward1 = -1
                reward2 = 1
                stats['win_loss'] += 1
            
            if learn:
                # Every move from this game gets terminal reward signal.
                for key in cpu1.moves:
                    total_reward, times = q_values.get(key, (0, 0))
                    q_values[key] = (total_reward + reward1, times + 1)
                for key in cpu2.moves:
                    total_reward, times = q_values.get(key, (0, 0))
                    q_values[key] = (total_reward + reward2, times + 1)

            break
        
        current_player, other_player = other_player, current_player


def save_model(q_values, filepath=MODEL_PATH):
    """Serialize learned Q values to disk."""
    with open(filepath, "wb") as file:
        pickle.dump(q_values, file)


def load_model(filepath=MODEL_PATH):
    """Load a saved Q table from disk."""
    with open(filepath, "rb") as file:
        q_values = pickle.load(file)
    if not isinstance(q_values, dict):
        raise ValueError("Loaded model is invalid: expected a dictionary.")
    return q_values


def train_cpus(first_gen_games=0, second_gen_games=0, test_games=0, q_values=None):
    """Train through random + epsilon-greedy phases, then optional greedy evaluation."""
    start_time = time.time()
    if q_values is None:
        q_values = {}
    cpu1 = CPU("CPU 1", 1)
    cpu2 = CPU("CPU 2", 2)

    stats = {'win_loss': 0, 'draws': 0}
    for _ in range(first_gen_games):
        simulate_game(cpu1, cpu2, q_values, stats, gen=1)
    print(f"\nThe gen 1 CPU played itself {first_gen_games} times.\nThe games ended in a win/loss {stats['win_loss']} times.\nThe games ended in a draw {stats['draws']} times.\n")

    stats = {'win_loss': 0, 'draws': 0}
    for _ in range(second_gen_games):
        simulate_game(cpu1, cpu2, q_values, stats, gen=2, epsilon=0.1)
    print(f"The gen 2 CPU played itself {second_gen_games} times.\nThe games ended in a win/loss {stats['win_loss']} times.\nThe games ended in a draw {stats['draws']} times.\n")

    # Evaluation-only pass: no learning updates.
    if test_games > 0:
        stats = {'win_loss': 0, 'draws': 0}
        for _ in range(test_games):
            simulate_game(cpu1, cpu2, q_values, stats, gen=3, epsilon=0, learn=False)
        print(f"The gen 3 CPU played itself {test_games} times.\nThe games ended in a win/loss {stats['win_loss']} times.\nThe games ended in a draw {stats['draws']} times.\n")
    end_time = time.time()
    dif = end_time-start_time
    return q_values, dif

def play_against_cpu(q_values, cpu_marker=2):
    """Play human vs learned policy in terminal."""
    human_marker = 2 if cpu_marker == 1 else 1
    cpu = CPU("CPU", cpu_marker)
    cpu.q_values = q_values
    human = Player("Human", human_marker)
    board = Board()
    current_player = cpu if cpu_marker == 1 else human
    while True:
        board.display()
        if current_player == cpu:
            move = cpu.get_and_save_move(board, gen=3, q_values=q_values, epsilon=0)
            board.make_move_cpu(cpu.marker, move)
            row, col = divmod(move, 3)
            print(f"\nCPU played at row {row}, col {col}\n")
        else:
            row, col = human.get_move(board)
            board.make_move_cpu(human.marker, row*3 + col)
        result = board.check_winner()
        if result != 0:
            board.display()
            if result == -1:
                print("\nIt's a draw!\n")
            elif result == cpu.marker:
                print("\nCPU wins!\n")
            else:
                print("\nYou win!\n")
            break
        current_player = human if current_player == cpu else cpu

def play_cpu_vs_cpu(q_values, cpu1_marker=1, cpu2_marker=2, seconds=1):
    """Watch two greedy CPU players in terminal."""
    cpu1 = CPU("CPU 1", cpu1_marker)
    cpu2 = CPU("CPU 2", cpu2_marker)
    cpu1.q_values = q_values
    cpu2.q_values = q_values
    board = Board()
    current_player, other_player = (cpu1, cpu2) if cpu1_marker == 1 else (cpu2, cpu1)

    board.display()
    time.sleep(seconds)
    while True:
        move = current_player.get_and_save_move(board, gen=3, q_values=q_values, epsilon=0)
        board.make_move_cpu(current_player.marker, move)
        row, col = divmod(move, 3)
        print(f"\n{current_player.name} played at row {row}, col {col}\n")
        board.display()
        time.sleep(seconds)

        result = board.check_winner()
        if result != 0:
            if result == -1:
                print("\nIt's a draw!\n")
            elif result == current_player.marker:
                print(f"\n{current_player.name} wins!\n")
            else:
                print(f"\n{other_player.name} wins!\n")
            break

        current_player, other_player = other_player, current_player

def cli_main():
    """Terminal entrypoint for training/testing/playing without Tkinter."""
    def _prompt_choice(prompt: str, choices: set[str]) -> str:
        while True:
            value = input(prompt).strip()
            if value in choices:
                return value
            print(f"Enter one of: {', '.join(sorted(choices))}\n")

    while True:
        match_up = input("\nChoose an option:\n1 = Player vs. CPU\n2 = Watch CPU vs. CPU\n3 = Test CPU playing itself\nAnything Else = Exit\nEnter number: ").strip()
        if match_up not in {'1','2','3'}:
            print("\nQuitting Program.\n")
            break

        loaded_q_values = None
        load_choice = input("\nLoad a saved model first? (y/n): ").strip().lower()
        if load_choice == "y":
            load_path = input("Enter model file path (default: tictactoe_qvalues.pkl): ").strip() or "tictactoe_qvalues.pkl"
            try:
                loaded_q_values = load_model(load_path)
                print(f"Loaded model from '{load_path}' with {len(loaded_q_values)} state-action entries.\n")
            except Exception as exc:
                print(f"Could not load model: {exc}\nStarting with a fresh model.\n")

        first_gen = int(input("\nEnter number of first generation (random) games to simulate: "))
        second_gen = int(input("\nEnter number of second generation (epsilon-greedy) games to simulate: "))


        if match_up == '3':
            test_games = int(input("\nEnter number of test games (gen 3) to simulate: "))
            q_vals, dif = train_cpus(first_gen, second_gen, test_games=test_games, q_values=loaded_q_values)
            print("Testing complete!\n")
            print(f"Total time spent training was {dif} seconds.\n")
        else:
            q_vals, dif = train_cpus(first_gen, second_gen, test_games=0, q_values=loaded_q_values)
            print(f"Total time spent training was {dif} seconds.\n")

        save_choice = input("Save current model? (y/n): ").strip().lower()
        if save_choice == "y":
            save_path = input("Enter save path (default: tictactoe_qvalues.pkl): ").strip() or "tictactoe_qvalues.pkl"
            try:
                save_model(q_vals, save_path)
                print(f"Model saved to '{save_path}'.\n")
            except Exception as exc:
                print(f"Could not save model: {exc}\n")

        if match_up == '1':
            order = _prompt_choice(
                "Do you want to play first or second?\n1 = First (X)\n2 = Second (O)\nEnter a number: ",
                {"1", "2"},
            )
            if order == "2":
                play_against_cpu(q_vals, cpu_marker=1)
            else:
                play_against_cpu(q_vals, cpu_marker=2)
        elif match_up == '2':
            seconds = float(input("Enter the delay between CPU moves in seconds (ex: 1.5): "))
            play_cpu_vs_cpu(q_vals, seconds=seconds)


if __name__ == "__main__":
    cli_main()
