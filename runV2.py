import numpy as np

from mcts.nodes import MonteCarloTreeSearchNode
from mcts.search import MonteCarloTreeSearch
from tictactoe import TicTacToeGameState, TicTacToeMove

board_size = 3
iteration = 1000


def draw_chessboard(board):
    """
    """
    board_size = board.shape[0]

    for i in range(board_size):
        print("")
        # nomor baris di sebelah kiri
        print("{0:3}".format(i).center(8) + "|", end="")
        for j in range(board_size):
            cell = board[i][j]
            if cell == 0:
                symbol = "_"
            elif cell == 1:
                symbol = "X"
            else:
                symbol = "O"
            print(symbol.center(8), end="")
    print("\n" + "_" * (10 + board_size * 8))


def get_human_action(state):
    """
    Minta input langkah manusia dalam format: row,col (contoh: 0,2)
    """
    location = input("Your move (row,col): ")

    try:
        x_str, y_str = location.split(",")
        x = int(x_str)
        y = int(y_str)
    except ValueError:
        print("Format salah. Contoh yang benar: 0,2")
        return get_human_action(state)

    # manusia = O = -1
    move = TicTacToeMove(x, y, TicTacToeGameState.o)

    if not state.is_move_legal(move):
        print("invalid move")
        return get_human_action(state)

    return move


def is_game_over(state):
    """
    Cek apakah game sudah selesai dan print hasilnya.
    Return True kalau selesai, False kalau belum.
    """
    if state.is_game_over():
        if state.game_result == 1.0:
            print("You lose!")
        elif state.game_result == 0.0:
            print("Tie!")
        elif state.game_result == -1.0:
            print("You Win!")
        return True
    return False


def choose_starting_player():
    """
    Menu awal: pilih siapa yang jalan duluan.
    Return: "computer" atau "human"
    """
    print("Pilih siapa yang jalan duluan:")
    print("1. Komputer (X)")
    print("2. Manusia  (O)")

    while True:
        choice = input("Masukkan pilihan [1/2]: ").strip()
        if choice == "1":
            return "computer"
        if choice == "2":
            return "human"
        print("Pilihan tidak valid, silakan pilih 1 atau 2.")


def init_state(starting_player: str):
    """
    Inisialisasi state awal berdasarkan siapa yang mulai.
    - Jika 'computer'  -> komputer (X = +1) langsung ambil langkah pertama dengan MCTS.
    - Jika 'human'     -> papan kosong, giliran pertama milik manusia (O = -1).
    """
    state = np.zeros((board_size, board_size))

    if starting_player == "computer":
        # komputer (X) mulai duluan
        initial_board_state = TicTacToeGameState(state=state, next_to_move=TicTacToeGameState.x)
        root = MonteCarloTreeSearchNode(state=initial_board_state, parent=None)
        mcts = MonteCarloTreeSearch(root)
        best_node = mcts.best_action(iteration)  # jumlah simulasi MCTS
        return best_node.state
    else:
        # manusia (O) mulai duluan, papan kosong
        initial_board_state = TicTacToeGameState(state=state, next_to_move=TicTacToeGameState.o)
        return initial_board_state


if __name__ == "__main__":
    # 1. Pilih siapa yang jalan duluan
    starting_player = choose_starting_player()

    # 2. Inisialisasi state awal
    current_state = init_state(starting_player)
    draw_chessboard(current_state.board)

    # 3. Loop utama permainan
    while not is_game_over(current_state):

        # --- Giliran manusia ---
        human_move = get_human_action(current_state)
        current_state = current_state.move(human_move)
        draw_chessboard(current_state.board)

        if is_game_over(current_state):
            break

        # --- Giliran komputer (MCTS) ---
        # Dari papan sekarang, set next_to_move = X (komputer)
        board_state = TicTacToeGameState(state=current_state.board, next_to_move=TicTacToeGameState.x)
        root = MonteCarloTreeSearchNode(state=board_state, parent=None)
        mcts = MonteCarloTreeSearch(root)
        best_computer_node = mcts.best_action(iteration)
        current_state = best_computer_node.state
        draw_chessboard(current_state.board)
