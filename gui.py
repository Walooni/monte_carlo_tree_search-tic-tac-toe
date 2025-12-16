import tkinter as tk
from tkinter import messagebox

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from tictactoe import TicTacToeGameState, TicTacToeMove
from mcts.nodes import MonteCarloTreeSearchNode
from mcts.search import MonteCarloTreeSearch

# Konstanta pemain
AI_PLAYER = 1      # X
HUMAN_PLAYER = -1  # O

BOARD_SIZE = 3     # kalau nanti tictactoe.py sudah kamu ubah ke board_size dinamis, tinggal sambungkan ke sana


class TicTacToeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic Tac Toe - MCTS")

        self.cell_size = 100
        canvas_size = self.cell_size * BOARD_SIZE

        self.canvas = tk.Canvas(self.root, width=canvas_size, height=canvas_size, bg="white")
        self.canvas.pack()

        # Bind klik mouse
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Tombol reset
        self.reset_button = tk.Button(self.root, text="Reset Game", command=self.reset_game)
        self.reset_button.pack(pady=5)

        # Inisialisasi state permainan (manusia duluan)
        self.current_state = TicTacToeGameState(
            state=np.zeros((BOARD_SIZE, BOARD_SIZE)),
            next_to_move=HUMAN_PLAYER
        )

        self.game_over = False
        self.draw_board()

    # =========================
    #  Bagian GUI papan
    # =========================
    def draw_board(self):
        self.canvas.delete("all")
        # Garis grid
        for i in range(1, BOARD_SIZE):
            # Vertical
            x = i * self.cell_size
            self.canvas.create_line(x, 0, x, BOARD_SIZE * self.cell_size)
            # Horizontal
            y = i * self.cell_size
            self.canvas.create_line(0, y, BOARD_SIZE * self.cell_size, y)

        # Gambar X dan O
        board = self.current_state.board
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                x0 = j * self.cell_size
                y0 = i * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size

                if board[i, j] == AI_PLAYER:
                    # Gambar X
                    margin = 20
                    self.canvas.create_line(x0 + margin, y0 + margin, x1 - margin, y1 - margin, width=2)
                    self.canvas.create_line(x0 + margin, y1 - margin, x1 - margin, y0 + margin, width=2)
                elif board[i, j] == HUMAN_PLAYER:
                    # Gambar O
                    margin = 20
                    self.canvas.create_oval(x0 + margin, y0 + margin, x1 - margin, y1 - margin, width=2)

    def on_canvas_click(self, event):
        if self.game_over:
            return

        # Hitung index baris/kolom dari koordinat klik
        col = event.x // self.cell_size
        row = event.y // self.cell_size

        if row < 0 or row >= BOARD_SIZE or col < 0 or col >= BOARD_SIZE:
            return

        # Hanya boleh jalan kalau memang giliran manusia
        if self.current_state.next_to_move != HUMAN_PLAYER:
            return

        move = TicTacToeMove(row, col, HUMAN_PLAYER)

        if not self.current_state.is_move_legal(move):
            messagebox.showinfo("Invalid", "Langkah tidak valid!")
            return

        # Terapkan langkah manusia
        self.current_state = self.current_state.move(move)
        self.draw_board()

        if self.check_game_over():
            return

        # Giliran AI
        self.root.after(300, self.ai_move)

    def reset_game(self):
        self.current_state = TicTacToeGameState(
            state=np.zeros((BOARD_SIZE, BOARD_SIZE)),
            next_to_move=HUMAN_PLAYER
        )
        self.game_over = False
        self.draw_board()

    def check_game_over(self):
        if self.current_state.is_game_over():
            self.game_over = True
            if self.current_state.game_result == 1.0:
                messagebox.showinfo("Game Over", "Komputer menang!")
            elif self.current_state.game_result == -1.0:
                messagebox.showinfo("Game Over", "Kamu menang!")
            else:
                messagebox.showinfo("Game Over", "Seri!")
            return True
        return False

    # =========================
    #  Bagian MCTS + visualisasi tree
    # =========================
    def ai_move(self):
        if self.game_over:
            return

        # Pastikan sekarang giliran AI
        if self.current_state.next_to_move != AI_PLAYER:
            # Paksa state baru dengan next_to_move = AI, papan sama
            self.current_state = TicTacToeGameState(
                state=self.current_state.board.copy(),
                next_to_move=AI_PLAYER
            )

        root_node = MonteCarloTreeSearchNode(state=self.current_state, parent=None)
        mcts = MonteCarloTreeSearch(root_node)

        # Jalankan MCTS
        best_node = mcts.best_action(1000)  # 1000 simulasi, bisa kamu naikkan

        # Update state dengan langkah AI
        self.current_state = best_node.state
        self.draw_board()

        # Tampilkan tree MCTS (berdasarkan root_node yang sudah diupdate selama best_action)
        self.show_mcts_tree(root_node)

        self.check_game_over()

    def show_mcts_tree(self, root_node, max_depth=3, max_nodes=50):
        """
        Visualisasi tree MCTS dengan networkx + matplotlib.
        - max_depth: kedalaman maksimum yang divisualkan
        - max_nodes: jumlah maksimum node yang dimasukkan ke graph (supaya tidak terlalu berat)
        """
        G = nx.DiGraph()

        # gunakan id(node) sebagai identifier unik
        def add_nodes_recursively(node, depth, counter):
            if depth > max_depth or counter[0] >= max_nodes:
                return

            node_id = id(node)
            # label: visit count dan average score (kalau ada)
            n = getattr(node, "n", getattr(node, "_number_of_visits", 0))
            # beberapa implementasi punya properti q, kalau tidak, kita skip
            q = getattr(node, "q", None)
            if q is None and hasattr(node, "_results"):
                # optional: bisa hitung sendiri rata-rata, tapi sederhana saja
                q = 0

            label = f"n={n}\nq={q}"

            G.add_node(node_id, label=label, depth=depth)

            counter[0] += 1
            if counter[0] >= max_nodes:
                return

            for child in getattr(node, "children", []):
                child_id = id(child)
                G.add_edge(node_id, child_id)
                add_nodes_recursively(child, depth + 1, counter)

        add_nodes_recursively(root_node, depth=0, counter=[0])

        if len(G.nodes) == 0:
            return

        # Layout sederhana: atur posisi berdasarkan depth
        pos = self._tree_layout_positions(G)

        plt.figure(figsize=(10, 6))
        nx.draw(
            G, pos,
            with_labels=False,
            node_size=800,
            arrows=True
        )

        # Gambar label manual supaya bisa multiline
        node_labels = nx.get_node_attributes(G, "label")
        for node, (x, y) in pos.items():
            label = node_labels.get(node, "")
            plt.text(x, y, label, fontsize=8, ha="center", va="center")

        plt.title("MCTS Search Tree (truncated)")
        plt.axis("off")
        plt.show()

    def _tree_layout_positions(self, G):
        """
        Layout sederhana untuk tree:
        - x: urutan node dalam satu level
        - y: -depth
        """
        # group node by depth
        depth_dict = {}
        for node, data in G.nodes(data=True):
            d = data.get("depth", 0)
            depth_dict.setdefault(d, []).append(node)

        pos = {}
        max_width = max(len(nodes) for nodes in depth_dict.values())
        for depth, nodes in depth_dict.items():
            # sebar node di level ini di sepanjang [0,1]
            step = 1.0 / (len(nodes) + 1)
            for i, node in enumerate(nodes, start=1):
                x = i * step
                y = -depth
                pos[node] = (x, y)
        return pos


if __name__ == "__main__":
    root = tk.Tk()
    app = TicTacToeGUI(root)
    root.mainloop()
