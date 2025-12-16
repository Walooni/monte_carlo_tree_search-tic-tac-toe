import sys
import pygame
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math


from tictactoe import TicTacToeGameState, TicTacToeMove
from mcts.nodes import MonteCarloTreeSearchNode
from mcts.search import MonteCarloTreeSearch

# ----- CONFIG -----
BOARD_SIZE = 3           # engine default tic-tac-toe size
WINDOW_SIZE = 600
FPS = 60

AI_PLAYER = 1            # X
HUMAN_PLAYER = -1        # O

# Modern clean style colors
BG_COLOR = (18, 24, 38)
PANEL_COLOR = (30, 39, 58)
GRID_COLOR = (210, 215, 225)
X_COLOR = (82, 139, 255)
O_COLOR = (255, 117, 140)
TEXT_COLOR = (235, 241, 255)
BUTTON_BG = (50, 63, 90)
BUTTON_BG_HOVER = (70, 90, 130)
BUTTON_TEXT = (240, 244, 255)


class Button:
    def __init__(self, rect, text, font, callback):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.font = font
        self.callback = callback

    def draw(self, screen, mouse_pos):
        is_hover = self.rect.collidepoint(mouse_pos)
        color = BUTTON_BG_HOVER if is_hover else BUTTON_BG
        pygame.draw.rect(screen, color, self.rect, border_radius=10)
        text_surf = self.font.render(self.text, True, BUTTON_TEXT)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def handle_event(self, event, mouse_pos):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(mouse_pos):
                if self.callback:
                    self.callback()


# ========================
#  DRAW GAME BOARD
# ========================
def draw_board(screen, state, font, info_text=""):
    screen.fill(BG_COLOR)
    cell_size = WINDOW_SIZE // BOARD_SIZE

    # Grid
    for i in range(1, BOARD_SIZE):
        pygame.draw.line(screen, GRID_COLOR,
                         (i * cell_size, 0),
                         (i * cell_size, WINDOW_SIZE), 2)
        pygame.draw.line(screen, GRID_COLOR,
                         (0, i * cell_size),
                         (WINDOW_SIZE, i * cell_size), 2)

    board = state.board
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            x0 = col * cell_size
            y0 = row * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            cell_value = board[row, col]

            if cell_value == AI_PLAYER:
                margin = cell_size * 0.25
                pygame.draw.line(screen, X_COLOR,
                                 (x0 + margin, y0 + margin),
                                 (x1 - margin, y1 - margin), 6)
                pygame.draw.line(screen, X_COLOR,
                                 (x0 + margin, y1 - margin),
                                 (x1 - margin, y0 + margin), 6)
            elif cell_value == HUMAN_PLAYER:
                margin = cell_size * 0.2
                pygame.draw.circle(screen, O_COLOR,
                                   (int((x0 + x1) / 2), int((y0 + y1) / 2)),
                                   int(cell_size / 2 - margin), 6)

    # Info text at top-left
    if info_text:
        text_surf = font.render(info_text, True, TEXT_COLOR)
        screen.blit(text_surf, (16, 12))


# ========================
#  MCTS TREE VISUALIZATION
# ========================
def show_mcts_tree(root_node, selected_node=None, max_depth=3, max_nodes=80):
    """
    Visualisasi tree MCTS:
    - Warna node = kualitas (q / winrate)
      hijau = bagus, kuning = netral, merah = jelek
    - Ukuran node = jumlah kunjungan (n)
    - Cabang yang dipilih AI (root -> selected_node) di-highlight putih
    """
    G = nx.DiGraph()
    node_stats = {}  # node_id -> {"n": ..., "q": ..., "depth": ...}

    def add_nodes(node, depth, counter):
        if depth > max_depth or counter[0] >= max_nodes:
            return

        node_id = id(node)
        if node_id in node_stats:
            return  # sudah diproses

        # ambil statistik dari node
        n_visits = getattr(node, "n", getattr(node, "_number_of_visits", 0))
        results = getattr(node, "_results", None)

        q_value = 0.0
        if isinstance(results, dict) and results:
            total = sum(results.values())
            if total > 0:
                wins = results.get(1.0, 0)
                losses = results.get(-1.0, 0)
                q_value = (wins - losses) / float(total)  # kira-kira di [-1, 1]

        node_stats[node_id] = {"n": n_visits, "q": q_value, "depth": depth}
        G.add_node(node_id)

        counter[0] += 1
        if counter[0] >= max_nodes:
            return

        for child in getattr(node, "children", []):
            child_id = id(child)
            G.add_edge(node_id, child_id)  # ini bisa menambahkan node 'child' ke G
            add_nodes(child, depth + 1, counter)

    add_nodes(root_node, depth=0, counter=[0])

    if len(G.nodes) == 0:
        print("Tree MCTS kosong atau terlalu kecil untuk divisualkan.")
        return

    # --- Cari path dari root ke selected_node (cabang yang dipilih AI) ---
    highlight_nodes = set()
    highlight_edges = set()
    if selected_node is not None:
        cur = selected_node
        while cur is not None:
            nid = id(cur)
            if nid in node_stats:
                highlight_nodes.add(nid)
                if getattr(cur, "parent", None) is not None:
                    pid = id(cur.parent)
                    if pid in node_stats:
                        highlight_edges.add((pid, nid))
            cur = getattr(cur, "parent", None)

    pos = _tree_layout_positions(G, node_stats)

    # ⚠️ Node & edge yang valid = hanya yang ada di 'pos'
    node_list = [n for n in G.nodes() if n in pos]
    all_edges = [(u, v) for (u, v) in G.edges() if u in pos and v in pos]
    highlight_edges = [(u, v) for (u, v) in highlight_edges if u in pos and v in pos]

    # --- siapkan warna & ukuran node ---
    node_colors = []
    node_sizes = []
    for nid in node_list:
        stats = node_stats.get(nid, {"n": 0, "q": 0.0})
        n_visits = stats["n"]
        q = stats["q"]

        # warna berdasarkan q
        if n_visits == 0:
            base_color = "#888888"  # abu-abu kalau belum pernah dikunjungi
        else:
            if q <= -0.3:
                base_color = "#e74c3c"  # merah
            elif q >= 0.3:
                base_color = "#2ecc71"  # hijau
            else:
                base_color = "#f1c40f"  # kuning

        # override kalau termasuk cabang yang dipilih
        if nid in highlight_nodes:
            base_color = "#230daf"  # putih terang

        node_colors.append(base_color)

        # ukuran node berdasarkan n (log supaya nggak kebesaran)
        if n_visits <= 0:
            size = 400
        else:
            size = 400 + 200 * math.log10(n_visits + 1)
        node_sizes.append(size)

    plt.figure(figsize=(10, 6))

    # draw semua edge (abu-abu dulu)
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=all_edges,
        edge_color="#777777",
        arrows=True,
        alpha=0.6,
    )

    # highlight edges di path terpilih (tebal & putih)
    if highlight_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=highlight_edges,
            width=3,
            edge_color="#ffffff",
            arrows=True,
            alpha=0.9,
        )

    # draw node dengan warna & ukuran sesuai
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=node_list,
        node_color=node_colors,
        node_size=node_sizes,
    )

    # label: n & q
    node_labels = {}
    for nid in node_list:
        stats = node_stats.get(nid, {"n": 0, "q": 0.0})
        node_labels[nid] = f"n={stats['n']}\nq={stats['q']:.2f}"

    for nid, (x, y) in pos.items():
        if nid not in node_list:
            continue
        label = node_labels.get(nid, "")
        plt.text(x, y, label, fontsize=8, ha="center", va="center", color="#000000")

    plt.title("MCTS Search Tree\nWarna = kualitas, ukuran = kunjungan, putih = cabang terpilih")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def _tree_layout_positions(G, node_stats):
    """
    Layout sederhana:
    - group node berdasarkan depth (dari node_stats)
    - x: sebar merata di [0,1] per level
    - y: -depth supaya level makin bawah makin negatif
    Hanya node yang ada di node_stats yang dikasih posisi.
    """
    depth_dict = {}
    for nid, stats in node_stats.items():
        d = stats.get("depth", 0)
        depth_dict.setdefault(d, []).append(nid)

    pos = {}
    for depth, nodes in depth_dict.items():
        step = 1.0 / (len(nodes) + 1)
        for i, nid in enumerate(nodes, start=1):
            x = i * step
            y = -depth
            pos[nid] = (x, y)
    return pos



# ========================
#  HELPER
# ========================
def get_cell_from_mouse(pos):
    cell_size = WINDOW_SIZE // BOARD_SIZE
    x, y = pos
    col = x // cell_size
    row = y // cell_size
    if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
        return row, col
    return None, None


# ========================
#  MAIN MENU
# ========================
def run_menu(screen, clock):
    font_title = pygame.font.SysFont("arial", 48, bold=True)
    font_sub = pygame.font.SysFont("arial", 24)
    font_btn = pygame.font.SysFont("arial", 24)

    starting_player = "human"   # or "ai"
    difficulty_levels = ["Easy", "Medium", "Hard"]
    difficulty_index = 1        # default Medium

    def toggle_starting_player():
        nonlocal starting_player
        starting_player = "ai" if starting_player == "human" else "human"

    def cycle_difficulty():
        nonlocal difficulty_index
        difficulty_index = (difficulty_index + 1) % len(difficulty_levels)

    chosen_settings = {"start": False, "quit": False}

    def on_start():
        chosen_settings["start"] = True

    def on_quit():
        chosen_settings["quit"] = True

    width, height = screen.get_size()
    button_width = 260
    button_height = 50
    center_x = width // 2

    btn_start_player = Button(
        rect=(center_x - button_width // 2, height // 2 - 40, button_width, button_height),
        text="First Move: Player (O)",
        font=font_btn,
        callback=toggle_starting_player,
    )
    btn_difficulty = Button(
        rect=(center_x - button_width // 2, height // 2 + 20, button_width, button_height),
        text="Difficulty: Medium",
        font=font_btn,
        callback=cycle_difficulty,
    )
    btn_play = Button(
        rect=(center_x - button_width // 2, height // 2 + 100, button_width, button_height),
        text="Start Game",
        font=font_btn,
        callback=on_start,
    )
    btn_quit = Button(
        rect=(center_x - button_width // 2, height // 2 + 160, button_width, button_height),
        text="Quit",
        font=font_btn,
        callback=on_quit,
    )

    buttons = [btn_start_player, btn_difficulty, btn_play, btn_quit]

    while True:
        clock.tick(FPS)
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
            if event.type == pygame.MOUSEBUTTONDOWN:
                for b in buttons:
                    b.handle_event(event, mouse_pos)

        if chosen_settings["quit"]:
            return None
        if chosen_settings["start"]:
            difficulty = difficulty_levels[difficulty_index]
            return {
                "starting_player": starting_player,
                "difficulty": difficulty,
            }

        screen.fill(BG_COLOR)

        # Top title panel
        pygame.draw.rect(screen, PANEL_COLOR, (0, 0, width, 120))
        title_surf = font_title.render("Tic Tac Toe - MCTS", True, TEXT_COLOR)
        title_rect = title_surf.get_rect(center=(width // 2, 50))
        screen.blit(title_surf, title_rect)

        subtitle_surf = font_sub.render(
            "Modern UI  •  Monte Carlo Tree Search  •  Pygame",
            True,
            TEXT_COLOR,
        )
        subtitle_rect = subtitle_surf.get_rect(center=(width // 2, 90))
        screen.blit(subtitle_surf, subtitle_rect)

        # Update button text based on current state
        btn_start_player.text = f"First Move: {'Player (O)' if starting_player == 'human' else 'Computer (X)'}"
        btn_difficulty.text = f"Difficulty: {difficulty_levels[difficulty_index]}"

        for b in buttons:
            b.draw(screen, mouse_pos)

        pygame.display.flip()


# ========================
#  GAME LOOP
# ========================
def run_game(screen, clock, settings):
    font = pygame.font.SysFont("arial", 24)

    diff = settings.get("difficulty", "Medium")
    if diff == "Easy":
        sim_count = 200
    elif diff == "Hard":
        sim_count = 2000
    else:
        sim_count = 800

    starting_player = settings.get("starting_player", "human")

    board = np.zeros((BOARD_SIZE, BOARD_SIZE))
    if starting_player == "human":
        current_state = TicTacToeGameState(state=board, next_to_move=HUMAN_PLAYER)
        info_text = "Your turn (O). Click a cell."
        pending_ai_opening = False
    else:
        current_state = TicTacToeGameState(state=board, next_to_move=AI_PLAYER)
        info_text = "Computer starts (X)..."
        pending_ai_opening = True

    game_over = False
    last_root_node = None

    while True:
        clock.tick(FPS)

        # If AI starts, let it make the first move once
        if pending_ai_opening and not game_over:
            pending_ai_opening = False
            root_node = MonteCarloTreeSearchNode(state=current_state, parent=None)
            mcts = MonteCarloTreeSearch(root_node)

            # Komputer hitung MCTS dulu
            best_node = mcts.best_action(sim_count)
            last_root_node = root_node

            # 🔍 TAMPILKAN TREE (highlight cabang best_node)
            show_mcts_tree(root_node, best_node)

            # Baru update state dengan langkah AI
            current_state = best_node.state

            if current_state.is_game_over():
                game_over = True
                if current_state.game_result == 1.0:
                    info_text = "Computer wins! (X)  •  Press R to replay or ESC for menu."
                elif current_state.game_result == -1.0:
                    info_text = "You win! (O)  •  Press R to replay or ESC for menu."
                else:
                    info_text = "Draw!  •  Press R to replay or ESC for menu."
            else:
                info_text = "Your turn (O). Click a cell."


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "menu"
                if event.key == pygame.K_r:
                    return "restart"
                if event.key == pygame.K_t:
                    if last_root_node is not None:
                        show_mcts_tree(last_root_node)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if game_over:
                    continue
                if current_state.next_to_move != HUMAN_PLAYER:
                    continue

                row, col = get_cell_from_mouse(event.pos)
                if row is None:
                    continue

                move = TicTacToeMove(row, col, HUMAN_PLAYER)
                if not current_state.is_move_legal(move):
                    info_text = "Invalid move. Try again."
                else:
                    # Player move
                    current_state = current_state.move(move)
                    if current_state.is_game_over():
                        game_over = True
                        if current_state.game_result == 1.0:
                            info_text = "Computer wins! (X)  •  Press R to replay or ESC for menu."
                        elif current_state.game_result == -1.0:
                            info_text = "You win! (O)  •  Press R to replay or ESC for menu."
                        else:
                            info_text = "Draw!  •  Press R to replay or ESC for menu."
                    else:
                        # AI move
                        info_text = "Computer thinking..."
                        draw_board(screen, current_state, font, info_text)
                        pygame.display.flip()

                        root_node = MonteCarloTreeSearchNode(state=current_state, parent=None)
                        mcts = MonteCarloTreeSearch(root_node)

                        # Komputer hitung MCTS dulu
                        best_node = mcts.best_action(sim_count)
                        last_root_node = root_node

                        # 🔍 TAMPILKAN TREE (highlight cabang best_node)
                        show_mcts_tree(root_node, best_node)

                        # Baru update papan dengan langkah terbaik
                        current_state = best_node.state


                        if current_state.is_game_over():
                            game_over = True
                            if current_state.game_result == 1.0:
                                info_text = "Computer wins! (X)  •  Press R to replay or ESC for menu."
                            elif current_state.game_result == -1.0:
                                info_text = "You win! (O)  •  Press R to replay or ESC for menu."
                            else:
                                info_text = "Draw!  •  Press R to replay or ESC for menu."
                        else:
                            info_text = "Your turn (O). Click a cell."

        draw_board(screen, current_state, font, info_text)
        pygame.display.flip()


# ========================
#  MAIN ENTRY
# ========================
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Tic Tac Toe - MCTS (Modern UI)")
    clock = pygame.time.Clock()

    while True:
        settings = run_menu(screen, clock)
        if settings is None:
            break

        while True:
            result = run_game(screen, clock, settings)
            if result == "quit":
                pygame.quit()
                sys.exit()
            if result == "menu":
                break
            if result == "restart":
                continue
        # kembali ke menu

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
