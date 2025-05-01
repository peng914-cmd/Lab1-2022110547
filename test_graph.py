import sys
import random
import re
from collections import defaultdict, deque

import networkx as nx
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit, QFileDialog, QTabWidget,
    QMessageBox, QGroupBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class TextGraphAnalyzer:
    def __init__(self, damping=0.85):
        self.adj = defaultdict(lambda: defaultdict(int))
        self.damping = damping

    def add_edge(self, u, v):
        self.adj[u][v] += 1
        if v not in self.adj:
            self.adj[v] = defaultdict(int)

    def build_graph_from_file(self, filepath):
        self.adj.clear()
        words = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                clean = re.sub(r'[^A-Za-z]', ' ', line).lower()
                words.extend(clean.split())
        for i in range(len(words) - 1):
            self.add_edge(words[i], words[i+1])
        return len(self.adj)

    def neighbors(self, u):
        return dict(self.adj[u])

    def weight(self, u, v):
        return self.adj[u].get(v, 0)

    def query_bridge_words(self, w1, w2):
        w1, w2 = w1.lower(), w2.lower()
        if w1 not in self.adj or w2 not in self.adj:
            return f"No '{w1}' or '{w2}' in the graph!"
        bridges = [mid for mid in self.neighbors(w1) if self.weight(mid, w2) > 0]
        if not bridges:
            return f"No bridge words from '{w1}' to '{w2}'!"
        return f"Bridge words from '{w1}' to '{w2}': {', '.join(bridges)}"

    def generate_new_text(self, input_text):
        clean = re.sub(r'[^A-Za-z\s]', ' ', input_text)
        ws = clean.split()
        output = []
        for i in range(len(ws)):
            output.append(ws[i])
            if i < len(ws) - 1:
                w, nxt = ws[i].lower(), ws[i+1].lower()
                bridges = [mid for mid in self.neighbors(w) if self.weight(mid, nxt) > 0]
                if bridges:
                    output.append(random.choice(bridges))
        return ' '.join(output)

    def calc_shortest_path(self, source, target):
        source, target = source.lower(), target.lower()
        verts = set(self.adj.keys())
        if source not in verts or target not in verts:
            return f"'{source}' or '{target}' not in graph!"
        dist = {v: float('inf') for v in verts}
        prev = {}
        dist[source] = 0
        pq = [(0, source)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == target:
                break
            if d > dist[u]:
                continue
            for v, w in self.neighbors(u).items():
                alt = d + w
                if alt < dist[v]:
                    dist[v], prev[v] = alt, u
                    heapq.heappush(pq, (alt, v))
        if target not in prev and source != target:
            return f"{source} cannot reach {target}!"
        path = deque()
        cur = target
        while True:
            path.appendleft(cur)
            if cur == source:
                break
            cur = prev[cur]
        return f"Shortest path: {' -> '.join(path)} (length={int(dist[target])})"

    def cal_pagerank(self, word, iterations=100):
        word = word.lower()
        verts = list(self.adj.keys())
        if word not in verts:
            return None
        n = len(verts)
        pr = {v: 1.0 / n for v in verts}
        for _ in range(iterations):
            new_pr = {}
            for v in verts:
                s = 0
                for u in verts:
                    if v in self.neighbors(u):
                        out_deg = len(self.neighbors(u))
                        if out_deg > 0:
                            s += pr[u] / out_deg
                new_pr[v] = (1 - self.damping) / n + self.damping * s
            pr = new_pr
        return pr[word]

    def random_walk(self):
        verts = list(self.adj.keys())
        if not verts:
            return 'Graph is empty!'
        curr = random.choice(verts)
        walk = [curr]
        seen = set()
        while True:
            nbrs = self.neighbors(curr)
            if not nbrs:
                break
            total = sum(nbrs.values())
            r = random.randint(1, total)
            s = 0
            for v, w in nbrs.items():
                s += w
                if s >= r:
                    nxt = v
                    break
            edge = (curr, nxt)
            if edge in seen:
                break
            seen.add(edge)
            curr = nxt
            walk.append(curr)
        return ' '.join(walk)


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)


class GraphAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.analyzer = TextGraphAnalyzer()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Text Graph Analyzer')
        self.resize(1000, 700)
        # Global stylesheet for a clean look
        self.setStyleSheet('''
            QMainWindow { background: #f0f0f0; }
            QPushButton { background: #4B9CD3; color: white; padding: 8px; border-radius: 4px; }
            QPushButton:hover { background: #358ac3; }
            QLineEdit, QTextEdit { padding: 6px; border: 1px solid #ccc; border-radius: 4px; }
            QTabWidget::pane { border: none; }
            QTabBar::tab { background: #ddd; padding: 8px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background: white; }
            QGroupBox { border: 1px solid #ccc; border-radius: 4px; margin-top: 10px; }
            QGroupBox:title { subcontrol-origin: margin; left: 10px; padding: 0px 5px; }
            QLabel { font-weight: bold; }
        ''')
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # --- Tab: Load & Build ---
        tab_load = QWidget()
        load_layout = QHBoxLayout()
        self.file_edit = QLineEdit()
        btn_browse = QPushButton('Browse')
        btn_build = QPushButton('Build Graph')
        self.lbl_status = QLabel('No graph loaded yet.')
        load_layout.addWidget(self.file_edit)
        load_layout.addWidget(btn_browse)
        load_layout.addWidget(btn_build)
        load_layout.addWidget(self.lbl_status)
        tab_load.setLayout(load_layout)
        self.tabs.addTab(tab_load, 'Load')
        btn_browse.clicked.connect(self.browse_file)
        btn_build.clicked.connect(self.build_graph)

        # --- Tab: Visualization ---
        tab_vis = QWidget()
        vis_layout = QVBoxLayout()
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        btn_refresh = QPushButton('Refresh Preview')
        btn_save = QPushButton('Save Image...')
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(btn_refresh)
        btn_layout.addWidget(btn_save)
        vis_layout.addWidget(self.canvas)
        vis_layout.addLayout(btn_layout)
        tab_vis.setLayout(vis_layout)
        self.tabs.addTab(tab_vis, 'Graph')
        btn_refresh.clicked.connect(self.refresh_preview)
        btn_save.clicked.connect(self.save_image)

        # --- Tab: Bridge Words ---
        tab_bridge = QWidget()
        gb_bridge = QGroupBox('Query Bridge Words')
        b_layout = QHBoxLayout()
        self.edit_w1 = QLineEdit()
        self.edit_w2 = QLineEdit()
        btn_query = QPushButton('Query')
        self.txt_bridge = QTextEdit()
        self.txt_bridge.setReadOnly(True)
        b_layout.addWidget(QLabel('Word1:'))
        b_layout.addWidget(self.edit_w1)
        b_layout.addWidget(QLabel('Word2:'))
        b_layout.addWidget(self.edit_w2)
        b_layout.addWidget(btn_query)
        gb_bridge.setLayout(b_layout)
        lay_bridge = QVBoxLayout()
        lay_bridge.addWidget(gb_bridge)
        lay_bridge.addWidget(self.txt_bridge)
        tab_bridge.setLayout(lay_bridge)
        self.tabs.addTab(tab_bridge, 'Bridge')
        btn_query.clicked.connect(self.query_bridge)

        # --- Tab: Generate Text ---
        tab_gen = QWidget()
        gb_input = QGroupBox('Input Text')
        self.txt_input = QTextEdit()
        gb_input.setLayout(QVBoxLayout())
        gb_input.layout().addWidget(self.txt_input)
        btn_gen = QPushButton('Generate')
        self.txt_output = QTextEdit()
        self.txt_output.setReadOnly(True)
        lay_gen = QVBoxLayout()
        lay_gen.addWidget(gb_input)
        lay_gen.addWidget(btn_gen)
        lay_gen.addWidget(QLabel('Generated Text:'))
        lay_gen.addWidget(self.txt_output)
        tab_gen.setLayout(lay_gen)
        self.tabs.addTab(tab_gen, 'Generate')
        btn_gen.clicked.connect(self.generate_text)

        # --- Tab: Path & PageRank ---
        tab_path = QWidget()
        gb_short = QGroupBox('Shortest Path')
        sp_layout = QHBoxLayout()
        self.edit_src = QLineEdit()
        self.edit_dst = QLineEdit()
        btn_sp = QPushButton('Calc Path')
        sp_layout.addWidget(QLabel('Source:'))
        sp_layout.addWidget(self.edit_src)
        sp_layout.addWidget(QLabel('Target:'))
        sp_layout.addWidget(self.edit_dst)
        sp_layout.addWidget(btn_sp)
        gb_short.setLayout(sp_layout)

        gb_pr = QGroupBox('PageRank')
        pr_layout = QHBoxLayout()
        self.edit_pr = QLineEdit()
        btn_pr = QPushButton('Calc PR')
        pr_layout.addWidget(QLabel('Word:'))
        pr_layout.addWidget(self.edit_pr)
        pr_layout.addWidget(btn_pr)
        gb_pr.setLayout(pr_layout)

        self.txt_pathrank = QTextEdit()
        self.txt_pathrank.setReadOnly(True)
        lay_path = QVBoxLayout()
        lay_path.addWidget(gb_short)
        lay_path.addWidget(gb_pr)
        lay_path.addWidget(self.txt_pathrank)
        tab_path.setLayout(lay_path)
        self.tabs.addTab(tab_path, 'Path/PR')
        btn_sp.clicked.connect(self.calc_path)
        btn_pr.clicked.connect(self.calc_pagerank)

        # --- Tab: Random Walk ---
        tab_rw = QWidget()
        rw_layout = QVBoxLayout()
        btn_rw = QPushButton('Random Walk')
        self.txt_rw = QTextEdit()
        self.txt_rw.setReadOnly(True)
        rw_layout.addWidget(btn_rw)
        rw_layout.addWidget(self.txt_rw)
        tab_rw.setLayout(rw_layout)
        self.tabs.addTab(tab_rw, 'Random Walk')
        btn_rw.clicked.connect(self.do_random_walk)

    def browse_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Text File', '', 'Text Files (*.txt);;All Files (*)')
        if fname:
            self.file_edit.setText(fname)

    def build_graph(self):
        path = self.file_edit.text().strip()
        if not path:
            QMessageBox.warning(self, 'Error', 'Please select a file first.')
            return
        try:
            n = self.analyzer.build_graph_from_file(path)
            self.lbl_status.setText(f'Graph loaded: {n} vertices')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to build graph: {e}')

    def refresh_preview(self):
        if not self.analyzer.adj:
            QMessageBox.warning(self, 'Error', 'Load a graph first.')
            return
        self.draw_graph_on_canvas()

    def draw_graph_on_canvas(self):
        G = nx.DiGraph()
        for u in self.analyzer.adj:
            for v, w in self.analyzer.neighbors(u).items():
                if w > 0:
                    G.add_edge(u, v, weight=w)
        fig = self.canvas.figure
        fig.clf()
        ax = fig.add_subplot(111)
        pos = nx.shell_layout(G) if len(G) < 15 else nx.kamada_kawai_layout(G)
        nx.draw_networkx_nodes(
            G, pos, node_size=600, node_color="#4B9CD3",
            alpha=0.9, linewidths=1.5, edgecolors="#2A2A2A", ax=ax
        )
        nx.draw_networkx_edges(
            G, pos, width=1.2, edge_color="#666666",
            arrowstyle="->", arrowsize=15,
            connectionstyle="arc3,rad=0.2", ax=ax
        )
        nx.draw_networkx_labels(
            G, pos, font_size=9, font_color="white",
            font_weight="bold", ax=ax
        )
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels,
            font_size=7, font_color="black",
            bbox=dict(facecolor="white", edgecolor="none",
                      alpha=0.7, boxstyle="round,pad=0.1"),
            rotate=False, label_pos=0.5, ax=ax
        )
        ax.set_facecolor("#F5F5F5")
        fig.tight_layout()
        self.canvas.draw()

    def save_image(self):
        fname, _ = QFileDialog.getSaveFileName(self, 'Save Graph Image', '', 'PNG Images (*.png);;All Files (*)')
        if fname:
            try:
                self.canvas.figure.savefig(fname)
                QMessageBox.information(self, 'Saved', f'Image saved to {fname}')
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to save image: {e}')

    def query_bridge(self):
        res = self.analyzer.query_bridge_words(self.edit_w1.text(), self.edit_w2.text())
        self.txt_bridge.setText(res)

    def generate_text(self):
        out = self.analyzer.generate_new_text(self.txt_input.toPlainText())
        self.txt_output.setText(out)

    def calc_path(self):
        res = self.analyzer.calc_shortest_path(self.edit_src.text(), self.edit_dst.text())
        self.txt_pathrank.setText(res)

    def calc_pagerank(self):
        pr = self.analyzer.cal_pagerank(self.edit_pr.text())
        if pr is None:
            self.txt_pathrank.setText('Word not in graph!')
        else:
            self.txt_pathrank.setText(f'PageRank({self.edit_pr.text().lower()}) = {pr:.4f}')

    def do_random_walk(self):
        res = self.analyzer.random_walk()
        self.txt_rw.setText(res)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = GraphAnalyzerGUI()
    win.show()
    sys.exit(app.exec_())
