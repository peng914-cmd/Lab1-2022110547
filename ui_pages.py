# ui_pages.py
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QLineEdit, QTextEdit, QPlainTextEdit, QFileDialog
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

class GraphViewPage(QWidget):
    def __init__(self, analyzer):
        super().__init__()
        self.analyzer = analyzer
        layout = QVBoxLayout()
        layout.addWidget(QLabel("邻接表："))
        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        layout.addWidget(self.view)

        btn = QPushButton("刷新")
        btn.clicked.connect(self.refresh)
        layout.addWidget(btn)
        self.setLayout(layout)

    def refresh(self):
        self.view.setPlainText(self.analyzer.adjacency_str())

class QueryBridgePage(QWidget):
    def __init__(self, analyzer):
        super().__init__()
        self.analyzer = analyzer
        layout = QVBoxLayout()
        layout.addWidget(QLabel("输入词 1："))
        self.w1 = QLineEdit()
        layout.addWidget(self.w1)

        layout.addWidget(QLabel("输入词 2："))
        self.w2 = QLineEdit()
        layout.addWidget(self.w2)

        btn = QPushButton("查询桥接词")
        btn.clicked.connect(self.on_query)
        layout.addWidget(btn)

        self.result = QLabel("")
        layout.addWidget(self.result)
        self.setLayout(layout)

    def on_query(self):
        res = self.analyzer.query_bridge_words(self.w1.text(), self.w2.text())
        self.result.setText(res)

class GenerateTextPage(QWidget):
    def __init__(self, analyzer):
        super().__init__()
        self.analyzer = analyzer
        layout = QVBoxLayout()

        layout.addWidget(QLabel("输入原文："))
        self.input = QTextEdit()
        layout.addWidget(self.input)

        btn = QPushButton("生成新文本")
        btn.clicked.connect(self.on_generate)
        layout.addWidget(btn)

        layout.addWidget(QLabel("生成结果："))
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output)

        self.setLayout(layout)

    def on_generate(self):
        self.output.setPlainText(
            self.analyzer.generate_new_text(self.input.toPlainText())
        )

class ShortestPathPage(QWidget):
    def __init__(self, analyzer):
        super().__init__()
        self.analyzer = analyzer
        layout = QVBoxLayout()

        layout.addWidget(QLabel("起点词："))
        self.src = QLineEdit()
        layout.addWidget(self.src)

        layout.addWidget(QLabel("终点词："))
        self.dst = QLineEdit()
        layout.addWidget(self.dst)

        btn = QPushButton("计算最短路径")
        btn.clicked.connect(self.on_find)
        layout.addWidget(btn)

        self.result = QLabel("")
        layout.addWidget(self.result)
        self.setLayout(layout)

    def on_find(self):
        self.result.setText(
            self.analyzer.calc_shortest_path(self.src.text(), self.dst.text())
        )

class PageRankPage(QWidget):
    def __init__(self, analyzer):
        super().__init__()
        self.analyzer = analyzer
        layout = QVBoxLayout()

        layout.addWidget(QLabel("词语："))
        self.word = QLineEdit()
        layout.addWidget(self.word)

        btn = QPushButton("计算 PageRank")
        btn.clicked.connect(self.on_compute)
        layout.addWidget(btn)

        self.result = QLabel("")
        layout.addWidget(self.result)
        self.setLayout(layout)

    def on_compute(self):
        pr = self.analyzer.cal_pagerank(self.word.text())
        if pr is None:
            self.result.setText("该词不在图中！")
        else:
            self.result.setText(f"PR({self.word.text().lower()}) = {pr:.4f}")

class RandomWalkPage(QWidget):
    def __init__(self, analyzer):
        super().__init__()
        self.analyzer = analyzer
        layout = QVBoxLayout()

        btn = QPushButton("开始随机游走")
        btn.clicked.connect(self.on_walk)
        layout.addWidget(btn)

        self.result = QTextEdit()
        self.result.setReadOnly(True)
        layout.addWidget(self.result)
        self.setLayout(layout)

    def on_walk(self):
        self.result.setPlainText(self.analyzer.random_walk())

class DrawGraphPage(QWidget):
    def __init__(self, analyzer):
        super().__init__()
        self.analyzer = analyzer
        layout = QVBoxLayout()

        self.canvas = FigureCanvas(plt.figure(figsize=(8, 6)))
        layout.addWidget(self.canvas)

        btn_save = QPushButton("保存为图片")
        btn_save.clicked.connect(self.save_image)
        layout.addWidget(btn_save)

        self.setLayout(layout)
        self.redraw()

    def redraw(self):
        fig = self.canvas.figure
        fig.clf()
        ax = fig.add_subplot(111)
        # 将网络图画到 ax
        self.analyzer.draw_to_ax(ax)
        self.canvas.draw()

    def save_image(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存图像", "graph.png", "PNG 文件 (*.png)"
        )
        if path:
            self.canvas.figure.savefig(path)
