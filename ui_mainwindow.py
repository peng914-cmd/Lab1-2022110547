# ui_mainwindow.py
import os
from PyQt5.QtWidgets import (
    QMainWindow, QAction, QFileDialog, QStackedWidget, QMessageBox
)
from text_analyzer import TextGraphAnalyzer
from ui_pages import (
    GraphViewPage, QueryBridgePage, GenerateTextPage,
    ShortestPathPage, PageRankPage, RandomWalkPage, DrawGraphPage
)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("文本图分析器")
        self.setGeometry(100, 100, 1000, 700)

        # 核心分析器
        self.analyzer = TextGraphAnalyzer()

        # 栈式布局容纳所有页面
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # 中文化页面名称
        self.pages = {
            "查看邻接表":    GraphViewPage(self.analyzer),
            "查询桥接词":    QueryBridgePage(self.analyzer),
            "生成新文本":    GenerateTextPage(self.analyzer),
            "最短路径":      ShortestPathPage(self.analyzer),
            "PageRank 计算": PageRankPage(self.analyzer),
            "随机游走":      RandomWalkPage(self.analyzer),
            "绘制图形":      DrawGraphPage(self.analyzer),
        }
        for page in self.pages.values():
            self.stack.addWidget(page)

        self._create_menu()

    def _create_menu(self):
        menu = self.menuBar()

        # —— 文件 菜单
        file_menu = menu.addMenu("文件(&F)")
        load_action = QAction("加载文本文件...", self)
        load_action.triggered.connect(self.load_file)
        file_menu.addAction(load_action)
        file_menu.addSeparator()
        exit_action = QAction("退出(&Q)", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # —— 导航 菜单
        nav_menu = menu.addMenu("导航(&N)")
        for name, page in self.pages.items():
            act = QAction(name, self)
            # 如果是“绘制图形”，切换时要先刷新
            if name == "绘制图形":
                act.triggered.connect(lambda _, w=page: (self.stack.setCurrentWidget(w), w.redraw()))
            else:
                act.triggered.connect(lambda _, w=page: self.stack.setCurrentWidget(w))
            nav_menu.addAction(act)

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择文本文件", "", "文本文件 (*.txt)"
        )
        if path:
            try:
                self.analyzer.build_graph_from_file(path)
                QMessageBox.information(
                    self, "成功", f"已加载：{os.path.basename(path)}"
                )
                # 自动切到查看邻接表并刷新
                page = self.pages["查看邻接表"]
                self.stack.setCurrentWidget(page)
                page.refresh()
                # 同时刷新绘图页面，以便能看到新图
                self.pages["绘制图形"].redraw()
            except Exception as e:
                QMessageBox.critical(self, "错误", str(e))
