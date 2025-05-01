# main.py
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont
from qt_material import apply_stylesheet
from ui_mainwindow import MainWindow

def main():
    # 1) 创建 QApplication
    app = QApplication(sys.argv)

    # 2) 全局字体
    app.setFont(QFont("Microsoft YaHei", 12))

    # 3) 先应用 qt_material 主题
    #    主题列表见 qt_material/themes
    apply_stylesheet(app, theme='dark_teal.xml')

    # 4) 再叠加一点自定义的 CSS（不会覆盖 qt_material 的核心变量，只做微调）
    app.setStyleSheet("""
        /* 菜单栏 */
        QMenuBar {
            background-color: #263238;
            color: #ECEFF1;
        }
        QMenuBar::item {
            spacing: 10px;
            padding: 4px 12px;
        }
        QMenuBar::item:selected {
            background-color: #37474F;
        }
        /* 下拉菜单 */
        QMenu {
            background-color: #263238;
            color: #ECEFF1;
            padding: 6px;
        }
        QMenu::item:selected {
            background-color: #37474F;
        }
        /* 按钮 */
        QPushButton {
            border-radius: 6px;
            padding: 8px 16px;
        }
        QPushButton:hover {
            background-color: #546E7A;
        }
        /* 输入框 */
        QLineEdit, QTextEdit, QPlainTextEdit {
            border: 1px solid #455A64;
            border-radius: 4px;
            padding: 4px;
        }
    """)

    # 5) 创建并展示主窗口
    win = MainWindow()
    win.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
