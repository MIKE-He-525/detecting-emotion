"""PyQt6 情感分析桌面应用。"""

import os
import sys
import warnings

from utils import SentimentPredictor

# 确保无论从何处启动，工作目录与导入路径均正确
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QApplication,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    print("错误：未安装 PyQt6。")
    print("请执行：conda activate coding && pip install PyQt6")
    sys.exit(1)

COLORS = {"positive": "#34c759", "negative": "#ff3b30", "neutral": "#8e8e93"}


class SentimentAnalyzer(QMainWindow):
    """情感分析主窗口，提供文本输入与情感预测展示。"""

    def __init__(self, predictor: SentimentPredictor):
        super().__init__()
        warnings.filterwarnings("ignore", message=".*iCCP.*")
        self.predictor = predictor
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Sentiment Analyzer")
        self.setMinimumSize(800, 600)
        self.setStyleSheet("""
            QMainWindow { background: #fff; }
            QLabel { color: #1d1d1f; font-size: 16px; }
            QTextEdit {
                border: 1px solid #d2d2d7; border-radius: 12px;
                padding: 10px; font-size: 14px; background: #f5f5f7;
                color: #000000;
            }
            QPushButton {
                background: #0071e3; color: white; border: none;
                border-radius: 12px; padding: 10px 20px; font-size: 14px;
            }
            QPushButton:hover { background: #0077ed; }
        """)

        layout = QVBoxLayout()
        title = QLabel("Analyze Your Text")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(title)

        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Enter your text here...")
        layout.addWidget(self.text_input)

        btn = QPushButton("Analyze Sentiment")
        btn.clicked.connect(self.analyze_sentiment)
        layout.addWidget(btn)

        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.result_label)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def analyze_sentiment(self):
        text = self.text_input.toPlainText()
        if not text:
            self.result_label.setText("Please enter some text")
            return
        sentiment = self.predictor.predict(text)
        color = COLORS.get(sentiment, "#1d1d1f")
        self.result_label.setStyleSheet(
            f"color: {color}; font-size: 18px; font-weight: bold;"
        )
        self.result_label.setText(f"Sentiment: {sentiment.title()}")


def main():
    app = QApplication(sys.argv)
    try:
        predictor = SentimentPredictor()
    except Exception as e:
        QMessageBox.critical(
            None,
            "模型加载失败",
            f"无法加载模型，请先运行 train.ipynb 训练。\n\n{e}",
        )
        sys.exit(1)

    window = SentimentAnalyzer(predictor)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
