"""文本处理与模型推理工具。"""

import os
import re

import joblib

from config import (
    BASE_DIR,
    LABEL_GROUPS,
    MODEL_ARTIFACTS_PREFIX,
    STOPWORDS_PATH,
    TFIDF_MODEL_PATH,
)

RAW_TO_MERGED = {
    raw: merged
    for merged, labels in LABEL_GROUPS.items()
    for raw in labels
}


def load_stopwords() -> set[str]:
    with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
        return set(f.read().splitlines())


def clean_text(text: str, stopwords: set[str] | None = None) -> str:
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower().strip()
    if stopwords is None:
        stopwords = load_stopwords()
    return " ".join(word for word in text.split() if word not in stopwords)


def merge_label(raw_label: str) -> str:
    return RAW_TO_MERGED.get(raw_label, "neutral")


class SentimentPredictor:
    """加载 tfidf_model.pkl 并预测情感。"""

    def __init__(self, base_dir: str = BASE_DIR):
        self.pipeline = self._load_model(base_dir)

    def _load_model(self, base_dir: str):
        if os.path.exists(TFIDF_MODEL_PATH):
            return joblib.load(TFIDF_MODEL_PATH)

        model_dirs = sorted(
            d for d in os.listdir(base_dir)
            if d.startswith(MODEL_ARTIFACTS_PREFIX)
        )
        for model_dir in reversed(model_dirs):
            path = os.path.join(base_dir, model_dir, "tfidf_model.pkl")
            if os.path.exists(path):
                return joblib.load(path)

        raise FileNotFoundError(
            "未找到模型，请先运行 train.ipynb 完成训练。"
        )

    def predict(self, text: str) -> str:
        return self.pipeline.predict([clean_text(text)])[0]
