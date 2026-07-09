"""项目配置（路径、标签映射、训练参数）。"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "tweet_emotions.csv")
STOPWORDS_PATH = os.path.join(BASE_DIR, "en_stopwords.txt")
TFIDF_MODEL_PATH = os.path.join(BASE_DIR, "tfidf_model.pkl")
MODEL_ARTIFACTS_PREFIX = "model_artifacts_"

TEST_SIZE = 0.2
RANDOM_STATE = 42

# 13 类合并为 3 类
LABEL_GROUPS = {
    "positive": [
        "happiness", "love", "fun", "enthusiasm", "relief", "surprise",
    ],
    "negative": [
        "sadness", "hate", "anger", "boredom", "empty", "worry",
    ],
    "neutral": ["neutral"],
}
