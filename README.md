# 情感分析（Sentiment Analyzer）

基于 Twitter 推文的三分类情感分析。训练用 Notebook，推理用 PyQt6 桌面应用。

**环境**：Anaconda `coding`

## 项目结构

```
.
├── train.ipynb          # 模型训练（Notebook）
├── app.py               # GUI 推理
├── config.py            # 配置
├── utils.py             # 文本处理 + 预测器
├── tweet_emotions.csv   # 数据集
├── en_stopwords.txt     # 停用词
├── tfidf_model.pkl      # 训练产物
└── requirements.txt
```

## 安装

```powershell
conda activate coding
pip install -r requirements.txt
python -m ipykernel install --user --name coding --display-name "Python (coding)"
```

## 使用

### 1. 训练模型

在项目根目录打开 `train.ipynb`，选择内核 **coding**，依次运行所有单元格。

### 2. 启动 GUI

```powershell
conda activate coding
python app.py
```

> 必须使用 `coding` 环境，系统自带的 Python 未安装 PyQt6，会导致 UI 无法启动。

## 情感类别

| 标签 | 原始标签 |
|---|---|
| positive | happiness, love, fun, enthusiasm, relief, surprise |
| negative | sadness, hate, anger, boredom, empty, worry |
| neutral | neutral |

## 模型效果

准确率 ~53%，Macro F1 ~0.52。

## 依赖

PyQt6、pandas、scikit-learn、joblib
