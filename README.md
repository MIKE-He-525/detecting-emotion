# Detecting Emotion

基于 Twitter 推文的三分类情感分析。Notebook 训练，PyQt6 桌面推理。

## 这是什么

对英文推文进行情感分析，将 13 类细粒度标签合并为 3 类：`positive` / `negative` / `neutral`。

使用 TF-IDF 特征提取 + 逻辑回归分类器，在 40,000 条推文上训练，提供 PyQt6 桌面应用实时预测。

## 特性

- **简单模型**：TF-IDF + 逻辑回归，训练快速，无需 GPU
- **桌面应用**：PyQt6 GUI，输入文本即可查看情感分类结果
- **可复现训练**：Jupyter Notebook 记录完整训练流程，包含数据预处理、模型训练、评估和保存

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

在 Jupyter 中打开 `train.ipynb`，依次运行所有单元格。训练完成后会生成 `tfidf_model.pkl` 和带时间戳的备份目录。

### 3. 启动桌面应用

```bash
python app.py
```

在文本框中输入英文文本，点击 **Analyze Sentiment** 查看预测结果。

## 情感类别

| 标签 | 原始标签 |
|---|---|
| positive | happiness, love, fun, enthusiasm, relief, surprise |
| negative | sadness, hate, anger, boredom, empty, worry |
| neutral | neutral |

## 模型效果

在 8,000 条测试集上：

- **准确率**: 53.25%
- **Macro F1**: 0.52

## 项目结构

```
.
├── train.ipynb                 # 模型训练笔记本
├── app.py                      # PyQt6 桌面应用
├── config.py                   # 配置（路径、标签映射、训练参数）
├── utils.py                    # 文本清洗与预测器
├── tweet_emotions.csv          # 数据集（40,000 条推文）
├── en_stopwords.txt            # 英文停用词表
├── tfidf_model.pkl             # 训练好的模型
├── model_artifacts_<timestamp> # 训练备份（模型 + 评估报告）
└── requirements.txt            # 依赖列表
```

## 依赖

- Python >= 3.8
- PyQt6 >= 6.5
- pandas >= 2.0
- scikit-learn >= 1.3
- joblib >= 1.3
- ipykernel >= 6.0（Notebook 训练需要）

---

**克隆仓库**

```bash
git clone https://github.com/MIKE-He-525/detecting-emotion.git
cd detecting-emotion
```
