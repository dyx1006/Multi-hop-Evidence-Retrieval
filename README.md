# FNLP Assignment 3: Multi-hop Evidence Retrieval

本项目实现 MuSiQue 多跳证据检索任务：对每个问题从全局段落语料库中检索 Top-5 支持段落，并用 Recall@5 评估。

## 环境

建议使用 Python 3.10+。

```bash
pip install -r requirements.txt
```

Dense Retrieval 默认使用 `Qwen/Qwen3-Embedding-0.6B`，可通过 `--model` 指定其他 SentenceTransformer 模型。

## 数据准备

```bash
python scripts/prepare_data.py
```

生成：

```text
data/corpus.jsonl
data/test_2hop.jsonl
data/test_3hop.jsonl
data/test_4hop.jsonl
```

如数据已存在，可跳过。

## BM25 检索

```bash
python scripts/run_bm25.py
```

输出：

```text
outputs/bm25/predictions_2hop.jsonl
outputs/bm25/predictions_3hop.jsonl
outputs/bm25/predictions_4hop.jsonl
```

## Dense 检索

先构建语料库向量和 FAISS 索引：

```bash
python scripts/build_index.py --batch-size 32 --device cuda:0
```


指定模型或更换模型时，需要在构建索引和检索时使用相同的 `--model`；如已有缓存，需加 `--force` 重建：

```bash
python scripts/build_index.py --model Qwen/Qwen3-Embedding-0.6B --device cuda:0 --force
python scripts/run_dense.py --model Qwen/Qwen3-Embedding-0.6B --device cuda:0 --batch-size 32
```

如安装了 `faiss-gpu`，也可以用 GPU 构建 FAISS 索引：

```bash
python scripts/build_index.py \
  --model Qwen/Qwen3-Embedding-0.6B \
  --device cuda:0 \
  --faiss-device cuda:0 \
  --batch-size 32 \
  --force
```

`run_dense.py` 默认会按测试集跳数选择查询 instruction；如需手动指定，可使用 `--query-prefix`。

输出：

```text
outputs/dense/predictions_2hop.jsonl
outputs/dense/predictions_3hop.jsonl
outputs/dense/predictions_4hop.jsonl
```

## 评估

检索脚本会自动打印各测试集 Recall@5，也可单独评估：

```bash
python scripts/evaluate.py \
  --prediction outputs/bm25/predictions_2hop.jsonl \
  --gold data/test_2hop.jsonl \
  --k 5
```

预测文件格式：

```json
{"id": "question_id", "retrieved_corpus_ids": [1, 2, 3, 4, 5]}
```

## 案例分析

```bash
python scripts/analyze_cases.py \
  --gold data/test_2hop.jsonl \
  --bm25 outputs/bm25/predictions_2hop.jsonl \
  --dense outputs/dense/predictions_2hop.jsonl \
  --limit 3
```

## 提交

压缩包应包含：

- 实验报告 PDF
- Python 源代码和 README
- `outputs/` 下的预测 JSONL 文件

无需提交原始数据集和模型权重。
