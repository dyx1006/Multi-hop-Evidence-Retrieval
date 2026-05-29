"""Shared utilities for FNLP Assignment 3 retrieval scripts."""

import json
import math
import re
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DENSE_MODEL = "Qwen/Qwen3-Embedding-0.6B"

DEFAULT_TEST_FILES = (
    DATA_DIR / "test_2hop.jsonl",
    DATA_DIR / "test_3hop.jsonl",
    DATA_DIR / "test_4hop.jsonl",
)

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def load_jsonl(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def corpus_text(item, title_weight=2):
    title = item.get("title", "")
    text = item.get("text", "")
    weighted_title = " ".join([title] * max(1, title_weight))
    return f"{weighted_title} {text}".strip()


def prediction_path(output_dir, test_file):
    stem = Path(test_file).stem.replace("test_", "predictions_")
    return Path(output_dir) / f"{stem}.jsonl"


def hop_label(test_file):
    name = Path(test_file).stem
    return name.replace("test_", "")


class SimpleBM25:
    """Small BM25 implementation used when rank_bm25 is unavailable."""

    def __init__(self, tokenized_corpus, k1=1.5, b=0.75):
        self.tokenized_corpus = tokenized_corpus
        self.k1 = k1
        self.b = b
        self.doc_len = [len(doc) for doc in tokenized_corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0
        self.term_freqs = [Counter(doc) for doc in tokenized_corpus]
        doc_freq = Counter()
        for doc in tokenized_corpus:
            doc_freq.update(set(doc))
        n_docs = len(tokenized_corpus)
        self.idf = {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def get_scores(self, query_tokens):
        scores = []
        for tf, dl in zip(self.term_freqs, self.doc_len):
            score = 0.0
            norm = self.k1 * (1.0 - self.b + self.b * dl / self.avgdl) if self.avgdl else self.k1
            for term in query_tokens:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                numerator = freq * (self.k1 + 1.0)
                denominator = freq + norm
                score += self.idf.get(term, 0.0) * numerator / denominator
            scores.append(score)
        return scores


def topk_indices(scores, k):
    indexed = enumerate(scores)
    return [idx for idx, _ in sorted(indexed, key=lambda item: item[1], reverse=True)[:k]]
