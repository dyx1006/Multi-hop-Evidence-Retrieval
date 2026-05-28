"""Run BM25 sparse retrieval and write top-k prediction JSONL files."""

import argparse
from pathlib import Path

from evaluate import evaluate
from retrieval_utils import (
    DATA_DIR,
    DEFAULT_TEST_FILES,
    SimpleBM25,
    corpus_text,
    hop_label,
    load_jsonl,
    prediction_path,
    save_jsonl,
    tokenize,
    topk_indices,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda iterable, **_: iterable


def build_bm25(tokenized_corpus, k1, b):
    try:
        from rank_bm25 import BM25Okapi

        return BM25Okapi(tokenized_corpus, k1=k1, b=b)
    except ImportError:
        print("[warn] rank_bm25 is not installed; using built-in BM25 implementation.")
        return SimpleBM25(tokenized_corpus, k1=k1, b=b)


def retrieve_dataset(bm25, corpus_ids, questions, k):
    predictions = []
    for item in tqdm(questions, desc="BM25 retrieval"):
        query_tokens = tokenize(item["question"])
        scores = bm25.get_scores(query_tokens)
        top_indices = topk_indices(scores, k)
        predictions.append(
            {
                "id": item["id"],
                "retrieved_corpus_ids": [int(corpus_ids[idx]) for idx in top_indices],
            }
        )
    return predictions


def main():
    parser = argparse.ArgumentParser(description="BM25 sparse retrieval for MuSiQue evidence retrieval.")
    parser.add_argument("--corpus", default=str(DATA_DIR / "corpus.jsonl"))
    parser.add_argument("--test-files", nargs="+", default=[str(p) for p in DEFAULT_TEST_FILES])
    parser.add_argument("--output-dir", default="outputs/bm25")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument(
        "--title-weight",
        type=int,
        default=3,
        help="Repeat titles this many times when building document text.",
    )
    args = parser.parse_args()

    corpus = load_jsonl(args.corpus)
    corpus_ids = [row["corpus_id"] for row in corpus]
    tokenized_corpus = [tokenize(corpus_text(row, title_weight=args.title_weight)) for row in corpus]
    print(f"Loaded corpus: {len(corpus)} paragraphs")

    bm25 = build_bm25(tokenized_corpus, k1=args.k1, b=args.b)

    recalls = []
    for test_file in args.test_files:
        test_file = Path(test_file)
        questions = load_jsonl(test_file)
        out_path = prediction_path(args.output_dir, test_file)
        predictions = retrieve_dataset(bm25, corpus_ids, questions, args.k)
        save_jsonl(predictions, out_path)
        recall, n_questions = evaluate(out_path, test_file, k=args.k)
        recalls.append(recall)
        print(f"{hop_label(test_file)} Recall@{args.k}: {recall:.4f} ({n_questions} questions) -> {out_path}")

    if recalls:
        print(f"Average Recall@{args.k}: {sum(recalls) / len(recalls):.4f}")


if __name__ == "__main__":
    main()
