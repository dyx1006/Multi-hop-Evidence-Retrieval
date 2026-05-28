"""Find case studies comparing BM25 and dense retrieval predictions."""

import argparse
from pathlib import Path

from evaluate import compute_recall_at_k
from retrieval_utils import DATA_DIR, load_jsonl


def load_predictions(path):
    return {row["id"]: row["retrieved_corpus_ids"] for row in load_jsonl(path)}


def corpus_lookup(path):
    return {row["corpus_id"]: row for row in load_jsonl(path)}


def format_ids(ids, corpus):
    chunks = []
    for cid in ids:
        item = corpus.get(cid, {"title": "<missing>", "text": ""})
        text = item.get("text", "")
        if len(text) > 220:
            text = text[:217] + "..."
        chunks.append(f"  - {cid} | {item.get('title', '')}: {text}")
    return "\n".join(chunks)


def main():
    parser = argparse.ArgumentParser(description="Print BM25-vs-dense retrieval case studies.")
    parser.add_argument("--gold", required=True, help="Gold test JSONL, e.g. data/test_2hop.jsonl")
    parser.add_argument("--bm25", required=True, help="BM25 prediction JSONL")
    parser.add_argument("--dense", required=True, help="Dense prediction JSONL")
    parser.add_argument("--corpus", default=str(DATA_DIR / "corpus.jsonl"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    gold_rows = load_jsonl(args.gold)
    bm25 = load_predictions(args.bm25)
    dense = load_predictions(args.dense)
    corpus = corpus_lookup(args.corpus)

    bm25_better = []
    dense_better = []
    for row in gold_rows:
        gold_ids = row["gold_corpus_ids"]
        bm25_retrieved = bm25.get(row["id"], [])
        dense_retrieved = dense.get(row["id"], [])
        bm25_recall = compute_recall_at_k(gold_ids, bm25_retrieved, args.k)
        dense_recall = compute_recall_at_k(gold_ids, dense_retrieved, args.k)
        record = (row, bm25_recall, dense_recall, bm25_retrieved, dense_retrieved)
        if bm25_recall > dense_recall:
            bm25_better.append(record)
        elif dense_recall > bm25_recall:
            dense_better.append(record)

    def print_cases(title, cases):
        print(f"\n## {title}")
        for row, bm25_recall, dense_recall, bm25_ids, dense_ids in cases[: args.limit]:
            print(f"\nID: {row['id']}")
            print(f"Question: {row['question']}")
            print(f"Answer: {row.get('answer', '')}")
            print(f"BM25 Recall@{args.k}: {bm25_recall:.2f}; Dense Recall@{args.k}: {dense_recall:.2f}")
            print("Gold:")
            print(format_ids(row["gold_corpus_ids"], corpus))
            print("BM25 Top-5:")
            print(format_ids(bm25_ids[: args.k], corpus))
            print("Dense Top-5:")
            print(format_ids(dense_ids[: args.k], corpus))

    print_cases("BM25 better than Dense", bm25_better)
    print_cases("Dense better than BM25", dense_better)


if __name__ == "__main__":
    main()
