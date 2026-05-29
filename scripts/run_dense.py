"""Run dense retrieval with a SentenceTransformer model and a FAISS index."""

import argparse
from pathlib import Path

from evaluate import evaluate
from retrieval_utils import (
    DATA_DIR,
    DEFAULT_DENSE_MODEL,
    DEFAULT_TEST_FILES,
    hop_label,
    load_jsonl,
    prediction_path,
    save_jsonl,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda iterable, **_: iterable


MULTIHOP_QUERY_PREFIX = (
    "Instruct: Retrieve the supporting paragraphs needed to answer the multi-hop question.\n"
    "Query: "
)
WEB_QUERY_PREFIX = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "


def default_query_prefix_for_file(test_file):
    if "4hop" in Path(test_file).stem:
        return WEB_QUERY_PREFIX
    return MULTIHOP_QUERY_PREFIX


def check_embedding_metadata(embeddings_path, model_name):
    meta_path = Path(embeddings_path).with_name("corpus_embeddings.meta.json")
    if not meta_path.exists():
        print("[warn] embedding metadata is missing; cannot verify model/index compatibility.")
        return

    import json

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("model") != model_name:
        raise ValueError(
            f"Cached embeddings were built with {meta.get('model')!r}, "
            f"but --model is {model_name!r}. Rebuild with scripts/build_index.py --force."
        )


def load_or_build_index(index_path, embeddings_path, model_name):
    import numpy as np

    check_embedding_metadata(embeddings_path, model_name)
    index_path = Path(index_path)
    try:
        import faiss
    except ImportError:
        faiss = None

    if index_path.exists() and faiss is not None:
        return "faiss", faiss.read_index(str(index_path))

    embeddings_path = Path(embeddings_path)
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Missing FAISS index {index_path} and embeddings {embeddings_path}. "
            "Run scripts/build_index.py first."
        )

    embeddings = np.load(embeddings_path).astype(np.float32)
    if faiss is None:
        print("[warn] faiss is not installed; using exact NumPy inner-product search.")
        return "numpy", embeddings

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(index_path))
    return "faiss", index


def encode_queries(model, questions, batch_size, query_prefix):
    import numpy as np

    texts = [f"{query_prefix}{item['question']}" if query_prefix else item["question"] for item in questions]
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)


def search_index(index_kind, index, query_embeddings, k):
    import numpy as np

    if index_kind == "faiss":
        _, indices = index.search(query_embeddings, k)
        return indices

    scores = query_embeddings @ index.T
    kth = min(k, scores.shape[1] - 1)
    candidate_indices = np.argpartition(-scores, kth=kth, axis=1)[:, :k]
    candidate_scores = np.take_along_axis(scores, candidate_indices, axis=1)
    order = np.argsort(-candidate_scores, axis=1)
    return np.take_along_axis(candidate_indices, order, axis=1)


def select_diverse_indices(row_indices, corpus_titles, k):
    selected = []
    seen_titles = set()

    for idx in row_indices:
        if idx < 0:
            continue
        title = corpus_titles[idx]
        if title in seen_titles:
            continue
        selected.append(idx)
        seen_titles.add(title)
        if len(selected) == k:
            return selected

    for idx in row_indices:
        if idx < 0 or idx in selected:
            continue
        selected.append(idx)
        if len(selected) == k:
            break

    return selected


def retrieve_dataset(
    model,
    index_kind,
    index,
    corpus_ids,
    corpus_titles,
    questions,
    batch_size,
    k,
    candidate_k,
    query_prefix,
    title_diversity,
):
    query_embeddings = encode_queries(model, questions, batch_size, query_prefix)
    indices = search_index(index_kind, index, query_embeddings, candidate_k)
    predictions = []
    for item, row_indices in tqdm(zip(questions, indices), total=len(questions), desc="Dense retrieval"):
        if title_diversity:
            row_indices = select_diverse_indices(row_indices, corpus_titles, k)
        else:
            row_indices = row_indices[:k]
        predictions.append(
            {
                "id": item["id"],
                "retrieved_corpus_ids": [int(corpus_ids[idx]) for idx in row_indices if idx >= 0],
            }
        )
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Dense retrieval for MuSiQue evidence retrieval.")
    parser.add_argument(
        "--model",
        default=DEFAULT_DENSE_MODEL,
        help=f"SentenceTransformer embedding model (default: {DEFAULT_DENSE_MODEL})",
    )
    parser.add_argument("--device", default=None, help="Example: cuda:0, mps, cpu. Auto-selected if omitted.")
    parser.add_argument("--corpus", default=str(DATA_DIR / "corpus.jsonl"))
    parser.add_argument("--index", default=str(DATA_DIR / "corpus.faiss"))
    parser.add_argument("--embeddings", default=str(DATA_DIR / "corpus_embeddings.npy"))
    parser.add_argument("--test-files", nargs="+", default=[str(p) for p in DEFAULT_TEST_FILES])
    parser.add_argument("--output-dir", default="outputs/dense")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=None)
    parser.add_argument(
        "--title-diversity",
        action="store_true",
        help="Select top-k from a larger candidate set while preferring different document titles.",
    )
    parser.add_argument(
        "--no-title-diversity",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--query-prefix",
        default=None,
        help="Query instruction prefix. If omitted, choose a default prefix by hop count.",
    )
    args = parser.parse_args()
    title_diversity = args.title_diversity and not args.no_title_diversity
    candidate_k = max(args.k, args.candidate_k or (50 if title_diversity else args.k))

    from sentence_transformers import SentenceTransformer

    corpus = load_jsonl(args.corpus)
    corpus_ids = [row["corpus_id"] for row in corpus]
    corpus_titles = [row.get("title", "") for row in corpus]
    index_kind, index = load_or_build_index(args.index, args.embeddings, args.model)
    index_size = index.ntotal if index_kind == "faiss" else index.shape[0]
    if index_size != len(corpus_ids):
        raise ValueError(f"Index size ({index_size}) does not match corpus size ({len(corpus_ids)}).")

    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model, device=args.device)

    recalls = []
    for test_file in args.test_files:
        test_file = Path(test_file)
        questions = load_jsonl(test_file)
        out_path = prediction_path(args.output_dir, test_file)
        query_prefix = args.query_prefix
        if query_prefix is None:
            query_prefix = default_query_prefix_for_file(test_file)
        predictions = retrieve_dataset(
            model=model,
            index_kind=index_kind,
            index=index,
            corpus_ids=corpus_ids,
            corpus_titles=corpus_titles,
            questions=questions,
            batch_size=args.batch_size,
            k=args.k,
            candidate_k=candidate_k,
            query_prefix=query_prefix,
            title_diversity=title_diversity,
        )
        save_jsonl(predictions, out_path)
        recall, n_questions = evaluate(out_path, test_file, k=args.k)
        recalls.append(recall)
        print(f"{hop_label(test_file)} Recall@{args.k}: {recall:.4f} ({n_questions} questions) -> {out_path}")

    if recalls:
        print(f"Average Recall@{args.k}: {sum(recalls) / len(recalls):.4f}")


if __name__ == "__main__":
    main()
