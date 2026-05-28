"""
Build FAISS index for the global corpus using Qwen3-Embedding-0.6B.

Encodes all corpus paragraphs and saves:
  1. data/corpus_embeddings.npy  - (N, dim) float32 embeddings
  2. data/corpus.faiss           - FAISS IndexFlatIP index

Usage:
    python scripts/build_index.py [--batch-size 32] [--devices cuda:0,cuda:1,cuda:2,cuda:3]
"""

import argparse
import json
import multiprocessing as mp
import shutil
import time
from pathlib import Path

import numpy as np

#import os
#os.environ["TRANSFORMERS_NO_TF"] = "1"

from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_evenly(items, num_chunks):
    chunk_size = (len(items) + num_chunks - 1) // num_chunks
    return [
        items[i * chunk_size : min((i + 1) * chunk_size, len(items))]
        for i in range(num_chunks)
    ]


def encode_chunk(worker_id, device, texts, batch_size, output_path):

    print(
        f"[worker {worker_id}] Loading model: {MODEL_NAME} "
        f"(device={device}, n={len(texts)})"
    )
    model = SentenceTransformer(MODEL_NAME, device=device)

    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    np.save(output_path, embeddings)
    print(
        f"[worker {worker_id}] Saved {embeddings.shape} to {output_path} "
        f"in {time.time() - t0:.1f}s"
    )
    return worker_id, embeddings.shape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--devices",
        type=str,
        default="cuda:0,cuda:1,cuda:2,cuda:3",
        help="Comma-separated devices used to encode corpus shards.",
    )
    args = parser.parse_args()

    # --- Step 1: Encode corpus ---
    emb_path = DATA_DIR / "corpus_embeddings.npy"
    if emb_path.exists():
        print(f"[skip] Embeddings already exist: {emb_path}")
        embeddings = np.load(emb_path)
    else:
        print("Loading corpus...")
        corpus = load_jsonl(DATA_DIR / "corpus.jsonl")
        corpus_texts = [f"{c['title']}: {c['text']}" for c in corpus]
        print(f"Corpus size: {len(corpus)} paragraphs")

        devices = [device.strip() for device in args.devices.split(",") if device.strip()]
        if not devices:
            raise ValueError("--devices must contain at least one device")

        part_dir = DATA_DIR / "corpus_embeddings.parts"
        if part_dir.exists():
            shutil.rmtree(part_dir)
        part_dir.mkdir(parents=True)

        chunks = split_evenly(corpus_texts, len(devices))
        tasks = [
            (
                worker_id,
                device,
                chunk,
                args.batch_size,
                part_dir / f"part_{worker_id:02d}.npy",
            )
            for worker_id, (device, chunk) in enumerate(zip(devices, chunks))
            if chunk
        ]

        print(
            f"Encoding corpus on {len(tasks)} workers "
            f"(devices={','.join(device for _, device, _, _, _ in tasks)}, "
            f"batch_size={args.batch_size})..."
        )
        t0 = time.time()
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(tasks)) as pool:
            for worker_id, shape in pool.starmap(encode_chunk, tasks):
                print(f"[worker {worker_id}] Finished shape={shape}")

        print("Merging embedding shards...")
        embeddings = np.concatenate(
            [np.load(part_dir / f"part_{worker_id:02d}.npy") for worker_id in range(len(tasks))],
            axis=0,
        )
        print(f"Done in {time.time() - t0:.1f}s, shape={embeddings.shape}")

        np.save(emb_path, embeddings)
        print(f"Saved embeddings to {emb_path}")
        shutil.rmtree(part_dir)

    # --- Step 2: Build FAISS index ---
    index_path = DATA_DIR / "corpus.faiss"
    if index_path.exists():
        print(f"[skip] FAISS index already exists: {index_path}")
        return

    import faiss

    dim = embeddings.shape[1]
    print(f"Building FAISS IndexFlatIP (dim={dim}, n={embeddings.shape[0]})...")
    t0 = time.time()
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"Index built in {time.time() - t0:.2f}s, total={index.ntotal}")

    faiss.write_index(index, str(index_path))
    print(f"Saved FAISS index to {index_path}")


if __name__ == "__main__":
    main()
