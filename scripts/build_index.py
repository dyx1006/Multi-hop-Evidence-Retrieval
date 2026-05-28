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

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_evenly(items, num_chunks):
    chunk_size = (len(items) + num_chunks - 1) // num_chunks
    return [
        items[i * chunk_size : min((i + 1) * chunk_size, len(items))]
        for i in range(num_chunks)
    ]


def encode_chunk(worker_id, model_name, device, texts, batch_size, output_path):
    import numpy as np
    from sentence_transformers import SentenceTransformer

    print(
        f"[worker {worker_id}] Loading model: {model_name} "
        f"(device={device}, n={len(texts)})"
    )
    model = SentenceTransformer(model_name, device=device)

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


def parse_cuda_device_id(device):
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    raise ValueError(f"FAISS GPU device must look like 'cuda' or 'cuda:N', got {device!r}")


def build_faiss_index(faiss, embeddings, faiss_device):
    dim = embeddings.shape[1]
    print(
        f"Building FAISS IndexFlatIP "
        f"(dim={dim}, n={embeddings.shape[0]}, device={faiss_device})..."
    )
    t0 = time.time()

    if faiss_device == "cpu":
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
    else:
        if not hasattr(faiss, "StandardGpuResources"):
            raise RuntimeError(
                "This faiss installation does not include GPU support. "
                "Install faiss-gpu in the target CUDA environment, or use --faiss-device cpu."
            )
        gpu_id = parse_cuda_device_id(faiss_device)
        resources = faiss.StandardGpuResources()
        cpu_index = faiss.IndexFlatIP(dim)
        gpu_index = faiss.index_cpu_to_gpu(resources, gpu_id, cpu_index)
        gpu_index.add(embeddings)
        index = faiss.index_gpu_to_cpu(gpu_index)

    print(f"Index built in {time.time() - t0:.2f}s, total={index.ntotal}")
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild embeddings and index even if cached files already exist.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Comma-separated devices used to encode corpus shards.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Alias for --devices when using one device.",
    )
    parser.add_argument(
        "--faiss-device",
        type=str,
        default="cpu",
        help="Device used to build the FAISS index: cpu, cuda, or cuda:N.",
    )
    args = parser.parse_args()

    # --- Step 1: Encode corpus ---
    emb_path = DATA_DIR / "corpus_embeddings.npy"
    index_path = DATA_DIR / "corpus.faiss"
    meta_path = DATA_DIR / "corpus_embeddings.meta.json"
    if emb_path.exists() and not args.force:
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("model") != args.model:
                raise ValueError(
                    f"Cached embeddings were built with {meta.get('model')!r}, "
                    f"but --model is {args.model!r}. Rerun with --force to rebuild."
                )
        print(f"[skip] Embeddings already exist: {emb_path}")
        import numpy as np

        embeddings = np.load(emb_path)
    else:
        print("Loading corpus...")
        corpus = load_jsonl(DATA_DIR / "corpus.jsonl")
        corpus_texts = [f"{c['title']}: {c['text']}" for c in corpus]
        print(f"Corpus size: {len(corpus)} paragraphs")

        devices_arg = args.devices or args.device
        if devices_arg:
            devices = [device.strip() for device in devices_arg.split(",") if device.strip()]
        else:
            import torch

            devices = ["cuda:0"] if torch.cuda.is_available() else ["cpu"]
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
                args.model,
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
            f"(devices={','.join(device for _, _, device, _, _, _ in tasks)}, "
            f"batch_size={args.batch_size})..."
        )
        t0 = time.time()
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(tasks)) as pool:
            for worker_id, shape in pool.starmap(encode_chunk, tasks):
                print(f"[worker {worker_id}] Finished shape={shape}")

        print("Merging embedding shards...")
        import numpy as np

        embeddings = np.concatenate(
            [np.load(part_dir / f"part_{worker_id:02d}.npy") for worker_id in range(len(tasks))],
            axis=0,
        )
        print(f"Done in {time.time() - t0:.1f}s, shape={embeddings.shape}")

        np.save(emb_path, embeddings)
        print(f"Saved embeddings to {emb_path}")
        meta_path.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "num_embeddings": int(embeddings.shape[0]),
                    "embedding_dim": int(embeddings.shape[1]),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        shutil.rmtree(part_dir)

    # --- Step 2: Build FAISS index ---
    if index_path.exists() and not args.force:
        print(f"[skip] FAISS index already exists: {index_path}")
        return

    try:
        import faiss
    except ImportError:
        print("[warn] faiss is not installed; embeddings were saved but FAISS index was not built.")
        print("Install faiss-cpu or faiss-gpu, then rerun this script to create data/corpus.faiss.")
        return

    index = build_faiss_index(faiss, embeddings, args.faiss_device)

    faiss.write_index(index, str(index_path))
    print(f"Saved FAISS index to {index_path}")


if __name__ == "__main__":
    main()
