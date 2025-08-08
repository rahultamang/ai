import argparse
import os
from rich import print

from friend_ai.config import ConfigLoader
from friend_ai.memory import MemoryStore


def main():
    parser = argparse.ArgumentParser(description="Memory demo: add or query local memory store")
    parser.add_argument("--text", type=str, help="Text to store as a memory")
    parser.add_argument("--query", type=str, help="Query text for semantic search")
    parser.add_argument("--top_k", type=int, default=None, help="Number of results to return")
    args = parser.parse_args()

    cfg = ConfigLoader.load()

    store = MemoryStore(
        persist_dir=cfg.app.db_dir,
        collection_name=cfg.memory.collection_name,
        embedding_model=cfg.memory.embedding_model,
    )

    if args.text:
        item = store.add(args.text, metadata={"source": "user"})
        print({"added": item})
    if args.query:
        top_k = args.top_k or cfg.memory.top_k_default
        results = store.query(args.query, top_k=top_k)
        for idx, item in enumerate(results, start=1):
            print(f"{idx}. {item.text}  [id={item.id}]")


if __name__ == "__main__":
    main()