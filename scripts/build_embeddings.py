import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List

import openai


def load_documents(path: Path) -> List[Dict[str, str]]:
    documents: List[Dict[str, str]] = []
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = record.get("text", "").strip()
                if not text:
                    continue
                documents.append({
                    "text": text,
                    "doc_hash": hashlib.md5(text.encode("utf-8")).hexdigest(),
                })
    else:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
        for chunk in chunks:
            documents.append({
                "text": chunk,
                "doc_hash": hashlib.md5(chunk.encode("utf-8")).hexdigest(),
            })
    return documents


def build_embeddings(documents: List[Dict[str, str]], model: str, batch_size: int) -> List[List[float]]:
    embeddings: List[List[float]] = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        response = openai.Embedding.create(
            model=model,
            input=[doc["text"] for doc in batch],
        )
        sorted_data = sorted(response["data"], key=lambda item: item["index"])
        embeddings.extend([item["embedding"] for item in sorted_data])
    return embeddings


def write_cache(documents: List[Dict[str, str]], embeddings: List[List[float]], model: str, output_path: Path) -> None:
    payload = {
        "model": model,
        "documents": [
            {"doc_hash": doc["doc_hash"], "embedding": emb}
            for doc, emb in zip(documents, embeddings)
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"寫入 {output_path}, 共 {len(documents)} 筆 embeddings")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="預先產生 RAG embeddings cache")
    parser.add_argument("--knowledge", default="data/knowledge_base.jsonl", type=Path, help="知識庫路徑")
    parser.add_argument("--output", type=Path, help="輸出 cache 檔案路徑，預設為 <knowledge>.embeddings.json")
    parser.add_argument("--model", default="text-embedding-3-small", help="OpenAI embedding 模型")
    parser.add_argument("--batch-size", type=int, default=64, help="每批次送入 API 的文本數量")
    args = parser.parse_args()

    if not openai.api_key:
        openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise SystemExit("請先設定 OPENAI_API_KEY 環境變數")

    documents = load_documents(args.knowledge)
    if not documents:
        raise SystemExit("找不到可用的文本 chunk")
    print(f"載入 {len(documents)} 筆資料，準備使用 {args.model} 建立 embeddings")
    embeddings = build_embeddings(documents, args.model, args.batch_size)
    output_path = args.output or Path(f"{args.knowledge}.embeddings.json")
    write_cache(documents, embeddings, args.model, output_path)
