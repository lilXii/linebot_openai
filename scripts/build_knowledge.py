import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Tuple

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    if PdfReader is None:
        raise RuntimeError("pypdf 尚未安裝，無法解析 PDF。請先在 requirements.txt 加入 pypdf 並安裝依賴。")
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buffer = ""
    for paragraph in paragraphs:
        candidate = f"{buffer}\n\n{paragraph}".strip()
        if len(candidate) <= chunk_size:
            buffer = candidate
            continue
        if buffer:
            chunks.extend(split_long_text(buffer, chunk_size, overlap))
            buffer = ""
        chunks.extend(split_long_text(paragraph, chunk_size, overlap))
    if buffer:
        chunks.extend(split_long_text(buffer, chunk_size, overlap))
    return chunks


def split_long_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    segments: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        segments.append(text[start:end].strip())
        start = max(start + chunk_size - overlap, end)
    return [seg for seg in segments if seg]


def iter_documents(data_dir: Path) -> Iterable[Tuple[Path, str]]:
    for path in sorted(data_dir.rglob('*')):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == '.txt':
            yield path, read_txt(path)
        elif suffix == '.pdf':
            yield path, read_pdf(path)


def build_knowledge(data_dir: Path, output_path: Path, chunk_size: int, overlap: int) -> None:
    records = []
    for doc_path, raw_text in iter_documents(data_dir):
        normalized = normalize_text(raw_text)
        chunks = chunk_text(normalized, chunk_size, overlap)
        for idx, chunk in enumerate(chunks):
            records.append({
                "doc_path": str(doc_path.relative_to(data_dir)),
                "chunk_id": idx,
                "text": chunk,
            })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"輸出 {len(records)} 個 chunks -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="將 data 目錄下的 PDF/TXT 轉為 RAG JSONL 知識庫")
    parser.add_argument("--data-dir", default="data", type=Path, help="來源資料夾")
    parser.add_argument("--output", default="data/knowledge_base.jsonl", type=Path, help="輸出 JSONL")
    parser.add_argument("--chunk-size", type=int, default=800, help="每個 chunk 最大字元數")
    parser.add_argument("--overlap", type=int, default=100, help="chunk 重疊字元數")
    args = parser.parse_args()
    build_knowledge(args.data_dir, args.output, args.chunk_size, args.overlap)
