#!/usr/bin/env python3
"""
Bulk ingest documents into the FAISS RAG index.

Usage:
    python scripts/ingest_docs.py ./my_papers/
    python scripts/ingest_docs.py ./my_papers/ --extensions pdf txt
"""

import argparse
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain.schema import Document
from src.rag.pipeline import RAGPipeline
from src.utils.helpers import setup_logging


def load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        logging.warning("Could not read PDF %s: %s", path, e)
        return ""


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into RAG index")
    parser.add_argument("folder", type=str, help="Path to folder containing documents")
    parser.add_argument("--extensions", nargs="+", default=["pdf", "txt"],
                        help="File extensions to ingest (default: pdf txt)")
    args = parser.parse_args()

    setup_logging(logging.INFO)
    folder = Path(args.folder)

    if not folder.exists():
        print(f"❌ Folder not found: {folder}")
        sys.exit(1)

    loaders = {"txt": load_txt, "pdf": load_pdf}
    docs = []

    for ext in args.extensions:
        for path in sorted(folder.rglob(f"*.{ext}")):
            loader = loaders.get(ext)
            if not loader:
                continue
            content = loader(path)
            if content.strip():
                docs.append(Document(page_content=content, metadata={"source": path.name}))
                print(f"  📄 Loaded: {path.name}")

    if not docs:
        print("⚠️  No documents found.")
        sys.exit(0)

    print(f"\n⏳ Ingesting {len(docs)} documents...")
    rag = RAGPipeline()
    chunks = rag.ingest(docs)
    print(f"✅ Done — {chunks} chunks indexed from {len(docs)} documents.")


if __name__ == "__main__":
    main()
