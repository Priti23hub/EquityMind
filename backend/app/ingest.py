"""
This is the offline part of RAG or you can say the scheduled piece of RAG.
This is where we will be ingesting the data into the vector database.
"""

import chromadb
import os
from chromadb.config import Settings
import sys
from pathlib import Path
import hashlib
from pypdf import PdfReader
from langchain_text_splitter import RecursiveCharacterTextSplitter


CHUNK_SIZE = 900
CHUNK_OVERLAP = 100

CHROMA_PATH = "./chroma_db"
COLLECTION = "filings"

def load_pages(path: Path) -> list[tuple[int, str]]:
    if path.suffix == ".pdf":
        return[(i+1, page.extract_text()) for i, page in enumerate(PdfReader(path).pages)]
    # if not pdf this is text file and just load text file as it is.
    return[(i+1, page) for i, page in enumerate(path.read_text().split("\f"))]

def main(folder: str) -> None:
    client = chromadb.PersistentClient(
        path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_or_create_collection(COLLECTION)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    ids, documents, metadata = [], [], []
    for path in sorted(Path(folder).iterdir()):
        if path.suffix not in {".pdf", ".txt"} or path.name == "README.md":
            continue
        for page, text in load_pages(path):
            for chunk in splitter.split_text(text):
                ids.append(hashlib.sha1(f'{path.name}-{page}-{chunk}'.encode()).hexdigest()[:16])
                documents.append(chunk)
                metadata.append({"source": path.name, "page": page})
        
    if not documents:
        print(f"No pdf and no text files found in folder.")
    collection.upsert(ids=ids, documents=documents, metadatas=metadata)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../data/filings")
