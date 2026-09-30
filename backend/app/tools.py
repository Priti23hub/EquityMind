import chromadb
from chromadb.config import Settings

from .guardrails import validate_tool_request


_filings = None

def _collections():
    global _filings
    if _filings is None:
        client = chromadb.PersistentClient(
            path="./chroma_db", settings=Settings(anonymized_telemetry=False)
        )
        _filings = client.get_or_create_collection("filings")

def search_filings(query: str, K: int = 10) -> list[dict] | dict:
    tool_check = validate_tool_request("search_filings", query)
    if not tool_check.allowed:
        return tool_check.as_event()
    if _collections().count() == 0:
        return []
    results = _collections().query(query_texts=[query], n_results=K)
    return [
        {"text": doc, "source": meta["source"], "page": meta["page"]}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]
