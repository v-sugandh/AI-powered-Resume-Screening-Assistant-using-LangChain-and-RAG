import os
import tempfile
from typing import Dict, List

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Queries that pull the different parts of a resume, in addition to the JD itself.
SECTION_QUERIES = [
    "technical skills tools programming languages frameworks",
    "work experience roles responsibilities achievements",
    "education degrees certifications courses",
    "projects publications research",
]


def _unique_label(stem: str, taken: set) -> str:
    label, n = stem, 2
    while label in taken:
        label = f"{stem} ({n})"
        n += 1
    return label


class ResumeIndex:
    """Holds one FAISS vector store (and retriever) per uploaded resume."""

    def __init__(self, embeddings, chunk_size: int = 800, chunk_overlap: int = 150):
        self.embeddings = embeddings
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
        )
        self.stores: Dict[str, FAISS] = {}

    @property
    def candidates(self) -> List[str]:
        return list(self.stores.keys())

    # ---------- ingestion ----------
    def add_resume(self, filename: str, file_bytes: bytes) -> str:
        stem = os.path.splitext(filename)[0]
        label = _unique_label(stem, set(self.stores))

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            path = tmp.name
        try:
            pages = PyPDFLoader(path).load()  # PDF Document Loader
        finally:
            os.remove(path)

        chunks = self.splitter.split_documents(pages)  # Text Splitter
        chunks = [c for c in chunks if c.page_content.strip()]
        if not chunks:
            raise ValueError(
                f"No extractable text in '{filename}'. It may be a scanned/image-only PDF."
            )
        for c in chunks:
            c.metadata["candidate"] = label
            c.metadata["source"] = filename

        self.stores[label] = FAISS.from_documents(chunks, self.embeddings)  # Embeddings + Vector DB
        return label

    # ---------- retrieval ----------
    def retriever(self, candidate: str, k: int = 5):
        return self.stores[candidate].as_retriever(search_kwargs={"k": k})  # Retriever

    def retrieve_context(self, candidate: str, query: str, k: int = 5) -> str:
        """Retrieve chunks for the query + resume-section queries, de-duplicate, keep reading order."""
        retriever = self.retriever(candidate, k=k)
        seen: Dict[str, Document] = {}
        for q in [query] + SECTION_QUERIES:
            for doc in retriever.invoke(q):
                seen.setdefault(doc.page_content, doc)
        ordered = sorted(
            seen.values(),
            key=lambda d: (d.metadata.get("page", 0), d.metadata.get("start_index", 0)),
        )
        return "\n---\n".join(d.page_content for d in ordered)
