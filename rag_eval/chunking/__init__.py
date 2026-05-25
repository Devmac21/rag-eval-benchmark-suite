"""Chunking strategies for corpus preprocessing."""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod

from rag_eval.schemas import DocumentChunk


def stable_chunk_id(source_doc_id: str, index: int, text: str) -> str:
    h = hashlib.sha256(f"{source_doc_id}:{index}:{text[:200]}".encode()).hexdigest()[:16]
    return f"{source_doc_id}_{index}_{h}"


class ChunkingStrategy(ABC):
    name: str

    @abstractmethod
    def chunk(
        self,
        doc_id: str,
        text: str,
        *,
        source_document_id: str | None = None,
    ) -> list[DocumentChunk]:
        raise NotImplementedError


class FixedWindowChunking(ChunkingStrategy):
    """Fixed character windows with optional overlap."""

    name = "fixed_window"

    def __init__(self, size: int = 512, overlap: int = 64):
        self.size = max(32, size)
        self.overlap = max(0, min(overlap, self.size - 1))

    def chunk(
        self,
        doc_id: str,
        text: str,
        *,
        source_document_id: str | None = None,
    ) -> list[DocumentChunk]:
        text = text.strip()
        if not text:
            return []
        sid = source_document_id or doc_id
        chunks: list[DocumentChunk] = []
        step = self.size - self.overlap
        i = 0
        idx = 0
        while i < len(text):
            piece = text[i : i + self.size]
            cid = stable_chunk_id(doc_id, idx, piece)
            chunks.append(
                DocumentChunk(id=cid, text=piece, metadata={"strategy": self.name, "source_doc_id": sid}),
            )
            idx += 1
            i += step
        return chunks


class SentenceChunking(ChunkingStrategy):
    """Greedy packing of sentences up to max_chars."""

    name = "sentence"

    _sent_split = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, max_chars: int = 600):
        self.max_chars = max_chars

    def chunk(
        self,
        doc_id: str,
        text: str,
        *,
        source_document_id: str | None = None,
    ) -> list[DocumentChunk]:
        text = text.strip()
        if not text:
            return []
        sid = source_document_id or doc_id
        sents = self._sent_split.split(text)
        chunks: list[DocumentChunk] = []
        buf: list[str] = []
        buf_len = 0
        idx = 0

        def flush():
            nonlocal buf, buf_len, idx
            if not buf:
                return
            piece = " ".join(buf).strip()
            cid = stable_chunk_id(doc_id, idx, piece)
            chunks.append(
                DocumentChunk(id=cid, text=piece, metadata={"strategy": self.name, "source_doc_id": sid}),
            )
            idx += 1
            buf = []
            buf_len = 0

        for s in sents:
            sep = 1 if buf else 0
            if buf_len + sep + len(s) > self.max_chars and buf:
                flush()
                sep = 0
            buf.append(s)
            buf_len += sep + len(s)
        flush()
        return chunks


class RecursiveChunking(ChunkingStrategy):
    """Split on paragraphs then sentences until under max_chars."""

    name = "recursive"

    def __init__(self, max_chars: int = 700):
        self.max_chars = max_chars

    def chunk(
        self,
        doc_id: str,
        text: str,
        *,
        source_document_id: str | None = None,
    ) -> list[DocumentChunk]:
        text = text.strip()
        if not text:
            return []
        sid = source_document_id or doc_id
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        sentence = SentenceChunking(max_chars=self.max_chars)
        chunks: list[DocumentChunk] = []
        idx_base = 0
        for pi, para in enumerate(paragraphs):
            if len(para) <= self.max_chars:
                cid = stable_chunk_id(doc_id, idx_base, para)
                chunks.append(
                    DocumentChunk(
                        id=cid,
                        text=para,
                        metadata={"strategy": self.name, "paragraph": pi, "source_doc_id": sid},
                    )
                )
                idx_base += 1
            else:
                for sub in sentence.chunk(f"{doc_id}_p{pi}", para, source_document_id=sid):
                    sub.metadata["strategy"] = self.name
                    sub.metadata["paragraph"] = pi
                    sub.metadata.setdefault("source_doc_id", sid)
                    chunks.append(sub)
        return chunks


STRATEGIES: dict[str, type[ChunkingStrategy]] = {
    FixedWindowChunking.name: FixedWindowChunking,
    SentenceChunking.name: SentenceChunking,
    RecursiveChunking.name: RecursiveChunking,
}


def get_strategy(name: str, **kwargs) -> ChunkingStrategy:
    cls = STRATEGIES.get(name)
    if not cls:
        raise ValueError(f"Unknown chunking strategy: {name}. Choose from {list(STRATEGIES)}")
    return cls(**kwargs)
