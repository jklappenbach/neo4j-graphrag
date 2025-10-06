import logging
import re
from typing import Any, Dict, List, Tuple

from haystack import component
from haystack.dataclasses import Document

logger = logging.getLogger(__name__)

@component
class CssSplitter:
    def __init__(self) -> None:
        logger.info("Entering CssSplitter.__init__")

    @component.output_types(documents=list[Document])
    def run(self, documents: List[Document], *, min_chunk_size: int = 500) -> Dict[str, Any]:
        """
        Accepts Documents and returns split Documents, compatible with DocumentSplitter output.
        If a produced chunk is smaller than min_chunk_size, it will be combined with the next chunk.
        """
        min_len = max(500, int(min_chunk_size or 0))
        split_docs: List[Document] = []
        for doc in documents or []:
            text = getattr(doc, "content", "") or ""
            path = (getattr(doc, "meta", {}) or {}).get("file_path") or (getattr(doc, "id", None) or "")
            raw_chunks = self.split(text=text, ext=".css", path=str(path))

            # combine small chunks forward
            chunks: List[Tuple[str, Dict[str, Any]]] = []
            i = 0
            while i < len(raw_chunks):
                cur_text, cur_meta = raw_chunks[i]
                buf_text, buf_meta = cur_text, dict(cur_meta or {})
                while len(buf_text) < min_len and i + 1 < len(raw_chunks):
                    nxt_text, nxt_meta = raw_chunks[i + 1]
                    buf_text = f"{buf_text}\n{nxt_text}"
                    for k, v in (nxt_meta or {}).items():
                        buf_meta.setdefault(k, v)
                    i += 1
                chunks.append((buf_text, buf_meta))
                i += 1

            # Map to Document objects so downstream _create_docs_from_splits-style consumers can ingest them
            for idx, (chunk_text, _meta) in enumerate(chunks):
                split_id = f"{doc.id}:::{idx}"
                meta = dict(doc.meta or {})
                if idx > 0:
                    meta["previous"] = f"{doc.id}:::{idx-1}"
                if idx < len(chunks) - 1:
                    meta["next"] = f"{doc.id}:::{idx+1}"
                split_docs.append(Document(content=chunk_text, meta=meta, id=split_id))
        return {"documents": split_docs}

    @staticmethod
    def split(*, text: str, ext: str, path: str) -> List[Tuple[str, Dict[str, Any]]]:
        logger.info("Entering CssSplitter.split path=%s", path)
        css = text or ""
        chunks: List[Tuple[str, Dict[str, Any]]] = []
        rule_re = re.compile(r"([^{]+)\{([^}]*)\}", re.DOTALL)
        last_end = 0
        for m in rule_re.finditer(css):
            selector = m.group(1).strip()
            block = m.group(0)
            if block.strip():
                chunks.append((block, {"language": "css", "path": path, "symbol_scope": "rule", "symbol_name": selector}))
            last_end = m.end()
        tail = css[last_end:].strip()
        if tail:
            chunks.append((tail, {"language": "css", "path": path, "symbol_scope": "module", "symbol_name": ""}))
        return chunks