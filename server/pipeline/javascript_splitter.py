import logging
import re
from typing import Any, Dict, List, Tuple

from haystack import component
from haystack.dataclasses import Document

logger = logging.getLogger(__name__)


@component
class JavascriptSplitter:
    def __init__(self) -> None:
        logger.info("Entering JavascriptSplitter.__init__")

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
            raw_chunks = self.split(text=text, ext=".js", path=str(path))

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

            for idx, (chunk_text, _meta) in enumerate(chunks):
                split_id = f"{doc.id}:::{idx}"
                meta = dict(doc.meta or {})
                if idx > 0:
                    meta["previous"] = f"{doc.id}:::{idx-1}"
                if idx < len(chunks) - 1:
                    meta["next"] = f"{doc.id}:::{idx+1}"
                split_docs.append(Document(content=chunk_text, meta=meta, id=split_id))
        return {"documents": split_docs}

    def split(self, *, text: str, ext: str, path: str) -> List[Tuple[str, Dict[str, Any]]]:
        logger.info("Entering JavascriptSplitter.split path=%s", path)
        code = text or ""
        chunks: List[Tuple[str, Dict[str, Any]]] = []

        func_patterns = [
            r"function\s+([A-Za-z0-9_]+)\s*\([^\)]*\)\s*\{",
            r"(?:const|let|var)\s+([A-Za-z0-9_]+)\s*=\s*\([^\)]*\)\s*=>\s*\{",
            r"(?:const|let|var)\s+([A-Za-z0-9_]+)\s*=\s*function\s*\([^\)]*\)\s*\{",
        ]
        class_pattern = r"class\s+([A-Za-z0-9_]+)\s*(?:extends\s+[A-Za-z0-9_]+\s*)?\{"

        def extract_block(start_idx: int) -> str:
            depth = 0
            i = start_idx
            n = len(code)
            while i < n:
                ch = code[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return code[start_idx : i + 1]
                i += 1
            return code[start_idx:]

        for m in re.finditer(class_pattern, code):
            name = m.group(1)
            brace_start = code.find("{", m.end() - 1)
            if brace_start != -1:
                block = extract_block(brace_start)
                src = code[m.start() : brace_start] + block
                if src.strip():
                    chunks.append(
                        (src, {"language": "js", "path": path, "symbol_type": "class", "symbol_name": name})
                    )

        for pat in func_patterns:
            for m in re.finditer(pat, code):
                name = m.group(1)
                brace_start = code.find("{", m.end() - 1)
                if brace_start != -1:
                    block = extract_block(brace_start)
                    src = code[m.start() : brace_start] + block
                    if src.strip():
                        chunks.append(
                            (src, {"language": "js", "path": path, "symbol_type": "function", "symbol_name": name})
                        )

        if not chunks and code.strip():
            chunks.append((code, {"language": "js", "path": path, "symbol_type": "module", "symbol_name": ""}))
        return chunks