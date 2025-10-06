import ast
import logging
from typing import Any, Dict, List, Tuple

from haystack import component
from haystack.dataclasses import Document

logger = logging.getLogger(__name__)


@component
class PythonSplitter:
    def __init__(self) -> None:
        logger.info("Entering PythonSplitter.__init__")

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
            raw_chunks: List[Tuple[str, Dict[str, Any]]] = self.split(text=text, path=str(path))

            # combine small chunks forward
            chunks: List[Tuple[str, Dict[str, Any]]] = []
            i = 0
            while i < len(raw_chunks):
                cur_text, cur_meta = raw_chunks[i]
                buf_text, buf_meta = cur_text, dict((cur_meta or {}))
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
                meta = (doc.meta or {}) | (_meta or {})
                if idx > 0:
                    meta["previous"] = f"{doc.id}:::{idx-1}"
                if idx < len(chunks) - 1:
                    meta["next"] = f"{doc.id}:::{idx+1}"
                split_docs.append(Document(content=chunk_text, meta=meta, id=split_id))
        return {"documents": split_docs}

    def split(self, *, text: str, path: str) -> List[Tuple[str, Dict[str, Any]]]:
        logger.info("Entering PythonSplitter.split path=%s", path)
        chunks: List[Tuple[str, Dict[str, Any]]] = []
        try:
            tree = ast.parse(text or "")
        except Exception as e:
            logger.warning("Failed to parse Python code in %s: %s", path, str(e))
            return self._split_generic(text=text or "", path=path, language="py")

        module_doc = ast.get_docstring(tree) or ""
        if module_doc.strip():
            chunks.append((module_doc, {"language": "py", "path": path, "symbol_scope": "module", "symbol_name": ""}))

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                src = self._get_source_segment(text, node)
                if src.strip():
                    chunks.append((src, {"language": "py", "path": path, "symbol_scope": "function", "symbol_name": node.name}))
            elif isinstance(node, ast.ClassDef):
                class_src = self._get_source_segment(text, node)
                if class_src.strip():
                    chunks.append((class_src, {"language": "py", "path": path, "symbol_scope": "class", "symbol_name": node.name}))
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        sub_src = self._get_source_segment(text, sub)
                        if sub_src.strip():
                            chunks.append((sub_src, {"language": "py", "path": path, "symbol_scope": "function", "symbol_name": f"{node.name}.{sub.name}"}))

        if not chunks and (text or "").strip():
            chunks.append((text, {"language": "py", "path": path, "symbol_scope": "module", "symbol_name": ""}))
        return chunks

    def _split_generic(self, *, text: str, path: str, language: str) -> List[Tuple[str, Dict[str, Any]]]:
        size = 1000
        parts = [text[i : i + size] for i in range(0, len(text), size)] if text else []
        return [(p, {"language": language, "path": path, "symbol_scope": "module", "symbol_name": ""}) for p in parts]

    @staticmethod
    def _get_source_segment(code: str, node: ast.AST) -> str:
        try:
            return ast.get_source_segment(code, node) or ""
        except Exception:
            return ""