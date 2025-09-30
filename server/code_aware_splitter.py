import ast
import logging
import re
from typing import Dict, Any, List, Tuple
from haystack import component

logger = logging.getLogger(__name__)

@component
class CodeAwareSplitter:
    """
    Custom splitter for code as a Haystack Component.

    Inputs:
      - text: str (the full file content)
      - ext: str (file extension, e.g., ".py")
      - path: str (file path)

    Outputs:
      - documents: List[Document] where each chunk is a Document with meta fields:
        language, path, symbol_scope (module/class/function/rule/tag), symbol_name, ext, chunk_index
    """
    def __init__(self) -> None:
        super().__init__()
        logger.info("Entering CodeAwareSplitter.__init__")

    @component.output_types(documents=List["Document"])  # type: ignore[name-defined]
    def run(self, *, text: str, ext: str, path: str) -> Dict[str, Any]:
        logger.info("Entering CodeAwareSplitter.run ext=%s path=%s", ext, path)
        # Defer import to avoid hard dependency when Haystack is unavailable
        try:
            from haystack.dataclasses import Document  # type: ignore
        except Exception:
            # Minimal fallback structure compatible with our usage
            class Document:  # type: ignore
                def __init__(self, content: str, meta: Dict[str, Any], id: str | None = None) -> None:
                    self.content = content
                    self.meta = meta
                    self.id = id

        chunks_with_meta = self.split(text=text, ext=ext, path=path)
        docs: List[Document] = []
        for idx, (chunk_text, base_meta) in enumerate(chunks_with_meta):
            meta = dict(base_meta)
            meta["ext"] = ext.lower()
            meta["chunk_index"] = idx
            docs.append(Document(content=chunk_text, meta=meta, id=f"{path}:::{idx}"))
        return {"documents": docs}

    # Internal API to keep previous logic; returns List[(text, meta)]
    def split(self, *, text: str, ext: str, path: str) -> List[Tuple[str, Dict[str, Any]]]:
        logger.info("Entering CodeAwareSplitter.split ext=%s path=%s", ext, path)
        ext = ext.lower()
        if ext == ".py":
            return self._split_python(text=text, path=path)
        if ext == ".js":
            return self._split_javascript(text=text, path=path)
        if ext == ".html":
            return self._split_html(text=text, path=path)
        if ext == ".css":
            return self._split_css(text=text, path=path)
        return self._split_generic(text=text, path=path, language=ext.strip("."))

    def _split_python(self, *, text: str, path: str) -> List[Tuple[str, Dict[str, Any]]]:
        logger.info("Entering CodeAwareSplitter._split_python path=%s", path)
        chunks: List[Tuple[str, Dict[str, Any]]] = []
        try:
            tree = ast.parse(text)
        except Exception:
            return self._split_generic(text=text, path=path, language="py")
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
        if not chunks and text.strip():
            chunks.append((text, {"language": "py", "path": path, "symbol_scope": "module", "symbol_name": ""}))
        return chunks

    def _split_javascript(self, *, text: str, path: str) -> List[Tuple[str, Dict[str, Any]]]:
        logger.info("Entering CodeAwareSplitter._split_javascript path=%s", path)
        chunks: List[Tuple[str, Dict[str, Any]]] = []
        func_patterns = [
            r"function\s+([A-Za-z0-9_]+)\s*\([^\)]*\)\s*\{",
            r"const\s+([A-Za-z0-9_]+)\s*=\s*\([^\)]*\)\s*=>\s*\{",
            r"const\s+([A-Za-z0-9_]+)\s*=\s*function\s*\([^\)]*\)\s*\{",
        ]
        class_pattern = r"class\s+([A-Za-z0-9_]+)\s*(extends\s+[A-Za-z0-9_]+\s*)?\{"

        def extract_block(start_idx: int) -> str:
            depth = 0
            i = start_idx
            n = len(text)
            while i < n:
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start_idx:i + 1]
                i += 1
            return text[start_idx:]

        for m in re.finditer(class_pattern, text):
            name = m.group(1)
            brace_start = text.find("{", m.end() - 1)
            if brace_start != -1:
                block = extract_block(brace_start)
                src = text[m.start(): brace_start] + block
                if src.strip():
                    chunks.append((src, {"language": "js", "path": path, "symbol_scope": "class", "symbol_name": name}))

        for pat in func_patterns:
            for m in re.finditer(pat, text):
                name = m.group(1)
                brace_start = text.find("{", m.end() - 1)
                if brace_start != -1:
                    block = extract_block(brace_start)
                    src = text[m.start(): brace_start] + block
                    if src.strip():
                        chunks.append((src, {"language": "js", "path": path, "symbol_scope": "function", "symbol_name": name}))

        if not chunks and text.strip():
            chunks.append((text, {"language": "js", "path": path, "symbol_scope": "module", "symbol_name": ""}))
        return chunks

    def _split_html(self, *, text: str, path: str) -> List[Tuple[str, Dict[str, Any]]]:
        logger.info("Entering CodeAwareSplitter._split_html path=%s", path)
        chunks: List[Tuple[str, Dict[str, Any]]] = []
        sections = re.split(r"(?i)(?=</?(html|head|body|section|article|div|main|nav|footer)\b)", text)
        buf = ""
        for part in sections:
            buf += part
            if re.match(r"(?i)</(html|head|body|section|article|div|main|nav|footer)\b", part or ""):
                if buf.strip():
                    chunks.append((buf, {"language": "html", "path": path, "symbol_scope": "tag", "symbol_name": ""}))
                buf = ""
        if buf.strip():
            chunks.append((buf, {"language": "html", "path": path, "symbol_scope": "module", "symbol_name": ""}))
        return chunks

    def _split_css(self, *, text: str, path: str) -> List[Tuple[str, Dict[str, Any]]]:
        logger.info("Entering CodeAwareSplitter._split_css path=%s", path)
        chunks: List[Tuple[str, Dict[str, Any]]] = []
        rule_re = re.compile(r"([^{]+)\{([^}]*)\}", re.DOTALL)
        last_end = 0
        for m in rule_re.finditer(text):
            selector = m.group(1).strip()
            block = m.group(0)
            if block.strip():
                chunks.append((block, {"language": "css", "path": path, "symbol_scope": "rule", "symbol_name": selector}))
            last_end = m.end()
        tail = text[last_end:].strip()
        if tail:
            chunks.append((tail, {"language": "css", "path": path, "symbol_scope": "module", "symbol_name": ""}))
        return chunks

    def _split_generic(self, *, text: str, path: str, language: str) -> List[Tuple[str, Dict[str, Any]]]:
        logger.info("Entering CodeAwareSplitter._split_generic path=%s language=%s", path, language)
        size = 1000
        parts = [text[i:i + size] for i in range(0, len(text), size)] if text else []
        return [(p, {"language": language, "path": path, "symbol_scope": "module", "symbol_name": ""}) for p in parts]

    @staticmethod
    def _get_source_segment(code: str, node: ast.AST) -> str:
        try:
            return ast.get_source_segment(code, node) or ""
        except Exception:
            return ""
