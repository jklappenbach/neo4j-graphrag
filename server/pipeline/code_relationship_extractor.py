import ast
import logging
from typing import List

from haystack import component, Document

logger = logging.getLogger(__name__)

@component
class CodeRelationshipExtractor:
    """
    A component that parses Python files to extract imports and function calls,
    storing them in the document's metadata.
    """

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        for doc in documents:
            if not doc.meta.get("file_path") or not doc.meta["file_path"].endswith(".py"):
                continue

            try:
                # Use Python's Abstract Syntax Tree to parse code
                tree = ast.parse(doc.content)
                imports = set()
                calls = set()

                for node in ast.walk(tree):
                    # Find all import statements
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.add(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.add(node.module)
                    # Find all function call statements
                    elif isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Name):
                            calls.add(node.func.id)
                        elif isinstance(node.func, ast.Attribute):
                            calls.add(node.func.attr)

                doc.meta["imports"] = list(imports)
                doc.meta["calls"] = list(calls)

            except SyntaxError:
                # Ignore files with syntax errors
                print(f"Skipping {doc.meta['file_path']} due to syntax error.")

        return {"documents": documents}