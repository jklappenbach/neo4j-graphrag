# This is a sample Python script.
import ast
from typing import List, Tuple


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def main(name):
    # Read a py file and see what we get
    with open("server/graph_rag_manager.py", "r") as f:
        code = f.read()
        imports, calls, functions, classes, nodes_with_source = _py_extract_symbols_and_calls(code)
        # print(f"Imports: {imports}")
        # print(f"Calls: {calls}")
        # print(f"Functions: {functions}")
        # print(f"Classes: {classes}")
        # print("Nodes with source:")
        for kind, name, src in nodes_with_source:
            print(f"- {kind} {name}:\n{src}\n")

def _py_extract_symbols_and_calls(code: str) -> Tuple[List[str], List[str], List[str], List[str], List[Tuple[str, str, str]]]:
    """Extract imported module names, called function names, defined functions, defined classes,
    and the source code for each parsed node of interest (imports, calls, functions, classes)."""
    imports: List[str] = []
    calls: List[str] = []
    functions: List[str] = []
    classes: List[str] = []
    nodes_with_source: List[Tuple[str, str, str]] = []

    def _node_source(n: ast.AST) -> str:
        try:
            return ast.get_source_segment(code, n) or ""
        except Exception:
            return ""

    try:
        tree = ast.parse(code)
    except Exception:
        return imports, calls, functions, classes, nodes_with_source

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                if n.name:
                    mod = n.name.split(".")[0]
                    imports.append(mod)
                    nodes_with_source.append(("import", mod, _node_source(node)))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mod = node.module.split(".")[0]
                imports.append(mod)
                nodes_with_source.append(("import_from", mod, _node_source(node)))
        elif isinstance(node, ast.FunctionDef):
            functions.append(node.name)
            nodes_with_source.append(("function", node.name, _node_source(node)))
        elif isinstance(node, ast.AsyncFunctionDef):
            functions.append(node.name)
            nodes_with_source.append(("async_function", node.name, _node_source(node)))
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
            nodes_with_source.append(("class", node.name, _node_source(node)))
        elif isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            if name:
                calls.append(name)
                nodes_with_source.append(("call", name, _node_source(node)))

    # Deduplicate while preserving order
    def _dedup(seq: List[str]) -> List[str]:
        return list(dict.fromkeys(seq))

    return _dedup(imports), _dedup(calls), _dedup(functions), _dedup(classes), nodes_with_source

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
