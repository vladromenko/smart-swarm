import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_owned_python_style():
    paths = (path for folder in ("src", "scripts", "tests")
             for path in (ROOT / folder).rglob("*.py"))
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            assert not isinstance(node, (ast.Break, ast.Continue)), (path, node.lineno)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.end_lineno - node.lineno + 1 <= 40, (path, node.name)
