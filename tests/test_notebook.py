import ast
import json
from pathlib import Path


def test_colab_notebook_is_valid_json_and_has_no_unfinished_placeholders():
    path = Path(__file__).parents[1] / "notebooks" / "task_06_music_genre_classification.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) >= 20
    all_source = "\\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "TODO" not in all_source
    assert "build_feature_table" not in all_source


def test_notebook_code_cells_are_python_syntax():
    path = Path(__file__).parents[1] / "notebooks" / "task_06_music_genre_classification.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        # IPython magics are valid in Colab but are not Python AST statements.
        source = "\\n".join(line for line in source.splitlines() if not line.lstrip().startswith("%"))
        ast.parse(source, filename=f"notebook-cell-{index}")
