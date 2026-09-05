from pathlib import Path

import pytest

from agent_core import prompt_loader
from agent_core.prompt_loader import PROMPT_ROOT, load_prompt, render_prompt


def test_prompt_root_is_central_asset_directory():
    assert PROMPT_ROOT == Path(__file__).resolve().parents[3] / "agent" / "prompts"


def test_loader_adds_markdown_suffix_and_reads_once(monkeypatch, tmp_path):
    asset = tmp_path / "example.md"
    asset.write_text("Example prompt\n", encoding="utf-8")
    monkeypatch.setattr(prompt_loader, "PROMPT_ROOT", tmp_path)
    reads = []
    read_text = Path.read_text

    def tracked_read(path, **kwargs):
        reads.append(path)
        return read_text(path, **kwargs)

    monkeypatch.setattr(Path, "read_text", tracked_read)
    assert load_prompt("example") == "Example prompt"
    assert load_prompt("example.md") == "Example prompt"
    assert reads == [asset]


def test_renderer_replaces_only_explicit_tokens():
    rendered = render_prompt(
        "runtime/actor",
        AGENT_STATE_DOC="STATE",
        ATOMIC_ACTIONS="ACTIONS",
    )
    assert "STATE" in rendered
    assert "ACTIONS" in rendered
    assert "{{AGENT_STATE_DOC}}" not in rendered

    point = render_prompt("vision/qwen_point", TARGET_NAME="Ritz Crackers")
    assert "Detect the Ritz Crackers" in point
    assert '{"box_2d": [ymin, xmin, ymax, xmax]}' in point


@pytest.mark.parametrize("name", ("../secret", "/tmp/secret"))
def test_loader_rejects_paths_outside_prompt_root(name):
    with pytest.raises(ValueError):
        load_prompt(name)


def test_renderer_rejects_missing_or_unused_values():
    with pytest.raises(ValueError, match="missing: ATOMIC_ACTIONS"):
        render_prompt("runtime/actor", AGENT_STATE_DOC="STATE")
    with pytest.raises(ValueError, match="unused: EXTRA"):
        render_prompt("runtime/episodic_associative", EXTRA="value")
