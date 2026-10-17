from pathlib import Path


def test_readme_explains_the_demo_and_shows_the_graph() -> None:
    text = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    lowered = text.lower()
    assert "```mermaid" in text
    assert "portfolio demo" in lowered
    assert "not a production" in lowered
    assert "does not call the public web" in lowered
    assert "stub" in lowered
    assert "Planner" in text
    assert "Human approval" in text
    assert "benchmark" in lowered
