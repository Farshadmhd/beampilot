"""The evaluation summary and the hero figure are generated, not hand-made."""

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_committed_summary_matches_the_result_files() -> None:
    summary = _script("make_evaluation_summary")
    data = {v: summary.load(summary.RESULT_FILES[v]) for v in summary.VERSIONS}
    expected = summary.summary(data)
    committed = summary.OUT.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert committed == expected, "run scripts/make_evaluation_summary.py"
    assert len(data["v1"]) == len(data["v2"]) == 83


def test_hero_figure_builds_from_the_saved_run() -> None:
    hero = _script("make_hero_figure")
    fig = hero.make_figure(hero.RUN, hero.FRAME_ID)
    caustic, frame = fig.axes
    assert "iso_compliant" in caustic.get_title()
    assert "re-simulated" in frame.get_title()
    assert frame.get_images()[0].get_array().shape == (512, 512)
