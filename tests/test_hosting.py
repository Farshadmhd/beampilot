"""Hosted demo (Hugging Face Space): live-run guards, key safety, shipped files.

See docs/hosting-plan.md, sections 2-5.
"""

import csv
import getpass
import json
import re
import runpy
import subprocess
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from test_demo import _lens_script_client, _scripted_client

from beampilot import app as demo_app
from beampilot.agent import AgentConfig

REPO = Path(__file__).resolve().parents[1]
SENTINEL = "dummy-key-SENTINEL-7f3a9c"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("BEAMPILOT_LIVE_RUNS", "BEAMPILOT_SPEND_CAP_USD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(demo_app, "_GLOBAL", demo_app.GlobalBudget())


def _request(name: str) -> SimpleNamespace:
    demo_app._BROWSERS.pop(name, None)
    return SimpleNamespace(session_hash=name)


# Spend cap (BEAMPILOT_SPEND_CAP_USD)


def test_spend_cap_is_read_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("BEAMPILOT_SPEND_CAP_USD", "0.3")
    assert demo_app.spend_cap_from_env() == 0.3
    assert demo_app.GlobalBudget().spend_cap_usd == 0.3
    for bad in ("", "lots", "-1", "0", "inf", "nan"):
        monkeypatch.setenv("BEAMPILOT_SPEND_CAP_USD", bad)
        assert demo_app.spend_cap_from_env() == demo_app.DEFAULT_SPEND_CAP_USD == 5.0


def test_spend_cap_admits_a_run_only_if_a_worst_case_run_still_fits() -> None:
    budget = demo_app.GlobalBudget(spend_cap_usd=1.0)
    assert budget.reserve() is None  # 0 + 0.55 <= 1
    assert budget.reserved_usd == pytest.approx(0.55)
    budget.add_cost(0.40)
    budget.release()
    assert budget.reserved_usd == 0.0
    assert budget.reserve() is None  # 0.40 + 0.55 = 0.95 <= 1
    budget.add_cost(0.10)
    budget.release()
    # 0.50 + 0.55 = 1.05 > 1: the next run could overshoot the cap.
    assert budget.out_of_budget() == demo_app.BUDGET_TEXT
    assert budget.reserve() == demo_app.BUDGET_TEXT
    assert budget.runs == 2 and budget.active == 0


def test_a_run_that_never_started_frees_its_reserve() -> None:
    budget = demo_app.GlobalBudget(spend_cap_usd=0.6)
    assert budget.reserve() is None
    budget.release(started=False)
    assert budget.runs == 0 and budget.reserved_usd == 0.0
    assert budget.out_of_budget() is None


def test_live_runs_add_their_cost_and_stop_at_the_cap(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(demo_app, "_client", _scripted_client)
    monkeypatch.setattr(demo_app, "LIVE_RUNS_DIR", tmp_path)
    monkeypatch.setattr(demo_app, "_GLOBAL", demo_app.GlobalBudget(spend_cap_usd=0.6))
    list(demo_app.live_run("clean", 0, "", _request("spend-1")))
    (saved,) = tmp_path.glob("*.json")
    cost = json.loads(saved.read_text(encoding="utf-8"))["cost_usd"]
    assert cost > 0
    assert demo_app._GLOBAL.spent_usd == pytest.approx(cost, abs=1e-5)
    assert demo_app._GLOBAL.reserved_usd == 0.0
    demo_app._GLOBAL.add_cost(0.6 - 0.55 - demo_app._GLOBAL.spent_usd + 0.01)
    update = next(demo_app.live_run("clean", 0, "", _request("spend-2")))
    assert update[2] == demo_app.BUDGET_TEXT


# Live switch (BEAMPILOT_LIVE_RUNS) and the page banner


def test_live_switch_values(monkeypatch) -> None:
    for off in ("off", "OFF", " false ", "0", "no"):
        monkeypatch.setenv("BEAMPILOT_LIVE_RUNS", off)
        assert not demo_app.live_switch_on()
    for on in ("on", "", "yes", "1"):
        monkeypatch.setenv("BEAMPILOT_LIVE_RUNS", on)
        assert demo_app.live_switch_on()


def test_switched_off_live_runs_never_start(monkeypatch) -> None:
    def no_client():
        raise AssertionError("no client may be made when live runs are off")

    monkeypatch.setattr(demo_app, "_client", no_client)
    monkeypatch.setenv("BEAMPILOT_LIVE_RUNS", "off")
    request = _request("switch-off")
    update = next(demo_app.live_run("clean", 0, "", request))
    assert update[2] == demo_app.LIVE_OFF_TEXT
    assert demo_app._GLOBAL.runs == 0 and demo_app._browser(request).runs == 0


def test_banner_and_start_button_at_page_load(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    banner, enabled = demo_app.live_banner()
    assert demo_app.NO_KEY_TEXT in banner and not enabled
    assert "MODEL_API_KEY" in banner  # the variable's name, never a value

    monkeypatch.setenv("BEAMPILOT_LIVE_RUNS", "off")  # checked first
    banner, enabled = demo_app.live_banner()
    assert demo_app.LIVE_OFF_TEXT in banner and not enabled

    monkeypatch.delenv("BEAMPILOT_LIVE_RUNS")
    monkeypatch.setenv("MODEL_API_KEY", SENTINEL)
    assert demo_app.live_banner() == ("", True)
    assert SENTINEL not in json.dumps(demo_app.live_banner())

    monkeypatch.setenv("BEAMPILOT_SPEND_CAP_USD", "0.3")  # below one worst-case run
    monkeypatch.setattr(demo_app, "_GLOBAL", demo_app.GlobalBudget())
    banner, enabled = demo_app.live_banner()
    assert demo_app.BUDGET_TEXT in banner and not enabled


def test_live_runs_have_a_queue_of_one_and_replays_do_not_wait() -> None:
    pytest.importorskip("gradio")
    app = demo_app.build_app()
    fns = {getattr(f.fn, "__name__", ""): f for f in app.fns.values()}
    assert demo_app.MAX_CONCURRENT_RUNS == 1
    assert fns["start_run"].concurrency_limit == 1
    assert fns["start_run"].concurrency_id == "live"
    assert fns["replay"].concurrency_id != "live"
    assert fns["replay"].concurrency_limit == "default"  # REPLAY_CONCURRENCY


# Live lens runs use the v3 limits; normal runs keep theirs.


def _saved_config(tmp_path: Path) -> dict:
    (saved,) = tmp_path.glob("*.json")
    return json.loads(saved.read_text(encoding="utf-8"))["config"]


def test_live_lens_runs_use_the_v3_limits(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(demo_app, "_client", _lens_script_client)
    monkeypatch.setattr(demo_app, "LIVE_RUNS_DIR", tmp_path)
    request = _request("lens-limits")
    list(demo_app.live_run("waist_outside_range", 0, "", request, lens_kit=True))
    config = _saved_config(tmp_path)
    assert config["max_model_calls"] == 80
    assert config["max_total_tokens"] == 3_000_000
    assert config["max_cost_usd"] == 0.50
    assert config["change_lens"] is True
    assert demo_app._browser(request).session.scenario.max_positions == 28


def test_normal_live_runs_keep_their_limits(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(demo_app, "_client", _scripted_client)
    monkeypatch.setattr(demo_app, "LIVE_RUNS_DIR", tmp_path)
    list(demo_app.live_run("clean", 0, "", _request("normal-limits")))
    config, default = _saved_config(tmp_path), AgentConfig()
    assert config["max_model_calls"] == default.max_model_calls
    assert config["max_total_tokens"] == default.max_total_tokens
    assert config["max_cost_usd"] == default.max_cost_usd
    assert not config.get("change_lens")


# The key never reaches the UI, a run file or the console.


def test_dummy_key_never_reaches_ui_run_file_or_console(
    monkeypatch, tmp_path, capfd
) -> None:
    """The real client with a fake server that echoes the key in its error."""
    import openai

    seen = []

    def server(request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("authorization", "")
        seen.append(auth)
        message = f"Incorrect API key provided: {auth}"
        return httpx.Response(401, json={"error": {"message": message}})

    real = openai.OpenAI

    def offline_openai(*args, **kwargs):
        transport = httpx.MockTransport(server)
        return real(*args, http_client=httpx.Client(transport=transport), **kwargs)

    monkeypatch.setattr(openai, "OpenAI", offline_openai)
    monkeypatch.setattr(demo_app, "_CLIENT", None)  # build a fresh client
    monkeypatch.setattr(demo_app, "LIVE_RUNS_DIR", tmp_path)
    monkeypatch.setenv("MODEL_API_KEY", SENTINEL)
    updates = list(demo_app.live_run("clean", 0, "", _request("sentinel")))

    assert seen and SENTINEL in seen[0]  # the key really went to the server
    shown = json.dumps(updates, default=str)
    assert SENTINEL not in shown
    assert "AuthenticationError (HTTP 401)" in shown
    files = list(tmp_path.glob("*.json"))
    assert files
    for path in files:
        assert SENTINEL not in path.read_text(encoding="utf-8")
    out, err = capfd.readouterr()
    assert SENTINEL not in out and SENTINEL not in err


# Shipped replay files


def _personal_strings() -> set[str]:
    """Local user name, home folder and git e-mail of whoever runs the tests."""
    values = {getpass.getuser(), Path.home().name}
    try:
        email = subprocess.run(
            ["git", "config", "user.email"],
            capture_output=True,
            text=True,
            cwd=REPO,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        email = ""
    if "@" in email:
        values |= {email, email.split("@", 1)[1]}
    generic = {"runner", "root", "user", "admin", "home"}
    return {v.lower() for v in values if len(v) >= 4 and v.lower() not in generic}


FORBIDDEN = {
    "Windows path": re.compile(r"[A-Za-z]:(\\\\|/)"),
    "home path": re.compile(r"/(home|Users)/\w"),
    "e-mail address": re.compile(r"[\w.+-]+@[\w-]+\.[A-Za-z]{2,}"),
    "key-like string": re.compile(r"\bsk-[A-Za-z0-9_-]{8,}|Bearer\s", re.IGNORECASE),
    "key name": re.compile(r"api[_-]?key|MODEL_API_KEY|BASETEN", re.IGNORECASE),
}


def test_shipped_run_files_hold_no_paths_names_or_keys() -> None:
    personal = _personal_strings()
    runs = demo_app.demo_runs()
    assert len(runs) == 5
    for path in runs:
        text = path.read_text(encoding="utf-8")
        for kind, pattern in FORBIDDEN.items():
            match = pattern.search(text)
            assert match is None, f"{path.name}: {kind} {match.group(0)!r}"
        lowered = text.lower()
        assert not [v for v in personal if v in lowered], path.name


def test_every_replay_has_a_recording_date() -> None:
    with demo_app.RECORDED_MANIFEST.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["file"] for row in rows} == {p.name for p in demo_app.demo_runs()}
    for row in rows:
        assert date.fromisoformat(row["recorded"]) >= date(2026, 10, 26)
        assert row["source"]


def test_replays_are_labelled_as_recorded(tmp_path) -> None:
    for label, value in demo_app.replay_choices():
        assert re.match(r"Recorded run, 2026-\d\d-\d\d, muse-spark-1\.2: ", label)
        assert (REPO / value).exists()
    shipped = (demo_app.DEMO_RUNS_DIR / "lens_v3_L5.json").relative_to(REPO)
    title = list(demo_app.replay(shipped.as_posix(), None, 0.0))[-1][0]
    assert "Recorded run, 2026-10-31, muse-spark-1.2" in title
    assert "a replay, not a live run" in title and "L5 (seed 904)" in title
    upload = tmp_path / "lens_v3_L5.json"  # same name, but uploaded
    upload.write_bytes((REPO / shipped).read_bytes())
    title = list(demo_app.replay(None, str(upload), 0.0))[-1][0]
    assert "Recorded run, date unknown, muse-spark-1.2" in title


# Space files: requirements, front matter, entry point


def _requirement_name(spec: str) -> str:
    name = re.split(r"[<>=!~;\[ ]", spec.strip(), maxsplit=1)[0]
    return name.lower().replace("_", "-")


def test_requirements_pin_exactly_the_project_dependencies() -> None:
    tomllib = pytest.importorskip("tomllib")
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    wanted = {_requirement_name(d) for d in project["project"]["dependencies"]}
    lines = (REPO / "requirements.txt").read_text(encoding="utf-8").splitlines()
    specs = [line.strip() for line in lines if line.strip() and line[0] != "#"]
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+==[0-9][\w.]*", s) for s in specs)
    assert {_requirement_name(s) for s in specs} == wanted
    assert "gradio" not in wanted  # pinned by sdk_version instead


def _front_matter(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    block = text[4 : text.index("\n---\n", 4)]
    pairs = (line.split(":", 1) for line in block.splitlines() if line.strip())
    return {key.strip(): value.strip().strip('"') for key, value in pairs}


def test_space_readme_front_matter_matches_the_project() -> None:
    meta = _front_matter(REPO / "space" / "README.md")
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    gradio_pin = re.search(r'"gradio==([\d.]+)"', pyproject).group(1)
    assert meta["sdk"] == "gradio"
    assert meta["sdk_version"] == gradio_pin
    assert meta["python_version"] == "3.12"
    assert (REPO / meta["app_file"]).is_file()
    assert meta["license"] == "mit"
    assert meta["suggested_hardware"] == "cpu-basic"
    assert 0 < len(meta["short_description"]) <= 60
    text = (REPO / "space" / "README.md").read_text(encoding="utf-8")
    assert "simulated" in text and "https://github.com/Farshadmhd/beampilot" in text
    # The GitHub README carries no front matter.
    assert not (REPO / "README.md").read_text(encoding="utf-8").startswith("---")


def test_space_entry_point_runs_the_app_from_the_src_layout(monkeypatch) -> None:
    monkeypatch.setattr(sys, "path", list(sys.path))
    namespace = runpy.run_path(str(REPO / "scripts" / "space_app.py"), run_name="x")
    assert namespace["main"] is demo_app.main
    assert str(REPO / "src") == sys.path[0]
    assert demo_app.DEMO_RUNS_DIR == REPO / "docs" / "runs"
