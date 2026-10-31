# CLAUDE.md — project rules for BeamPilot

BeamPilot is an agentic laser beam characterisation tool (ISO 11146 second-moment
beam widths, caustic fit, M²). It uses the Meta Model API with Muse Spark as the
agent.

## Agent design
- `docs/agent-design.md` is the source of truth for the agent, tools, units and
  physics. Follow it. If code needs to differ from it, stop and ask before
  changing either.

## Hackathon rules
- This is a hackathon project (Oct 26 – Nov 4, 2026). All work must be new,
  written during the hackathon. Do not import or adapt pre-existing code.
- The repository is public and MIT licensed. Everything committed must be
  publishable.
- SAM 3.1 is not used. Beam widths use ISO 11146 second moments, which is the method the standard prescribes. Muse Spark (Meta Model API) is the Meta model in this project.
- Never use code, data or parameters from the maintainer's employers or
  institutes. Derive everything from public standards (ISO 11146) and textbook
  physics.

## Meta Model API
- OpenAI-compatible API, base URL `https://api.meta.ai/v1`.
- The API key is read from the environment variable `MODEL_API_KEY`, loaded
  from a local `.env` file via python-dotenv.
- Never commit secrets. `.env` is git-ignored; only `.env.example` with a
  placeholder is tracked. Never ask for API keys in chat and never print them.

## Model backends
- Meta Model API: base URL https://api.meta.ai/v1, key in env var MODEL_API_KEY. Main agent backend.
- Baseten: base URL https://inference.baseten.co/v1, model exactly "meta-models/muse-spark-open-1026" (Muse Spark Open, Meta open weights). Key in its own env var, listed in .env.example. Optional second backend for comparison.
- "Baseten" is a company name, not a typo of "baseline". Never remove or rename Baseten entries in .env.example or the code.

## change_lens switch
- `change_lens` (artificial waist, `docs/change-lens-design.md`) is offered only when
  `AgentConfig(change_lens=True)` and the scenario has a `LensKit`. The default is
  off; then tools, schemas and system prompt are byte-identical to the v1/v2 agent
  (a hash test guards this). Keep it that way: never let lens code change the
  switch-off path.
- Everything the agent sees about a lens is computed from the fit, never from the
  hidden beam (only the 8 mm clipping refusal uses the truth, as a yes/no).
- Report wording: "ISO 11146-1 procedure applied at an artificial waist (thin-lens
  model)", never "ISO compliant for the original beam".

## Data
- Use simulated beam images (with known ground truth) or openly licensed data
  only. Record the source and licence of any external data.

## Code standards
- src layout: package code in `src/beampilot/`, tests in `tests/`, helper
  scripts in `scripts/`.
- Python >= 3.10. Type hints on all public functions.
- NumPy-style docstrings. State physical units for every parameter and return
  value (e.g. "beam width in µm", "wavelength in m").
- Tests with pytest. Prefer physics-based known-answer tests, e.g. a simulated
  Gaussian beam with known waist w0 must give M² ≈ 1 within a stated tolerance.
- Lint and format with ruff (`ruff check .` and `ruff format .`); config is in
  `pyproject.toml`.
- Ask before adding new dependencies.
- Small, focused commits with imperative messages under 70 characters.
- Never force push. Ask before deleting files, restructuring, or rewriting
  Git history.

## Local workflow (Windows, PowerShell)
- Do not rely on `Activate.ps1`; call the venv interpreter directly:
  - `.venv\Scripts\python.exe -m pip install -e ".[dev]"`
  - `.venv\Scripts\python.exe -m ruff check .`
  - `.venv\Scripts\python.exe -m pytest`
- Do not change the PowerShell execution policy without asking.

## Communication
- Explain each step to the maintainer in English (one or two sentences per
  step).
- All code, comments, commit messages and docs are in English.
