# CLAUDE.md — project rules for BeamPilot

BeamPilot is an agentic laser beam characterisation tool (ISO 11146 second-moment
beam widths, caustic fit, M²). It uses the Meta Model API: Muse Spark as the
agent and SAM 3.1 for beam segmentation.

## Hackathon rules
- This is a hackathon project (Oct 26 – Nov 4, 2026). All work must be new,
  written during the hackathon. Do not import or adapt pre-existing code.
- The repository is public and MIT licensed. Everything committed must be
  publishable.
- Meta models must be used: Muse Spark for the agent, SAM 3.1 for segmentation.
- Never use code, data or parameters from the maintainer's employers or
  institutes. Derive everything from public standards (ISO 11146) and textbook
  physics.

## Meta Model API
- OpenAI-compatible API, base URL `https://api.meta.ai/v1`.
- The API key is read from the environment variable `MODEL_API_KEY`, loaded
  from a local `.env` file via python-dotenv.
- Never commit secrets. `.env` is git-ignored; only `.env.example` with a
  placeholder is tracked. Never ask for API keys in chat and never print them.

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
- Explain each step to the maintainer in Persian (one or two sentences per
  step), keeping technical terms in English.
- All code, comments, commit messages and docs are in English.
