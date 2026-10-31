# Hosting plan: BeamPilot demo on Hugging Face Spaces

Status: **approved 31 Oct 2026**, built on branch `add-hosting`. Decisions taken
by the maintainer:

1. **Plan:** HF PRO, subscribed by the maintainer and kept until after the
   community-vote result on 16 Nov 2026.
2. **Lens replays:** ship **both**: the v3 L5 run and the seed-920 smoke run, so
   five replay files in total.
3. **Delivery:** upload with the `hf` CLI (section 6).

Goal: a public URL where anyone can **replay** saved agent runs (no key needed)
and, while the budget lasts, start **live runs** of the Muse Spark agent.

## 1. Platform: what the Hugging Face docs say (checked 31 Oct 2026)

| Topic | What the docs say | Source |
|---|---|---|
| Free CPU tier | CPU Basic (2 vCPU, 16 GB) has no hourly cost, **but** Gradio and Docker Spaces "require a paid plan to create" (PRO for personal accounts). Only static Spaces are free for everyone. | [Spaces overview](https://huggingface.co/docs/hub/spaces-overview#hardware-resources), [GPU Spaces, CPU table note](https://huggingface.co/docs/hub/spaces-gpus) |
| Free-account exception | Free accounts in good standing (verified email, account older than 30 days) can host up to 2 Gradio Spaces on **ZeroGPU**. | [ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu) |
| PRO price | $9 per month; the list includes "Host ZeroGPU, Gradio & Docker Spaces". | [Pricing](https://huggingface.co/pricing) |
| Sleep | A `cpu-basic` Space sleeps when inactive "currently, 48 hours"; the next visitor restarts it automatically. You can't change the sleep time on free hardware. | [GPU Spaces, sleep time](https://huggingface.co/docs/hub/spaces-gpus#sleep-time) |
| Secrets | Secrets are private. Their value can't be read back from the settings page, and they are not copied when someone duplicates the Space. The app sees them as environment variables (`os.getenv`). | [Spaces overview, secrets](https://huggingface.co/docs/hub/spaces-overview#managing-secrets) |
| Variables | Variables are public and are copied to duplicates. Use them for non-secret settings only. | same |
| Front matter | Keys in a YAML block at the top of `README.md`: `title`, `emoji`, `colorFrom`, `colorTo`, `sdk`, `sdk_version`, `python_version`, `app_file`, `short_description`, `license`, `pinned`, `suggested_hardware`. | [Config reference](https://huggingface.co/docs/hub/spaces-config-reference) |
| Dependencies | `requirements.txt` is installed by the runtime; `sdk_version` sets the Gradio version. | [Gradio Spaces](https://huggingface.co/docs/hub/spaces-sdks-gradio) |
| Rebuild | Every new commit rebuilds and restarts the Space. | [Spaces overview](https://huggingface.co/docs/hub/spaces-overview) |
| Network | Outbound requests may use ports 80, 443 and 8080 only. The Meta Model API is HTTPS on port 443. | same |
| Git auth and large files | Push access needs a login (`hf auth login`, or a token in the git credential prompt). Binary files go through Git Xet. | [Repositories: getting started](https://huggingface.co/docs/hub/repositories-getting-started), [Xet](https://huggingface.co/docs/hub/xet/using-xet-storage) |

**Verdict: the free CPU tier does not fit as asked.** A free account can no
longer create a Gradio Space on CPU Basic.

**Alternative (recommended): HF PRO for one month ($9).** Everything else in this
plan stays the same: Gradio SDK, `cpu-basic`, and the same secrets. Cancel after
the judging period.

Not recommended: ZeroGPU on a free account. ZeroGPU is built for functions that
use the `@spaces.GPU` decorator, and BeamPilot runs no GPU code. It would be an
untested misuse of GPU hardware, and the 30-day account-age rule may not be met.

**Decision 1 (taken):** PRO.

## 2. Replay without a key

Replay reads JSON files from `docs/runs/` and calls no API. It works with no
secret set. Five files ship; three were already tracked:

| File | What it shows | Status |
|---|---|---|
| `docs/runs/mystery_s7.json` | Mystery beam (632.8 nm, M² ≈ 1.43) with a built-in **laser power ×3** fault: the agent met 6 saturated frames, lowered the exposure, and reached `iso_compliant` in 18 positions ($0.09). | tracked |
| `docs/runs/clean_room_light_live.json` | Clean beam with a **room-light fault** added by the operator before measurement 6: +3 counts of background ($0.07). | tracked |
| `docs/runs/clean_operator_note.json` | Operator note ("stage blocked 60–80 mm") respected ($0.05). | tracked |
| `docs/runs/lens_v3_L5.json` (byte-identical copy of `runs/evaluation-v3/L5_agent.json`) | **Lens run** from the pre-registered v3 evaluation: the waist lies outside the stage; the agent inserts +750 mm at −20 mm and reaches `iso_compliant` with the back-calculated original beam ($0.12). | added |
| `docs/runs/lens_smoke_s920.json` (byte-identical copy of `runs/20261031-183949_waist_outside_range_s920_muse-spark-1.2.json`) | **Lens run**, the change_lens smoke run before v3: +25 mm at −20 mm, `iso_compliant` ($0.13). | added |

**Decision 2 (taken):** ship both lens runs.

**Labels.** Every replay is labelled on the page as "Recorded run, <date>,
muse-spark-1.2", both in the run list and above the replay, so that a visitor
never takes a replay for a live run. The model name comes from the file
(`config.model`).

The run files themselves hold no date. So the date comes from a small manifest,
`docs/runs/recorded.csv` (file, date, source of the date):

- the two lens runs: 31 Oct 2026, from the original file timestamps;
- the three older runs: 27 Oct 2026, the date of commit becabe2, which added
  them. Their original timestamped files are no longer kept.

A test checks that every replay file has a manifest entry. An uploaded file is
labelled "Recorded run, date unknown, <model>".

Read-only checks done on all five candidates on 31 Oct:

- Searched for `C:\`, `Users`, `farsh`, the maintainer's e-mail domain, `MODEL_API_KEY`, `BASETEN`, `sk-`, `api_key`, `Bearer`, `.env`: **0 matches in every file**.
- Searched for path-like strings (`X:\…`, `/home/…`, `/Users/…`): **none**.
- Top-level keys are only scenario/case, seed, positions, truth, scenario
  parameters, fault events, lens changes, tool log, config (model name and limits),
  operator note, stop reason, usage, cost, report and steps.
- `load_run` and `replay_events` load L5 (68 events), seed 920 (88) and
  mystery_s7 (59) without errors.

A new test (section 5) repeats this scan for every `docs/runs/*.json` file, so a
file added later can't leak a path or key.

## 3. Live runs: only with a secret, and capped

A live run is possible only when the Space secret `MODEL_API_KEY` is set **and**
the live switch is on. The protections:

| Protection | Today in `app.py` | Planned |
|---|---|---|
| Queue concurrency | `queue(default_concurrency_limit=2)`, `MAX_CONCURRENT_RUNS = 2` | **1 live run at a time.** Set `concurrency_limit=1, concurrency_id="live"` on the Start event only, and `MAX_CONCURRENT_RUNS = 1`. Replay events keep the default concurrency, so replay never waits behind a 15-minute live run. |
| Per-run caps | `AgentConfig`: $0.50, 1.5 M tokens and 60 model calls (lens: 80 calls, 28 positions); 15-minute timeout | Normal runs unchanged. **Live lens runs use the v3 limits:** 80 model calls, 3 M tokens, $0.50 and 28 positions. |
| Per-tab caps | 3 runs and 2 M tokens per tab (convenience only) | 3 runs; **3 M tokens** per tab, so that one lens run can use its full v3 token limit. |
| Process caps | 50 runs; `BEAMPILOT_TOKEN_CAP` (default 60 M tokens, about $20) | unchanged, plus the spend cap below |
| **Spend cap per restart** | none | New env var `BEAMPILOT_SPEND_CAP_USD`, **default 5**. A thread-safe running total adds `run.cost_usd` after each run. A run starts only if spent + $0.55 (the per-run cap plus one call of overshoot) ≤ cap. So the cap cannot be exceeded, even by the last run. |
| **Live switch** | none | New env var `BEAMPILOT_LIVE_RUNS` (`on`/`off`, default `on` so local use is unchanged). On the Space it is a public **Variable**. Setting it to `off` turns live runs off without a code push. |
| **Clear messages** | a "needs MODEL_API_KEY" text, shown only after Start is clicked | A banner when the page loads, and the Start button disabled, in three cases (checked in this order): **switched off** ("Live runs are switched off on this demo right now. Replay works."), **no key** ("Live runs are not set up on this demo (no MODEL_API_KEY). Replay works without one."), **budget used** ("This demo has used its live-run budget for now; use Replay."). The server-side check stays the real guard; the banner is only information. |

Caveat: the $5 budget counter lives in process memory. It resets on every restart:
a push, a settings change, or waking from sleep after 48 h. Visitors can't force a
restart, but the cap is per restart, not lifetime. A typical live run costs
$0.05–0.13, so $5 is roughly 40–100 runs. If the Meta Model API console offers a
spending limit on the key, set it as a lifetime backstop (I have not checked
whether it does).

Scope: all changes are in `src/beampilot/app.py` (the UI) and new files. The
agent, prompt and tools stay frozen. With `change_lens` off, nothing changes, so
the hash test still holds.

## 4. The key is never logged, printed or saved

- **Where the key lives:** only in the Space secret `MODEL_API_KEY`. `make_client()` reads it from the environment; there is no `.env` on the Space. The UI never asks for a key.
- **What can leak today:** nothing found. API errors are already shown by type and HTTP status only, never by message text. `app.py` and `agent.py` contain no `print` or `logging` calls. Run files store the model name and limits, not the client.
- **New test:** set `MODEL_API_KEY` to a sentinel string and run a live run against a fake client. Then assert the sentinel appears in none of: the saved run JSON, the UI outputs (log, report, messages), and the captured stdout and stderr.
- **Hard rules:**
  - no Gradio `debug=True`;
  - no `show_error=True`;
  - no `OPENAI_LOG` variable on the Space;
  - live run files stay on the Space's temporary disk (`runs/`) and are not offered for download.
- **Baseten:** the Baseten key is **not** set on the Space. The app uses only the Meta Model API.

## 5. Files needed

1. **Space README `space/README.md`.** The GitHub `README.md` gets **no** front
   matter. The deploy step copies `space/README.md` over `README.md` in the
   exported folder only. The Space README has the front matter below, plus a few
   plain lines:
   - what the demo does;
   - that all data is simulated;
   - that replays need no key;
   - a link to the GitHub repository.

   ```yaml
   ---
   title: BeamPilot
   emoji: 🎯
   colorFrom: indigo
   colorTo: blue
   sdk: gradio
   sdk_version: 6.30.0
   python_version: "3.12"
   app_file: scripts/space_app.py
   short_description: Muse Spark agent measures laser M² to ISO 11146
   license: mit
   pinned: false
   suggested_hardware: cpu-basic
   ---
   ```

   A test reads `space/README.md` and checks the front matter against the
   project:
   - `sdk_version` equals the Gradio pin in `pyproject.toml`;
   - `app_file` exists;
   - `short_description` is at most 60 characters.
2. **`requirements.txt`**, pinned to the versions tested locally. Gradio is
   **not** listed, because `sdk_version` pins it (6.30.0, the same pin as the
   `demo` extra).

   ```text
   numpy==2.5.4
   scipy==1.18.1
   scikit-image==0.26.0
   matplotlib==3.11.2
   openai==3.38.0
   python-dotenv==1.2.4
   ```

   A test checks that these names match `[project] dependencies` in `pyproject.toml`.
3. **Entry point `scripts/space_app.py`** (about 10 lines). It puts `src/` on
   `sys.path` and calls `beampilot.app.main()`. This keeps the src layout and the
   `REPO_ROOT` lookup that finds `docs/runs/`; a root `app.py` would also work. The
   package is not pip-installed on the Space, because an installed copy would
   not find `docs/runs/`.
4. **`docs/runs/lens_v3_L5.json`, `docs/runs/lens_smoke_s920.json`, and
   `docs/runs/recorded.csv`**: the two lens replays and the manifest of
   recording dates (section 2).
5. **Code and tests:**
   - the `app.py` changes from section 3 and the replay labels;
   - tests for the spend cap (fake costs), the live switch, the no-key banner, the key sentinel, the run-file scan and the requirement pins;
   - the run-file scan looks for drive and home paths, e-mail addresses and key-like strings. It also looks for the local user name, home folder and git e-mail of whoever runs the tests, read at test time, so no personal data is written into the test;
   - the new env vars added to `.env.example` as comments (no values; Baseten entries untouched).
6. **README:** a "Hosted demo" line with the Space URL, added after the first
   successful deploy.

## 6. Code delivery: a git remote is a problem, so upload instead

You asked for a separate git remote for the Space and a push of `main`. Two
problems block a plain push:

1. **History contains a binary file.** `docs/figures/hero.png` (330 kB) is stored
   as a plain git blob in commits 18397f1 and 6c1b414. Forum reports show the Hub
   rejects pushes that contain binary files not stored through Xet, and that
   removing the file from the latest commit is not enough. Examples:
   [HF forum: push rejected due to binary files](https://discuss.huggingface.co/t/push-reject-due-to-binary-files/88990),
   [an example error log](https://huggingface.co/spaces/Alexramsal/AcuarelaPortraits/blob/main/git_error.txt).
   The fix would be **rewriting history**, which our rules forbid.
2. **A new Space already has a first commit** (`.gitattributes`, `README.md`),
   created by the Hub. Pushing `main` on top of it is not a fast-forward, so it
   would need a **force push** (forbidden) or a merge of the Space's commit into
   `main`.

**Recommended instead: upload a clean export of `main` with the `hf` CLI.**

- `git archive main` exports exactly the committed files: no `.env`, no `runs/`, no `.venv`.
- `hf upload` sends them as one new commit on top of the Space's first commit. That is no force push and no history rewrite. The CLI handles the PNG through Xet by itself.
- GitHub stays the only git history. The Space is a deploy target.

**Decision 3 (taken):** upload with the `hf` CLI.

**`.gitattributes`:** the repository has none (checked 31 Oct 2026). So the
upload cannot overwrite the Space's own `.gitattributes`, which the Hub created
for Xet/LFS handling. The deploy command still passes
`--exclude ".gitattributes"`, so that a file added later can't replace it.

## 7. Steps for the maintainer (browser and PowerShell)

I will not handle any token or password. Steps 1–4 are in the browser; steps 5–7
are in PowerShell, run by you.

1. **Plan:** subscribe to PRO (Decision 1), and keep it until after the vote result
   on 16 Nov: https://huggingface.co/subscribe/pro
2. **Create the Space:** https://huggingface.co/new-space. Settings:
   - Owner: your user; name `beampilot`; licence MIT;
   - SDK **Gradio**; hardware **CPU Basic (free)**; visibility **Public**.
3. **Add the secret and variables** under the Space's *Settings → Variables and secrets*:
   - **New secret** `MODEL_API_KEY`: paste the key there yourself.

     If the Meta Model API console allows a second key, create a **dedicated
     key for the Space**, and revoke it after judging; then the Space can never
     spend from your main key. (Unverified whether the console allows a second
     key.)
   - **New variable** `BEAMPILOT_LIVE_RUNS` = `off` for the first deploy. Turn it on after the replay check.
   - **New variable** `BEAMPILOT_SPEND_CAP_USD` = `5`.
4. **Create a write token** at https://huggingface.co/settings/tokens. Choose
   *Fine-grained*, with write access to this one Space only.
5. **Install the `hf` CLI in its own tools venv** (not in the project venv):

   ```powershell
   py -3.12 -m venv C:\dev\hf-tools
   C:\dev\hf-tools\Scripts\python.exe -m pip install -U huggingface_hub
   C:\dev\hf-tools\Scripts\hf.exe auth login
   ```

   `hf auth login` asks for the token in a hidden prompt; paste it there. Answer
   **n** to "Add token as git credential?" (we don't push with git).
6. **Deploy main** (replace `<HF_USER>`; run from `C:\dev\beampilot` on a clean,
   up-to-date `main`):

   ```powershell
   git switch main
   git status
   git pull
   Remove-Item -Recurse -Force "$env:TEMP\beampilot-space" -ErrorAction SilentlyContinue
   git archive --format=zip -o "$env:TEMP\beampilot-space.zip" main
   Expand-Archive "$env:TEMP\beampilot-space.zip" -DestinationPath "$env:TEMP\beampilot-space" -Force
   Copy-Item "$env:TEMP\beampilot-space\space\README.md" "$env:TEMP\beampilot-space\README.md" -Force
   C:\dev\hf-tools\Scripts\hf.exe upload <HF_USER>/beampilot "$env:TEMP\beampilot-space" . --repo-type space --exclude ".gitattributes" --commit-message "Deploy main $(git rev-parse --short main)"
   ```

   What each line does:
   1. `git switch main`: switch to `main`.
   2. `git status`: check there are no local changes.
   3. `git pull`: fetch the latest `main` from GitHub.
   4. `Remove-Item`: delete the previous export.
   5. `git archive`: export the committed files of `main` (no `.env`, no `runs/`) as a zip.
   6. `Expand-Archive`: unzip it.
   7. `Copy-Item`: put the Space README, with its front matter, in place of the GitHub README.
   8. `hf upload`: upload the folder as one commit to the Space, keeping the Space's own `.gitattributes`.

   The Space builds and restarts. Watch the *Logs* tab on the Space page.
7. **Later deploys:** repeat step 6. Files deleted from `main` are not deleted on
   the Space by an upload; remove them in the Space's *Files* tab, if any.
8. **Optional:** `C:\dev\hf-tools\Scripts\hf.exe auth logout` when done; revoke the
   token after the hackathon.

## 8. Dry run on the Space (after the first deploy)

1. With `BEAMPILOT_LIVE_RUNS=off`, check:
   - the banner says live runs are off;
   - the Start button is disabled;
   - all **five** replays play, each labelled "Recorded run, <date>,
     muse-spark-1.2", including both lens runs with their segment plots.
2. Search the build and run logs for the first characters of the key; there
   should be no match. Check by eye, and never paste the key anywhere.
3. Set `BEAMPILOT_LIVE_RUNS=on` and do one live run (about $0.05–0.13).
   - Check that a second tab queues instead of running in parallel.
   - Check the cost line in the report.
4. Temporarily set `BEAMPILOT_SPEND_CAP_USD=0.3` and confirm that the budget
   message appears; then set it back to 5.
5. Record the Space URL in the README (one commit), then upload again.

## 9. Time estimate

| Work | Who | Time |
|---|---|---|
| Code: switch, spend cap, concurrency, banner, entry point, requirements, front matter, lens replay file | Claude | 1.5 h |
| Tests (section 4–5) and the ruff/pytest gate, small commits on a branch | Claude | 1 h |
| Browser steps 1–4 and CLI steps 5–6 | maintainer | 20–30 min |
| First build, dry run (section 8), fixes | both | 1 h |
| Buffer for Space-specific surprises | — | 1 h |
| **Total** | | **about 4.5–5 h**; API cost ≤ $1 for the dry run |

This fits comfortably before the deadline (Wed 4 Nov, 21:00 ET), leaving the
video script as the next step.
