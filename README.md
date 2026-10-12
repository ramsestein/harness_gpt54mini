# prueba_metodos_llm

Comparison of **prompting architectures** on the AnesLLM clinical dataset
(anesthesia decision windows, VitalDB). The dataset is not part of this repository
(see `.gitignore`); the code reads it from `dataset/data/`.

Base model: **`gpt-5.4-mini`** (configurable with `ANESLLM_MODEL`).

## Layout

```
src/                     # reusable core (shared by every harness)
  config.py              #   paths, .env, model settings
  dataset.py             #   window loading from dataset/data/
  prompting.py           #   window rendering + prompt template
  llm.py                 #   model client (OpenAI-compatible API)
  parsing.py             #   answer -> action from the vocabulary
  metrics.py             #   accuracy / agreement with result_1 / result_plausible
  results.py             #   prediction persistence (JSONL)
  harness_base.py        #   Prediction + Strategy (common interface)
  harness_registry.py    #   strategy discovery
  model_probe.py         #   which models a key reaches, and which accept reasoning_effort
  view.py                #   the shared view of a window: what no model may see
  leaderboard.py         #   cross-strategy tables (leaderboard.csv, leaderboard_models.csv)
  control/               #   no-model baselines and the tree controls (see src/control/README.md)
harness/                 # one folder per strategy (see harness/README.md)
tests/                   # pytest suite (executable specification)

# not in the repository (see .gitignore):
results_harness/         # per-strategy run outputs (predictions + summary)
dataset/                 # the dataset, its reports and its original utilities
```

## Status

Test-driven: the suite (`tests/`) defines the contract of `src/` and of every strategy
(574 tests, `live` excluded). Nine prompt architectures, one folder each under
`harness/`:

- `zero_shot` — clinical prose prompt, one call.
- `structured_json` — the same call with the whole prompt as a JSON document.
- `self_consistency` — `zero_shot` sampled N times + majority vote (shares its prompt).
- `few_shot` — the k most similar training windows in the prompt, retrieved by
  standardised Euclidean distance over the numeric features.
- `cot` — ReAct-style chain of thought: explicit lookups, trends and comparisons
  before answering.
- `critic_verifier` — a second, critical call confirms or overrides the first.
- `critic_verifier_sim` — the second call judges **simulated** futures: both candidates
  are driven through the Python Anesthesia Simulator five minutes ahead.
- `rag` — `zero_shot` plus the whole contents of a context document (`rag/context.md`)
  prepended to the prompt.
- `shap_prompt` — `zero_shot` plus the whole SHAP ranking of the tree control: which
  features push its answer and in which direction, before the window itself.

Plus `support`, which calls no model: the tree control answers the windows it is confident
about and an already-stored `zero_shot` run answers the rest, so a run is free. It leans on
the `zero_shot` of **its own effort and its own model** and records which one it used.

And the no-model controls in `src/control/`: the chance level, the label prevalence, the
clinician's own action, pure retrieval, a simulator-driven policy, and the trained tree
control with its failure predictor.

## Shared view of the window (`src/view.py`)

Every strategy and every control sees **the same window**:

- `src/view.py` lists the fields no model may see: identifiers and pipeline bookkeeping,
  future and outcome fields (`surgery_pct`, case/operation/anaesthesia end, ICU stay,
  discharge, in-hospital death, every `pt_intraop_*` whole-case total), the `n_*` counters,
  the excluded pipeline products (`causal_depth_json`, `depth_score`, `remi_causal_flag`,
  `drug_concentrations_json`), the volatile agent (`exp_sevo`, `mac`) and the plasma
  concentrations `cp_*` and pump targets `ct_*` (views 3-4: for remifentanil they already
  carry the action at t = 0). It also drops any
  time-series sample stamped after the decision point.
- `dataset` applies the view on load: `window.input` **is** the view (228 of the 280 fields);
  the untouched record stays in `window.raw_input` for auditing only.
- `prompting.render_case` renders **every** field of the view that has a value: the clinical
  summary, then a "Full monitoring record" block with everything else. All the language
  strategies send it unchanged (`structured_json` wraps it in JSON; `shap_prompt` adds the
  whole SHAP ranking of the tree control after it).
- The retrieval index, the XGBoost control, the failure predictor and the SHAP analysis read
  the numeric fields of the same view.
- The view has a version (`view.VIEW_VERSION`). `run.json`, the retrieval index and the
  trained models record it; a stale index is rebuilt, a stale model or a run on another view
  is refused, and the leaderboard only lists runs of the current view (`--all-views` to list
  them all, with a `view` column). `tests/test_shared_view.py` pins the contract down.

### Relaunching on a new view

```powershell
python harness/few_shot/build_index.py --rebuild          # 1. retrieval index
python -m src.control.train_xgboost                        # 2. tree control
python -m src.control.xgboost_diagnostics                  # 3. SHAP ranking (shap_prompt)
python -m src.control.failure_model                        # 4. failure predictor (support)
python -m src.control.run_controls --split test --only all --workers 16   # 5. controls
# 6. every strategy, effort and k; zero_shot before support (support reuses its answers)
python -m src.leaderboard                                  # 7. table
```

## Another model in the same comparison

The model is a property of the **run**, not of the harness: `harness/run.py` takes it,
together with the environment variable holding its key and the endpoint that serves it.

```powershell
python harness/run.py --strategy zero_shot --split test --model deepseek-flash `
    --api-key-env DEEPSEEK_API_KEY --base-url https://api.deepseek.com/v1 --workers 64
```

The three land in `run.json` and in every prediction's metadata; the key itself never does,
only the name of the variable holding it. Two runs of the same strategy, split, effort and
knob on two models are **two experiments**, and the table keeps them apart:
`leaderboard.csv` lists them all with a `model` column, `leaderboard_models.csv` only the
runs made with another model. `--resume` restores the effort, the workers and the knob from
`run.json`, but **not** the model, the key or the endpoint: pass those again, or the resumed
windows are recomputed with the project's model and silently mixed into the run.

`src/model_probe.py` answers the two questions worth asking before choosing a model: which
models a key reaches, and which of them accept the `reasoning_effort` the harness sends.

```powershell
python -m src.model_probe --api-key-env GEMINI_API_KEY `
    --base-url https://generativelanguage.googleapis.com/v1beta/openai/ --list-only
python -m src.model_probe --api-key-env GEMINI_API_KEY --base-url ... `
    --models gemini-pro-latest --effort high
```

It calls each model twice, with and without the parameter, which separates "it takes the
parameter" from "it answers only when the parameter is omitted" from "the key does not reach
it". Note the third case a gateway can produce: *accepting* the parameter and ignoring it.
A single successful call only proves the parameter was not rejected; comparing the tokens of
a real answer with and without it is what shows whether it did anything.

## Launching a grid

A grid is `harness/run.py` once per combination (strategy x effort x knob), `zero_shot`
before `support` at the same effort, because `support` reuses its answers. The repository
carries no batch launcher: `launch.py` used to expand that grid for the project's model and
was removed, so a run over several settings is a sequence of `harness/run.py` calls (or a
script of your own). Whatever launches them, a run is finished only when `summary.json`
reports `errors: 0`: a run that lost its windows to a provider problem (an empty balance, a
quota) still writes a run directory, and its metrics are computed over the windows that
survived, so they look like results and are not.

## Usage

```powershell
python -m pip install -r requirements.txt

# run a strategy over a split
python harness/run.py --strategy zero_shot --split dev --cases 2

# tests
pytest -m "not live"     # fast, no network
pytest -m live           # smoke tests against gpt-5.4-mini (uses .env)
pytest                   # everything
```

Environment variables (see `.env`):

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | model credentials |
| `ANESLLM_MODEL` | `gpt-5.4-mini` | base model |
| `ANESLLM_BASE_URL` | — | alternative endpoint |
| `ANESLLM_TIMEOUT` | `60` | timeout in seconds |
| `ANESLLM_MAX_RETRIES` | `3` | retries on SDK errors |
| `ANESLLM_TEMPERATURE` | `0` | sampling temperature (`none` omits it) |
| `ANESLLM_REASONING_EFFORT` | `high` | reasoning effort (`none` omits it) |
| `ANESLLM_SKIP_LIVE` | — | `1` disables the `live` tests |

## Generation settings

Every call asks for **temperature 0** and **`reasoning_effort: high`**
(override the latter per run with `--effort none|low|medium|high`).
`gpt-5.4-mini` is a reasoning model: it rejects any temperature other than its
default and does not know the `minimal` effort. The client drops a parameter
(or value) the model refuses, retries, and remembers only that exact rejected
value: the first call pays for it. Whatever was dropped is reported per window in
`prediction.meta.dropped_params`.

## Results

> These are the runs of the current shared view (view 4), `test` split, 3,457 windows.
> Numbers from earlier views are not comparable: several strategies saw different fields
> and the tree control read future ones.

`result_real` — the prediction equals the action the clinician took — best run per model:

| model | best strategy | result_real |
|---|---|---|
| `gpt-5.4-mini` (the project's model) | `few_shot` k=8, high | 0.3899 |
| `deepseek-flash` | `few_shot` k=10, high | 0.4206 |
| `gemini-pro-latest` | `few_shot` k=8, high | 0.4304 (2 windows failed) |

The same metric for the controls:

| control | result_real |
|---|---|
| `human` (replaying the clinician's own action) | 1.0000 |
| `xgboost` (the tree control) | 0.7495 |
| `retrieval_majority` | 0.4128 |
| `persistence` | 0.3254 |
| `always_no_action` (the majority class) | 0.2728 |
| `sim_normality` | 0.2112 |
| `random` (chance) | 0.1918 |

Three readings worth keeping in mind. First, **no prompt architecture beats the tree
control**, and none beats a plain retrieval plus majority vote of what the clinicians did in
similar windows: most sit at or below the trivial `always_no_action`. Second, `support` at
threshold 0 *is* the tree control (0.7495) and walks towards the language model as the
threshold grows (0.5762 at k=80 on the project's model), which is what makes the hybrid worth
measuring: it prices the model against the control window by window. Third, the labels are
not equally hard: `result_1`/`result_aux` come from a model trained on the clinicians'
actions, so a control that votes those labels scores much higher on them than on the real
action, which is why the retrieval control votes `result_real` only.

Every number comes from the runs under `results_harness/` (not committed), summarised by
`python -m src.leaderboard` into `results_harness/leaderboard.csv` (`--models-only` writes
`results_harness/leaderboard_models.csv` with the runs of the other models).
