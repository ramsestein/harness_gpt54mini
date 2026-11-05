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
- `rag` — `zero_shot` plus the whole contents of a curated context document
  (`harness/rag/context.md`: verbatim guideline excerpts with their sources), inserted right after
  the patient state.
- `shap_prompt` — `zero_shot` plus the SHAP ranking of the tree control, inserted in the
  same place: it ranks the window variables by their **mean absolute** SHAP value, so it
  says how much each one moves that model's answer, not in which direction.

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
  the numeric fields of the same view. The three tree-based ones (XGBoost, the failure
  predictor, SHAP) read them through the *control view*, which recomputes the effect-site
  features without the samples after -30 s (see [Effect-site leak at t0](#effect-site-leak-at-t0)).
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
| `gemini-pro-latest` | `few_shot` k=8, high | 0.4304 — **provisional**, 2 of the 3,457 windows failed |

> Provisional means `summary.json` does not report `errors: 0`, so the metrics come from the
> windows that survived. The four rows in that state are all `gemini-pro-latest`: `few_shot`
> none k=8 (1 window), `few_shot` high k=8 (2), `zero_shot` none (2) and `zero_shot` high
> (1). They are four windows that this model does not answer at all (they time out even with
> a five-minute allowance, deterministically for two of them), so a `--resume` only closes
> them if the provider answers them that time.

The same metric for the controls:

| control | result_real |
|---|---|
| `human` (replaying the clinician's own action) | 1.0000 |
| `xgboost` (the tree control, control view) | 0.6349 |
| `retrieval_majority` | 0.4128 |
| `persistence` | 0.3254 |
| `always_no_action` (the majority class) | 0.2728 |
| `sim_normality` | 0.2112 |
| `random` (chance) | 0.1918 |

Three readings worth keeping in mind. First, **no prompt architecture beats the tree
control**, and none beats a plain retrieval plus majority vote of what the clinicians did in
similar windows **outright**: the best `few_shot` of `deepseek-flash` (+0.008) and of
`gemini-pro-latest` (+0.017) against the kNN are ties under a per-case paired bootstrap
(95% CI [-0.013, +0.030] and [-0.002, +0.036]), while the project's model sits below it
(-0.023, CI [-0.045, -0.002]). Those two models do beat `gpt-5.4-mini` with `few_shot`
(+0.033 and +0.041, significant), so what survives is: the kNN is not beaten, the other two
models match it, and `gpt-5.4-mini` stays under it. Most architectures sit at or below the
trivial `always_no_action`. Second, `support` at
threshold 0 *is* the tree control (0.6349) and walks towards the language model as the
threshold grows (0.4050 at k=80 on the project's model), which is what makes the hybrid worth
measuring: it prices the model against the control window by window. Third, the labels are
not equally hard: `result_1`/`result_aux` come from a model trained on the clinicians'
actions, so a control that votes those labels scores much higher on them than on the real
action, which is why the retrieval control votes `result_real` only.

Every number comes from the runs under `results_harness/` (not committed), summarised by
`python -m src.leaderboard` into `results_harness/leaderboard.csv` (`--models-only` writes
`results_harness/leaderboard_models.csv` with the runs of the other models).

## Effect-site leak at t0

View 4 keeps the effect-site concentration (Ce) at the decision point. Near the decision the
Ce series is sampled every 30 s, and for remifentanil the decision point falls at or just
after the pump's target change, so the t0 sample (and every Ce feature computed with it:
current value, slopes, deltas, trend code) already moves with the action: it records the
decision, not the state the decision was taken in. The dataset side is documented in the
AnesLLM dataset's `docs/audit_report.md` §3.1.

The tree-based controls (XGBoost, `support`, the failure predictor, the SHAP ranking behind
`shap_prompt`) therefore read a **control view** on top of the shared one
(`src/control/control_view.py`, `CONTROL_VIEW_VERSION = 2`):

- every `ce_*`, `propofol_ce_*` and `remi_ce_*` feature recomputed from the raw Ce series with
  the samples at t <= -30 s only (same names as in the shared view);
- the target set by the last past adjustment of each drug (`prior_target_propofol`,
  `prior_target_remi`), which the prompt already states in prose. Its age is left out on
  purpose: the `no_action` windows were sampled away from recent adjustments, so "time since
  the last adjustment" mostly encodes how the benchmark was built.

Trained models record the control view, and a model fitted without it is refused. Where the
cut goes, measured with XGBoost on `test` (Ce features recomputed up to each cut, no prior
targets):

| last Ce sample kept | result_real |
|---|---|
| 0 s (the t0 sample) | 0.740 |
| -30 s | 0.629 |
| -60 s | 0.614 |
| -90 s | 0.595 |
| -120 s | 0.548 |

Dropping the t0 sample costs 11 points and every further 30 s only 1.5-2: the jump is the
leak, the slope before it is legitimate information that ages. With the prior targets the
retrained control scores **0.6349** (0.7495 with the t0 sample). It still beats every language
model, by 0.20-0.25 (0.32-0.36 with the t0 sample), and `support` and the SHAP ranking follow
it.

### The language models do not exploit it

The language models stay on view 4, t0 sample included, so they keep information the tree
controls are denied. The asymmetry favours them: the 0.20-0.25 gap is measured with the tree
at a disadvantage. Their runs stay valid because they do not turn that sample into correct
answers, which `test` shows three ways.

**Ablation.** `zero_shot_ce_cut` (`harness/zero_shot_ce_cut/`) is `zero_shot` with every
effect-site field of the prompt cut at -30 s, as in the control view. Against the `zero_shot`
run of the same model and effort (paired, case-clustered bootstrap):

| model | effort | `zero_shot` | cut at -30 s | difference (95% CI) |
|---|---|---|---|---|
| `gpt-5.4-mini` | none | 0.2760 | 0.2826 | +0.007 [-0.004, +0.018] |
| `gpt-5.4-mini` | high | 0.2496 | 0.2525 | +0.003 [-0.007, +0.012] |
| `deepseek-flash` | none | 0.2742 | 0.2713 | -0.003 [-0.016, +0.010] |
| `deepseek-flash` | high | 0.2800 | 0.2670 | -0.013 [-0.024, -0.001] |
| `gemini-pro-latest` | none | 0.2598 | 0.2575 | -0.002 [-0.011, +0.006] |
| `gemini-pro-latest` | high | 0.2631 | 0.2544 | -0.009 [-0.018, -0.000] |

The `gemini-pro-latest` rows are on the windows both runs scored (3433 and 3455 of 3457; the
rest failed at the API).

**The jump.** Where the remifentanil Ce at t0 sits at least 0.1 above the last sample at
t <= -30 s, 63% of the 142 windows are `increase_opioid`; where it sits at least 0.1 below,
64% of the 434 windows are `reduce_opioid`; in the 2844 windows without such a jump both are
22%. A model exploiting the jump would choose that action there. Share of the windows where
each run does, with the t0 sample (`zero_shot`) / without it (`zero_shot_ce_cut`):

| windows | label | `gpt-5.4-mini` none | high | `deepseek-flash` none | high | `gemini-pro-latest` none | high |
|---|---|---|---|---|---|---|---|
| jump up (142) | 63% `increase_opioid` | 6% / 10% | 13% / 15% | 8% / 9% | 12% / 6% | 12% / 13% | 12% / 12% |
| jump down (434) | 64% `reduce_opioid` | 22% / 22% | 5% / 6% | 9% / 8% | 8% / 7% | 7% / 6% | 7% / 7% |

**Every stored run.** On the 567 `test` windows that only the t0 sample makes answerable (the
uncut tree gets them right, the cut one does not), the 50 stored language-model runs average
0.190, against 0.237 on the windows both trees miss. The largest gap of any of them is +0.042,
from a `few_shot` run, which like `rag` shows labelled cases in the prompt; `support` before
the cut, which answered with the uncut tree, gained +0.35 there.

At high effort `deepseek-flash` and `gemini-pro-latest` lose a little without the sample
(-0.013 and -0.009, intervals that just exclude zero). For `deepseek-flash` part of it is the
jump: after an upward jump it chooses `increase_opioid` twice as often with the sample as
without it. `gemini-pro-latest` makes the same choices after a jump with and without it, and
its loss is spread over the windows the cut changes (-0.014) and the rest (-0.007). Either
loss is at most 0.013, against the 0.111 the tree loses. A learner fitted to the labels finds
the shortcut; models that are not fitted to them do not.
