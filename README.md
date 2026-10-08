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
  leaderboard.py         #   cross-strategy table (results_harness/leaderboard.csv)
  control/               #   no-model baselines (see src/control/README.md)
harness/                 # one folder per strategy (see harness/README.md)
tests/                   # pytest suite (executable specification)

# not in the repository (see .gitignore):
results_harness/         # per-strategy run outputs (predictions + summary)
dataset/                 # the dataset, its reports and its original utilities
```

## Status

Test-driven: the suite (`tests/`) defines the contract of `src/` and of every strategy
(321 tests, `live` excluded). Seven prompt architectures, one folder each under
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

Plus the no-model controls in `src/control/`: the chance level, the label prevalence, the
clinician's own action, pure retrieval, and a simulator-driven policy.

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

Reference numbers on the `test` split (3,457 windows), best run per strategy. The three
metrics are `result_real` (the prediction equals the action the clinician took),
`result_1` (it equals the consensus primary action) and `plausible` (it is `result_1` or
`result_aux`):

| strategy | result_real | result_1 | plausible |
|---|---|---|---|
| `human` (the clinician, replaying `result_real`) | 1.0000 | 0.5684 | 0.8131 |
| `retrieval_majority` (no model) | 0.4675 | 0.5432 | 0.7637 |
| `few_shot` | 0.4056 | 0.4452 | 0.6644 |
| `structured_json` | 0.2754 | 0.3156 | 0.5022 |
| `always_no_action` (majority class) | 0.2728 | 0.3928 | 0.4741 |
| `critic_verifier` | 0.2693 | 0.2988 | 0.4897 |
| `cot` | 0.2621 | 0.2641 | 0.4848 |
| `critic_verifier_sim` | 0.2598 | 0.2456 | 0.5114 |
| `zero_shot` | 0.2569 | 0.2326 | 0.4753 |
| `self_consistency` | 0.2537 | 0.2603 | 0.4628 |
| `sim_normality` (no model) | 0.2332 | 0.1938 | 0.4808 |
| `random` (chance) | 0.1918 | 0.1993 | 0.3917 |

Two readings worth keeping in mind. First, **no prompt architecture beats a plain
retrieval plus majority vote of what the clinicians did in similar windows**, and most of
them sit at or below the trivial `always_no_action`. Second, the labels are not equally
hard: `result_1`/`result_aux` come from a model trained on the clinicians' actions, so a
control that votes those labels scores much higher on them than on the real action —
which is why the retrieval control votes `result_real` only.

Every number comes from the runs under `results_harness/` (not committed); they are
summarised by `python -m src.leaderboard` into `results_harness/leaderboard.csv`.
