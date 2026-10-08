# Strategy harness

Each **strategy** lives in its own folder: `harness/<name>/`.

```
harness/
├── zero_shot/          # 1 call, direct prompt
├── few_shot/           # k labelled examples from train in the prompt
├── cot/                # short reasoning before the letter
├── self_consistency/   # N samples + majority vote
├── structured_json/    # JSON output with action + probability
└── critic_verifier/    # generator + critic (2 steps)
```

## Strategy contract

Every folder must contain `strategy.py` with **one** concrete subclass of
`src.harness_base.Strategy`:

```python
from src.harness_base import Prediction, Strategy
from src import dataset, llm, parsing, prompting


class MyStrategy(Strategy):
    name = "my_strategy"            # must match the folder name
    description = "What it does, in one line."

    def predict(self, window) -> Prediction:
        prompt = prompting.build_prompt(window)
        text = llm.complete(prompt)          # always through src.llm (never the SDK)
        action = parsing.parse_action(text)
        if action is None:
            action = "no_action"             # or raise; see the folder's README
        return Prediction(
            window_id=window.window_id,
            action=action,
            raw_output=text,
            meta={},
        )
```

Rules:

- **Do not** reimplement dataset loading, prompts, the LLM client, parsing or metrics: all of that comes from `src/`.
- **Do not** open OpenAI SDK connections inside the harness: use `src.llm`.
- Call the model through the module (`llm.complete(...)`) so tests can intercept it.
- The prediction must always be an action from the vocabulary (`dataset.ACTIONS`).

## Launcher

``harness/run.py`` runs one strategy over a slice of a split and writes the
results to ``results_harness/<strategy>/<run_id>/``:

```
results_harness/zero_shot/20261008-153012_dev/
├── predictions.jsonl   # one line per window (action, raw output, real action, ...)
├── summary.json        # accuracy, agreement with result_1, by difficulty, confusion
└── run.json            # what was run: strategy, split, cases, limit, model, timestamps
```

```powershell
python harness/run.py --strategy zero_shot --split dev --cases 2   # first 2 cases of dev
python harness/run.py --strategy zero_shot --split test --limit 20 # first 20 windows of test
python harness/run.py --strategy zero_shot --split dev --cases 2 --dry-run  # no model calls
```

| Flag | Meaning |
|---|---|
| `--strategy` | folder under `harness/` (default `zero_shot`) |
| `--split` | `dev`, `train` or `test` (default `dev`) |
| `--cases N` | use only the first N cases of the split |
| `--limit N` | cap the number of windows |
| `--workers N` | windows processed concurrently (default 1) |
| `--effort LEVEL` | reasoning effort: `none`, `low`, `medium` or `high` (default: config, `high`) |
| `--samples N` | samples for sample-based strategies such as `self_consistency` |
| `--results-root` | override the results folder |
| `--dry-run` | render the prompts without calling the model |
| `--quiet` | no per-window progress output |

The model is called with temperature 0 and `reasoning_effort: "high"`. If the
model refuses a parameter or a value, the client drops it and records it in
`predictions.jsonl` → `meta.dropped_params` (e.g. `gpt-5.4-mini` rejects
temperature 0 and the `minimal` effort).

## Tests

```powershell
pytest                       # full suite (the `live` tests call the real model)
pytest -m "not live"         # no real calls
$env:ANESLLM_SKIP_LIVE=1; pytest
```

Base model: `gpt-5.4-mini` (configurable with `ANESLLM_MODEL`).
