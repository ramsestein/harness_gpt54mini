# Strategy: `support`

The **tree model where it is sure, the language model where it is not**. This strategy runs the
XGBoost control over the split, keeps its answer only on the windows it is confident enough
about, and for the remaining windows reuses the answers an earlier `zero_shot` run already
recorded. Nothing is called: a run costs no tokens and takes about a minute.

## The three knobs

`effort` — which `zero_shot` run supplies the fallback. **This is the run's own effort**:
`--effort none` defers to the `zero_shot` run launched with no reasoning effort, `--effort low`
to the low one and `--effort high` to the high one, so the hybrid is measured against the
language model it actually leans on. Each prediction records it (`reasoning_effort`), which is
how the leaderboard separates the three variants — a strategy that calls no model records no
effort of its own in `run.json`, so the metadata is the source.

`threshold` — how sure is sure enough, on the `[0, 1]` scale of the score below. From the
launcher it is a percentage through the generic `--samples` flag: `--samples 60` means "trust
the control when its score is at least 0.60". `0` keeps every window and reproduces the
control's own accuracy, which is the number the deferral has to beat.

`signal` — what the score means:

| signal | score | needs |
| --- | --- | --- |
| `confidence` (default) | the control's own probability for its best action | the control model |
| `failure` | `1 - P(the control is wrong)` from the failure predictor | `python -m src.control.failure_model` |
| `combined` | the average of the two | both |

The failure predictor is the one from `src/control/failure_model.py`: a classifier trained on
out-of-fold failures of this very control, which reads the window's own variables. Measured on
`test`, it ranks the control's failures with AUC 0.7675, while the control's own confidence
reaches 0.8445 — so `confidence` is the stronger risk signal, and `failure` is kept because it
is the direct reading of the question "where will the tree model be wrong?".

## Where the fallback comes from

The newest **finished, non-dry** run of `zero_shot` **with the same effort**, or the one given
in `fallback_run`. Its `predictions.jsonl` is read by `window_id`, so the fallback is exactly the
answers already on record and no call is repeated. The run it used is written into every
prediction's metadata (`fallback_run`), and a window the stored run does not cover keeps the
control's answer (flagged `fallback_missing`).

```powershell
# the three variants of the same threshold, one per effort of the stored zero_shot answers
python harness/run.py --strategy support --split test --samples 60 --effort none --workers 8
python harness/run.py --strategy support --split test --samples 60 --effort low  --workers 8
python harness/run.py --strategy support --split test --samples 60 --effort high --workers 8
```

With no `--effort` at all the run follows the shared configured default (`high` at the moment
of writing), exactly like every other strategy — so pass the flag explicitly when sweeping. If
no stored run of that effort exists, the strategy says which effort it was looking for instead
of silently using another one.

## Run

```powershell
# the reference: the control alone
python harness/run.py --strategy support --split test --samples 0 --workers 8

# defer everything below the 0.60 confidence score
python harness/run.py --strategy support --split test --samples 60 --workers 8
```

## What to expect

The control answers ~0.815 of `test` and the stored `zero_shot` answers between 0.247 and 0.295
depending on their effort, so handing windows over to the language model **lowers** the overall
accuracy: the mixture is a weighted average of the two, and the windows the control is least
sure about are still answered better by the control than by the prompt. The value of the table
row is therefore not the headline accuracy but the trade-off it makes visible: which share of
the windows can be delegated, and what that costs, for each effort of the fallback.

## Metadata per window

```json
{
  "signal": "confidence",
  "threshold": 0.6,
  "k": 60,
  "reasoning_effort": "high",
  "score": 0.7312,
  "confidence": 0.7312,
  "xgb_action": "reduce_opioid",
  "source": "fallback",
  "fallback_action": "no_action",
  "fallback_run": "20261008-184642_test"
}
```

## Cost

No model call. 3,457 test windows take about a minute (one XGBoost inference per window plus
reading the stored predictions).
