# Strategy: `shap_prompt`

`zero_shot` plus the **results of the SHAP analysis** already run on the gradient-boosting
control. Same call, same prose rendering of the window, same five options, same instructions
(`zero_shot`'s `prompt.txt` is the template). The addition is the ranked list of the window
variables by how much they move that model's answer.

## What it asks

"Does telling the model what the tree model looks at help it decide, with the whole window
in hand?" The window is the shared `render_case` every strategy sends; the addition is the
**whole** SHAP ranking (`--samples` is accepted and ignored, so no run can cut it). A ranking
that names fields outside the shared view comes from a stale model and is refused.

## Where the ranking comes from

`results_harness/control/_analysis/shap_importance.csv`, written by

```powershell
python -m src.control.xgboost_diagnostics
```

Read at run time, so it always follows the current model. The block explains what the numbers
mean (a mean absolute SHAP value says how much a variable matters, not which decision it
favours) and then lists `rank. variable: value` for the top rows — **every ranked variable of the shared view**.

## Where the reference goes

```
zero_shot prompt (wording unchanged)
  ├── the patient state, rendered as clinical prose
  ├── "What a model trained on thousands of past decisions of this kind pays attention to: ..."
  │    1. remi_ce_slope_short: 0.9274
  │    2. cp_remi: 0.5019
  │    ...
  ├── "Based ONLY on the information above, decide ..."
  └── the five options (a-e)
```

## Run

```powershell
python harness/run.py --strategy shap_prompt --split test --effort none --workers 64

# paste only the top 30 rows of the ranking
python harness/run.py --strategy shap_prompt --split test --samples 30 --effort none --workers 64
```

How many rows are pasted is the launcher's `--samples` flag (`n_samples`/`k` are accepted as
aliases); `rows=` does the same from code. The number is recorded as `k` in the leaderboard, so
runs with different amounts of the analysis stay distinguishable.

## Metadata per window

```json
{
  "importance_path": "...\\control\\_analysis\\shap_importance.csv",
  "rows_shown": 234,
  "k": 234,
  "prompt_chars": 7986
}
```

## Cost

One call per window, with a moderate prompt: ~8.0k characters (~2k tokens) against ~3.3k
(~900 tokens) for `zero_shot`.
