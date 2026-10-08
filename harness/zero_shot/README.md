# Strategy: `zero_shot`

**Baseline.** A single model call with the window rendered as clinical prose and
the prompt template in this folder (`prompt.txt`, five options, answer = 1 letter).

- Cost: 1 call per window.
- Prompt: `prompt.txt` — all window data as text, five options only
  (`a` increase_hypnotic, `b` reduce_hypnotic, `c` increase_opioid,
  `d` reduce_opioid, `e` no_action).
- Output parsing: `src.parsing.parse_action`; an unparseable answer falls back to
  `no_action` and is flagged in `Prediction.meta["parsed"]`.

## Run

```powershell
python harness/run.py --strategy zero_shot --split dev --cases 2
```

Results land in `results_harness/zero_shot/<run_id>/`.
