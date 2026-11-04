# Strategy: `zero_shot_ce_cut`

Ablation of `zero_shot` for the effect-site leak at t0 (main README, "Effect-site leak at
t0"): the same template, prose and call, with every effect-site field the window shows
(`ce_*`, `propofol_ce_*`, `remi_ce_*` and the `*_ce_trend` words of the prose) recomputed
from the Ce samples at t <= -30 s by `src/control/control_view.py`, the cut the tree
controls read. Compare it, paired, with the `zero_shot` run of the same model and effort.

It is a separate strategy so that it never replaces `zero_shot` in the leaderboard and
`support` never takes its answers as the fallback.

```powershell
python harness/run.py --strategy zero_shot_ce_cut --split test --effort none --workers 64
```

Two differences come with the cut: the dataset's `ce_*` scalars are the native-resolution
value at t0 (after the cut only the 30 s series is left), and its trend codes agree with the
control view's rule on ~98.6 % of the training windows. Each prediction records
`ce_cut_sec` and `control_view_version`.

On `test` the cut moves accuracy by +0.007 and +0.003 for `gpt-5.4-mini` (none, high) and by
-0.003 and -0.013 for `deepseek-flash`, the last the only interval that excludes zero; the
main README has the table, the intervals and what the models do after a jump of the Ce at t0.
