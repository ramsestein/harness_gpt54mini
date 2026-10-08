# Control baselines (no model)

The reference points the model strategies are compared against. They never call the
model, so they are free and instantaneous, and they answer two questions about the
dataset itself:

| Baseline | Question it answers |
|---|---|
| `random` | What does picking uniformly among the five actions score? (the chance level) |
| `always_<action>` (five of them) | What does always answering that one option score? (the label prevalence) |
| `human` | What does the clinician score? (the human value of the experiment) |
| `retrieval_majority` | What does the k most similar past situations' **real** outcome score? (retrieval without a model) |
| `sim_normality` | What does choosing the action whose simulated future is closest to normality score? |

The best of the five constants is the **majority-class baseline**: the number any
model has to beat to be useful.

The `human` baseline answers with `result_real`, so its agreement with the real action
is 1.0 by construction and carries no information. What it measures is the other two:
how often the clinician's action equals the consensus `result_1` (≈ 55 %) and how often
it is a plausible answer, `result_1` or `result_aux` (≈ 80 %). That is the ceiling of
the dataset, not a model result.

All of them also land in the cross-strategy table
(`results_harness/leaderboard.csv`, written by `src/leaderboard.py`).

## `retrieval_majority` (opt-in: loads the few-shot index)

The non-model reading of the few-shot strategy, and nothing more. Per window: retrieve
the **k = 10** most similar training windows with the same similarity tool few-shot uses
(the case window goes in raw) and answer with the action the clinicians took **most
often** in that neighbourhood -- what anesthesiologists did in the situations that look
most like this one.

Only `result_real` takes part. The dataset's `result_1`/`result_aux` are the output of a
model trained on the clinicians' actions, so scoring with them is circular: it measures a
consensus, not the clinicians. An earlier version of this control did exactly that
(`results_harness/control/retrieval_consensus/`) and was dropped. Measured on `test`,
the circularity cost almost nothing on the real action but inflated the circular metrics,
which is the point:

| control | result_real | result_1 | plausible |
|---|---|---|---|
| `retrieval_consensus` (circular, dropped) | 0.4706 | 0.6378 | 0.8296 |
| `retrieval_majority` (pure) | **0.4675** | 0.5432 | 0.7637 |

The similarity index already stores the real action of every training window, so this
control needs no extra corpus or cache.

```powershell
python -m src.control.run_controls --split test --only retrieval_majority
```

## `sim_normality` (opt-in: a simulation per window)

The only control that runs a policy instead of a constant. Per window: calibrate a
patient from the window, simulate the five actions five minutes ahead, and answer with
the future closest to the centre of the normal range, with every signal one dimension
normalised by its tolerance:

```
score = sqrt( sum_i ((x_i - target_i) / tolerance_i)^2 )
```

| signal | target | tolerance | why |
|---|---|---|---|
| `bis` | 50.0 | 10.0 | surgical anaesthesia band 40-60 |
| `map` | 77.5 | 12.5 | perfusion band 65-90 (no hypotension, no hypertension) |
| `co` | 6.0 | 2.0 | cardiac output band 4-8 L/min |
| `tol` | 0.0 | 1.0 | nociception index in [0, 1]; the target is 0 |

The bounds were measured on the simulator's own output, not guessed. Both tables are
overridable (`targets=`, `tolerances=`) so the normalisation can be re-run as a
sensitivity analysis; each run records the table it used and the per-signal deviation
of every candidate.

It is **not** part of the default sweep, and ``--workers`` does not help it: the
simulation is CPU-bound Python that holds the GIL, so threads give no speedup here
(measured parallel factor 1.02). One window costs ~0.4 s, of which ~0.23 s is building
the PAS patient and only ~0.15 s is the integration; the full `test` split takes ~22 min.

```powershell
python -m src.control.run_controls --split test --only sim_normality --workers 16
python -m src.control.run_controls --split test --only all --workers 16
```

## Run

```powershell
python -m src.control.run_controls --split test
python src/control/run_controls.py --split dev --cases 2
python -m src.control.run_controls --split test --seed 3 --results-root C:\scratch\control
```

| Flag | Meaning |
|---|---|
| `--split {dev,train,test}` | split to run on (default `dev`) |
| `--cases N` / `--limit N` | same selection semantics as the harness launcher |
| `--seed N` | seed of the random baseline (default 0) |
| `--only NAMES` | comma-separated controls to run (default: the four cheap ones; `all` adds `sim_normality`) |
| `--workers N` | windows in parallel (only worth it for `sim_normality`) |
| `--results-root DIR` | override the output folder (default `results_harness/control/`) |
| `--progress` | also print one line per window (off by default: a baseline answers the same thing thousands of times) |
| `--quiet` | print nothing but the comparison path |

## Results

Each baseline gets its own run folder, exactly like a harness strategy:

```
results_harness/control/
    random/20261008-171233_test/{predictions.jsonl,summary.json,run.json}
    always_no_action/20261008-171234_test/...
    always_increase_opioid/...
    ...
    human/20261008-171300_test/...
    _comparison/20261008-171233_test.csv     # one row per baseline
```

The comparison table has a row per baseline:

| Column | Meaning |
|---|---|
| `baseline` | folder name (`random`, `always_<action>`, `human`) |
| `n_windows` / `n_scored` | windows selected / windows scored |
| `agreement_result_real` | the prediction equals the action the clinician took (the accuracy the strategies are ranked by) |
| `agreement_result_1` | the prediction equals the consensus `result_1` |
| `agreement_result_plausible` | the prediction is `result_1` or `result_aux`: an acceptable answer even when it is not the action the clinician took |
| `chance_level` | `1/5 = 0.2`, only filled for `random` |
| `ci95_halfwidth` | half-width of the 95% normal-approximation interval of that proportion |
| `run_dir` | the run folder the row was read from |

`chance_level` and `ci95_halfwidth` exist so a model's accuracy can be placed against
chance: a constant baseline's accuracy is exactly its label prevalence, and the five
of them always add up to 1.

## Determinism

The random draw is generated per window from `(seed, window_id)`, never from the
position of the window in the batch. The run is therefore independent of how the
windows are processed, and rerunning the same seed reproduces it exactly.

## Implementation

- `baselines.py` — `RandomBaseline`, `ConstantBaseline` and `HumanBaseline`, all
  `Strategy` subclasses with `uses_model = False`, so they run through the same launcher
  as the strategies (`harness/run.py`) and reuse the record/summary/resume machinery.
  The run metadata records `model: null`.
- `retrieval_majority.py` — retrieval over the few-shot index plus the majority vote of
  the clinicians' real actions.
- `sim_normality.py` — the simulation policy. It reuses the PAS adapter of
  `harness/critic_verifier_sim/simulator.py` (the one place that knows how to calibrate
  and drive the simulator).
- `run_controls.py` — the launcher: runs the selected controls and writes the comparison table.

A dropped earlier rule (retrieve the neighbours of each simulated window and keep the
one whose neighbourhood holds the most `no_action`) left its runs under
`results_harness/control/sim_retrieval/`. It was abandoned because the simulator only
moves 2 of the 234 features of a window, so the five futures landed in essentially the
same neighbourhood (31 of 40 windows tied at k=8, accuracy 0.075).
