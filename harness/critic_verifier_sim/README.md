# Strategy: `critic_verifier_sim`

A `critic_verifier` variant in which the second step **simulates** the candidates
instead of asking for a second opinion on the same question.

1. **Generator** (`generator_prompt.txt`) — proposes the **two** most appropriate
   candidate actions instead of the single best one.
2. **Simulator** (`simulator.py`) — both candidates are applied to a patient
   calibrated from the window and simulated **five minutes** ahead.
3. **Critic** (`critic_prompt.txt`) — sees the two resulting windows **blind**
   (simulated signals only) and picks the more beneficial one. That window points
   back to the action that produced it, and that action is the prediction.

## The simulator

[`python-anesthesia-simulator`](https://pypi.org/project/python-anesthesia-simulator/)
(PAS) v1.0.0, already present in the environment. It models
propofol / remifentanil / norepinephrine and outputs BIS, MAP, CO and TOL.

**Calibration.** `Patient.find_equilibrium` needs the CasADi `ipopt` plugin, which
the pip wheel does not ship, so it is *not* used. `initialized_at_given_input`
solves the equilibrium analytically with `numpy.linalg.solve`, so the calibration
here is a **bisection**: first the propofol/remifentanil rate ratio that reproduces
the window's BIS, then the norepinephrine rate that reproduces its MAP. Both land
within 0.5 of the target in ~140 ms per window (tested on real `test` windows).

**Missing baselines.** A window does not always carry those two numbers: `BIS` is
absent in 5.6% of `train`, **85.8% of `dev`** and 2.8% of `test`; `MAP` in 1.8% of
`train`, 9.9% of `dev` and 3.0% of `test`. When one is missing it is **imputed with
the ideal value** — `BIS 50` (the middle of the target band) and `MAP 70 mmHg`
(safely above the 65 threshold) — and the axis records that in
`bis_source` / `map_source` (`observed` or `imputed`), both in
`Calibration.imputed` and in the run metadata. Nothing else in the window is
touched, and the imputation only affects the operating point of the simulation:
the comparison between the two candidates is unchanged.

`dev` is therefore a poor split for this strategy and `test` is the honest one.

**Action → dose.** Multiplicative on the calibrated rate:

| Action | Drug | Factor |
|---|---|---|
| `increase_hypnotic` | propofol | ×1.5 |
| `reduce_hypnotic` | propofol | ×0.5 |
| `increase_opioid` | remifentanil | ×1.5 |
| `reduce_opioid` | remifentanil | ×0.5 |
| `no_action` | — | ×1.0 |

**What the resulting window contains.** Only the simulated signals — BIS, MAP, CO
and TOL. PAS has no HR, EtCO2, SpO2 or labs, so the critic cannot see them; the
window is *not* a dataset window with two fields swapped.

## Blind presentation

The two windows are shown as `WINDOW 1` / `WINDOW 2` in an order derived from a
hash of the `window_id`: deterministic (runs stay reproducible) but not constant,
so a position bias in the critic cannot become a systematic bias in the results.
The prompt never names the action behind a window.

## Judging criterion

The critic is given explicit targets: BIS between 40 and 60, MAP at or above
65 mmHg without hypertension, TOL close to 0. Without a criterion "more
beneficial" cannot be answered; this is the only content added beyond the
simulated signals.

## Metadata per window

```json
{
  "candidates": ["increase_opioid", "increase_hypnotic"],
  "calibration": {"u_propo": 0.0614, "u_remi": 0.0307, "u_nore": 1.6614,
                  "bis": 41.2, "map": 78.4,
                  "bis_residual": 0.0, "map_residual": 0.0,
                  "bis_source": "observed", "map_source": "imputed"},
  "simulated": [{"action": "increase_opioid", "bis": 39.1, "map": 82.5, "co": 5.1, "tol": 0.02}],
  "presented": ["increase_hypnotic", "increase_opioid"],
  "choice": 2,
  "chosen": "increase_opioid",
  "critic_parsed": true,
  "fallback": false,
  "generator_raw": "Candidate 1: c\nCandidate 2: a",
  "critic_raw": "..."
}
```

Failure modes, all recorded:

| Situation | Result |
|---|---|
| fewer than two distinct candidates | `action: no_action`, `parsed: false`, `meta.error` → the launcher retries it with `--resume` |
| the critic's answer cannot be parsed | the first *presented* window is kept, `critic_parsed: false`, `fallback: true` |
## Run

```powershell
python harness/run.py --strategy critic_verifier_sim --split dev --cases 2
python harness/run.py --strategy critic_verifier_sim --split test --workers 64 --effort high
```

Results land in `results_harness/critic_verifier_sim/<run_id>/`. `--dry-run` shows
the generator prompt (the first call).

## Cost

Two model calls per window, plus roughly one second of CPU for the two simulations
(300 steps of 1 s each). `--samples` is not used by this strategy.
