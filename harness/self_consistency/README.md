# Strategy: `self_consistency`

`zero_shot`, sampled **N times**, with the answers aggregated by **majority vote**.
That is the only difference from `zero_shot`: the prompt, the five options, the
single-letter answer and the parsing are identical.

- Prompt: it deliberately reads `../zero_shot/prompt.txt`, so the two strategies
  can never drift apart. A test asserts both send byte-identical messages.
- `n_samples` (default 5) calls per window; `temperature` defaults to **1.0**
  because sampling needs stochasticity. `gpt-5.4-mini` rejects any non-default
  temperature, so it is dropped (see `meta.dropped_params`) and the diversity comes
  from the model itself.
- Ties are broken by `dataset.ACTIONS` order, so a run is deterministic.
- Samples that do not parse do not vote. If *no* sample parses, the prediction
  falls back to `no_action` with `meta.parsed = false`.

## Metadata per window

```json
{
  "n_samples": 5,
  "parsed_samples": 5,
  "votes": { "reduce_opioid": 3, "no_action": 2 },
  "agreement": 0.6,
  "samples": ["d", "d", "d", "e", "e"],
  "latencies_s": [1.2, 1.1, 1.3, 1.0, 1.1],
  "total_latency_s": 5.7,
  "dropped_params": ["temperature"]
}
```

## Run

```powershell
python harness/run.py --strategy self_consistency --split dev --cases 2 --samples 5 --workers 8
```

Results land in `results_harness/self_consistency/<run_id>/`.

## Cost

**N times** the model calls of `zero_shot` (5x by default).
