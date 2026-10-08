# Strategy: `critic_verifier`

Two calls per window: **generate**, then **critically review** the proposal.

1. **Generator** (`build_generator_messages`) — the *same* prompt as `zero_shot`.
   It deliberately reads `../zero_shot/prompt.txt`, so the proposal is directly
   comparable with the zero-shot baseline. A test asserts the first call sends
   byte-identical messages.
2. **Critic** (`critic_prompt.txt`, this folder) — receives the same window plus
   the proposed action *by name* and either confirms it or replaces it, again with
   a single letter. `.replace("{proposal}", ...)` fills the proposal in.

The critic has the final say:

| Situation | Final action | `meta.parsed` |
|---|---|---|
| critic parses | the critic's action | `true` |
| critic does not parse, generator does | the proposal (unchanged) | `true` |
| neither parses | `no_action` | `false` |

A generator answer that does not parse is passed to the critic as
`none (the first answer could not be parsed)` — the critic can still decide.

## Metadata per window

```json
{
  "proposal": "reduce_opioid",
  "final": "no_action",
  "changed": true,
  "generator_parsed": true,
  "critic_parsed": true,
  "generator_raw": "d",
  "critic_raw": "e",
  "latencies_s": [1.2, 1.4],
  "total_latency_s": 2.6,
  "dropped_params": ["temperature"]
}
```

`raw_output` is the critic's text (the final decision); `generator_raw` keeps the
proposal so nothing is lost.

## Run

```powershell
python harness/run.py --strategy critic_verifier --split dev --cases 2 --workers 8
```

Results land in `results_harness/critic_verifier/<run_id>/`.

`--dry-run` shows only the **generator** prompt (the first call).

## Cost

**2 model calls per window** (double `zero_shot`).
