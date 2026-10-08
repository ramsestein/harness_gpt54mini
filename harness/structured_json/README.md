# Strategy: `structured_json`

Same call as `zero_shot` (one model call, five options, one-letter answer), but
**the whole prompt is a JSON document**: instructions and decision window travel
together in a single JSON object instead of prose.

- `prompt.json` (this folder) holds the *instruction half*: `role`, `task`,
  `patient_state_notes`, `options`, `option_meanings` and `output_format`.
- The strategy adds `patient_state` = the sanitised window, so the window is a
  **nested JSON object**, not a string, and dumps the result with
  `json.dumps(..., ensure_ascii=False)` (sorted by insertion order, deterministic).
- `indent` can be set to pretty-print the prompt (more tokens, easier to read).
- Output parsing: identical to `zero_shot` (`src.parsing.parse_action`).
- The system message is the same as `zero_shot`; only the user message differs.

The user message therefore looks like:

```json
{
  "role": "expert anesthesiologist in the middle of a real surgical case",
  "task": "Based ONLY on 'patient_state' below, decide the single best next anesthetic adjustment.",
  "options": { "a": "increase_hypnotic", "b": "reduce_hypnotic", "c": "increase_opioid",
               "d": "reduce_opioid", "e": "no_action" },
  "output_format": "Respond with exactly one character: the letter of your choice (a, b, c, d, or e). Output nothing else.",
  "patient_state": { "map_current": 77.0, "bis_trend": "falling", "vitals_json": { "...": "..." } }
}
```

## Run

```powershell
python harness/run.py --strategy structured_json --split dev --cases 2
```

Results land in `results_harness/structured_json/<run_id>/`.

## Cost

The JSON window is ~10.5k characters (~2.6k tokens) against ~0.9k characters
(~0.2k tokens) for the prose version: roughly **11x more input tokens** per window.
