# Strategy: `structured_json`

Same call as `zero_shot` (one model call, five options, one-letter answer), but
**the whole prompt is a JSON document**: instructions and decision window travel
together in a single JSON object instead of prose.

- `prompt.json` (this folder) holds the *instruction half*: `role`, `task`,
  `patient_state_notes`, `options`, `option_meanings` and `output_format`.
- The strategy adds `patient_state` = `src.prompting.render_case(window.input)`, the
  **same rendering every strategy sends** (the shared view of `src.view`). Only the
  wrapper is JSON: the strategy varies the prompt format, never the information. The
  result is dumped with `json.dumps(..., ensure_ascii=False)` (deterministic).
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
  "patient_state": "A 69-year-old male patient (...) is undergoing ...\nCurrent vitals: ...\n\nFull monitoring record (dataset field codes):\n..."
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
