# Strategy: `rag`

`zero_shot` plus a **reference file of clinical guidance**. The call is identical: the same
prose rendering of the decision window, the same five options, the same `zero_shot` instructions
(their `prompt.txt` is the template, shared on purpose so the two runs differ in one thing
only). The addition is the whole content of `context.md`, inserted after the patient state.

## The reference material

`harness/rag/context.md` (28.6 kB) is a hand-curated set of **verbatim** excerpts from published
sources, each block attributed under its quote, with the provenance in
`harness/rag/context_sources.json`:

- VitalDB (where the corpus comes from),
- the AAGBI/SIVA 2019 TIVA guidelines (TCI, propofol time course, target concentrations,
  propofol–opioid interaction, monitoring),
- the DIPRIVAN and ULTIVA prescribing information (pharmacodynamics, dose adjustment
  intervals, elderly patients),
- NICE HTG292 (BIS targets, depth of anaesthesia),
- the ESAIC 2024 postoperative-delirium guideline (index-based monitoring),
- the POQI 2024 consensus on intraoperative arterial pressure.

Nothing is paraphrased. That is the point of the experiment: the model receives real guidance,
not a summary of it.

## Where the reference goes

```
zero_shot prompt (wording unchanged)
  ├── the patient state, rendered as clinical prose
  ├── "Reference information: published clinical guidance relevant to this decision. Use it
  │    together with the patient state above."
  └── context.md, verbatim
  ├── "Based ONLY on the information above, decide ..."   ← still holds: the reference is above
  └── the five options (a-e)
```

## Run

```powershell
python harness/run.py --strategy rag --split test --effort none --workers 64
```

`context_file=` overrides the file (a smaller excerpt, another corpus).

## Metadata per window

```json
{
  "context_file": "...\\harness\\rag\\context.md",
  "context_chars": 28590,
  "prompt_chars": 30434
}
```

## Cost

One call per window, but a **large** prompt: ~30.4k characters (~8k tokens) against ~3.3k
(~900 tokens) for `zero_shot`, i.e. roughly nine times the input tokens for the same output.
The per-window cost is in the metadata so a run can be priced after the fact.
