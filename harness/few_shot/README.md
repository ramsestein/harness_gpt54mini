# Strategy: `few_shot`

The same single call as `zero_shot`, but the prompt is preceded by the **k most
similar past decision windows** taken from the `train` split, each one labelled
with the adjustment the clinician actually made.

## Similarity

- Search space: the whole `train` split (80,933 windows).
- A window is represented by the **numeric fields of the shared view** (`src/view.py`,
  234 before the shared view): standardised
  with the train mean/std, missing values imputed with the train mean, Euclidean
  distance.
- **The labels never take part in the similarity.** `result_1`, `result_aux` and
  `result_real` live in `output`, which the index never reads. A test asserts that
  two windows with identical variables but different labels get identical vectors
  and identical distances.
- Excluded from the vector: identifiers (`pt_caseid`, `pt_subjectid`, `case_id`)
  and, in this version, everything non-numeric — the nested `*_json` series, sex,
  procedure name, monitoring type and the trend labels.
- The similarity is computed **per window**: every window gets its own neighbours.

## Leakage

- Only `train` windows can be retrieved (`retrieval.SOURCE_SPLIT`).
- A window never retrieves its own case (`exclude_same_case=True` by default).

## Index cache

Building the index takes ~45 s and writes ~175 MB under
`results_harness/few_shot/_index/`. It is built on first use and reused after that.

```powershell
python harness/few_shot/build_index.py             # prebuild / reuse the cache
python harness/few_shot/build_index.py --cases 20  # quick debug index (never cached)
python harness/few_shot/build_index.py --rebuild
```

## Run

```powershell
python harness/run.py --strategy few_shot --split dev --cases 2 --samples 3 --workers 8
```

`k` defaults to 3; from the launcher it is set with the generic `--samples N`
flag (`n_samples` is accepted as an alias for `k`).

## Metadata per window

```json
{
  "k": 3,
  "neighbors": [
    {"window_id": "case5652_w78365", "case_id": "5652", "action": "reduce_opioid", "distance": 4.118}
  ]
}
```

## Cost

One call per window. The prompt is ~3.5x the `zero_shot` prompt with k=2
(3.3k characters ≈ 820 tokens against 0.9k ≈ 230).
