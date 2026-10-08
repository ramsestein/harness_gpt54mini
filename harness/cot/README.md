# Strategy: `cot`

**Chain-of-thought driven by a ReAct loop.** The same single call as `zero_shot`
— same clinical prose, same five options, same one-letter answer — but the prompt
makes the model work the case before committing:

1. **Three clinical axes** (the CoT part): haemodynamics (MAP, HR, their trend),
   depth of hypnosis (BIS, propofol effect-site concentration, recent hypnotic
   changes) and analgesia (remifentanil concentration, recent opioid changes,
   nociceptive signs).
2. **A ReAct loop** (Thought → Action → Observation), repeated at most four times,
   alternating what it still needs to establish with what the case actually shows.
3. A closing `Thought:` line and the final `Answer: <letter>`.

## The actions

There are no tools — the case text is all the information that exists — so each
`Action` is a *self-query* the model answers from that text:

| Action | Meaning |
|---|---|
| `LOOKUP <variable>` | current value of a monitored variable |
| `TREND <signal>` | recent direction of a signal |
| `HISTORY <drug>` | recent infusion-rate changes of a drug |
| `COMPARE <o1> vs <o2>` | weigh two of the five options against each other |

The prompt also forbids inventing numbers, requires reading the trend as well as
the current value, forbids deciding on a single variable, and asks to consider
`no_action` seriously.

## Parsing

The trace names options and action names on the way (`option c is tempting...`),
so the decision is read from the closing label: `src.parsing.parse_action` treats a
letter after `answer` as authoritative and only falls back to
`option`/`choice`/`letter` hints, then to a lone letter, then to an action name.

## Metadata per window

```json
{
  "reasoning": "Thought: the depth of hypnosis is the axis I need first.\nAction: LOOKUP bis_current\n...",
  "react": {"thoughts": 3, "actions": 2, "observations": 2},
  "parsed": true
}
```

`raw_output` keeps the whole response; `meta.reasoning` holds only the part before
the closing `Answer:` line, and `meta.react` counts the trace markers so you can
check whether the model actually followed the ReAct format.

## Run

```powershell
python harness/run.py --strategy cot --split dev --cases 2
python harness/run.py --strategy cot --split test --workers 64 --effort high
```

Results land in `results_harness/cot/<run_id>/`.

## Cost

One call per window like `zero_shot`, but the prompt is longer (the ReAct
instructions) and so is the response (the trace).
