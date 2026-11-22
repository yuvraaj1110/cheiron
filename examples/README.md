# Example runs

Real request/response pairs from `python scripts/run_examples.py` against the
live ClinicalTrials.gov API. One file per question, each holding the exact
`request` sent and the full `response` returned.

See [`index.md`](index.md) for the table of questions, chart types and trial
counts. The two network files are large (0.5–0.8 MB) because every node and
edge carries its citations.

`planner_eval.json` and `planner_eval.md` are a different thing: the output of
`python scripts/eval_planners.py`, which scores both planners on 15 fixed
questions without calling the API. See [`planner_eval.md`](planner_eval.md).
