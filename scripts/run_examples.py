"""Run the example questions through the real pipeline and save each result.

Run: python scripts/run_examples.py
Uses the live ClinicalTrials.gov API, and the LLM planner when OPENAI_API_KEY
is set in .env (the rule planner otherwise). Writes examples/NN_slug.json,
each holding {"request": ..., "response": ...}, plus examples/index.md.
"""

import asyncio
import json
from pathlib import Path

from app.api.schemas import VisualizationRequest
from app.ctgov.client import CTGovClient
from app.pipeline import run
from app.planner.llm_planner import make_client

OUT = Path("examples")

EXAMPLES = [
    ("trend_pembrolizumab", {"query": "How has the number of trials for this drug changed per year since 2015?", "drug_name": "Pembrolizumab"}),
    ("phases_breast_cancer", {"query": "How are breast cancer trials distributed across phases?"}),
    ("compare_pembrolizumab_nivolumab", {"query": "Compare phases for trials involving pembrolizumab vs nivolumab"}),
    ("countries_recruiting_diabetes", {"query": "Which countries have the most recruiting trials for diabetes?"}),
    ("network_sponsors_drugs_nsclc", {"query": "Show a network of sponsors and drugs for non-small cell lung cancer trials"}),
    ("network_drug_cooccurrence_pembrolizumab", {"query": "Which drugs frequently co-occur with pembrolizumab in combination studies?", "drug_name": "pembrolizumab"}),
    ("count_recruiting_alzheimers", {"query": "How many recruiting Alzheimer's disease trials are there?"}),
]


def summary_line(n: int, slug: str, request: dict, response: dict) -> str:
    viz, meta = response["visualization"], response["meta"]
    planner = "rule" if any("rule planner" in a for a in response["notes"]["assumptions"]) else "LLM"
    return (
        f"| {n:02d} | {request['query']} | `{viz['type']}` | {viz['title']} | "
        f"{meta['studies_analyzed']:,} of {meta['total_matched']:,} | {planner} | [{slug}]({n:02d}_{slug}.json) |"
    )


async def main() -> None:
    OUT.mkdir(exist_ok=True)
    llm = make_client()
    rows = [
        "| # | Question | Viz | Title | Trials analysed | Planner | File |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    async with CTGovClient() as client:
        for n, (slug, body) in enumerate(EXAMPLES, start=1):
            response = await run(VisualizationRequest(**body), client, llm)
            payload = {"request": body, "response": response.model_dump(mode="json")}
            (OUT / f"{n:02d}_{slug}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
            rows.append(summary_line(n, slug, body, payload["response"]))
            print(rows[-1])
    (OUT / "index.md").write_text("# Example runs\n\nReal outputs from `python scripts/run_examples.py`.\n\n" + "\n".join(rows) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
