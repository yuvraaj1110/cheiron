"""End-to-end: request -> pipeline -> VisualizationResponse, and the HTTP layer.

ClinicalTrials.gov is replaced by FakeCTGov, which serves the saved fixtures,
and no OpenAI client is passed, so the rule planner runs. No network.
"""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app import main
from app.api.schemas import VisualizationRequest, VisualizationResponse
from app.ctgov.client import FetchResult
from app.pipeline import run
from tests.helpers import load_fixture


class FakeCTGov:
    """Answers every search with one fixture; records the params it saw."""

    def __init__(self, fixture: str | None, total_count: int | None = None, error: Exception | None = None):
        self.fixture = fixture
        self.total_count = total_count
        self.error = error
        self.calls: list[dict] = []

    async def fetch(self, params: dict, max_studies: int | None = None) -> FetchResult:
        self.calls.append(params)
        if self.error:
            raise self.error
        if self.fixture is None:
            return FetchResult(studies=[], total_count=0, truncated=False)
        data = load_fixture(self.fixture)
        studies = data["studies"]
        total = self.total_count or data["totalCount"]
        return FetchResult(studies=studies, total_count=total, truncated=total > len(studies))


def raw_by_id(fixture: str) -> dict[str, str]:
    return {
        s["protocolSection"]["identificationModule"]["nctId"]: json.dumps(s, ensure_ascii=False)
        for s in load_fixture(fixture)["studies"]
    }


async def ask(query: str, fixture: str | None, **fields) -> VisualizationResponse:
    return await run(VisualizationRequest(query=query, **fields), FakeCTGov(fixture))


async def test_distribution_end_to_end():
    response = await ask("How are breast cancer trials distributed across phases?", "breast_cancer", condition="breast cancer")
    spec = response.visualization
    assert spec.type == "bar_chart"
    assert sum(row["trial_count"] for row in spec.data) == 100  # one phase bucket per trial
    assert response.meta.total_matched == 17038
    assert response.meta.truncated is True
    assert any("not a random sample" in note for note in response.notes.assumptions)
    assert any("observational" in note for note in response.notes.assumptions)
    assert set(response.meta.timings_ms) == {"plan", "fetch", "normalize", "compute", "build", "cite"}


async def test_every_citation_is_verbatim_end_to_end():
    response = await ask(
        "Which drugs frequently co-occur in combination studies with pembrolizumab?",
        "pembrolizumab",
        drug_name="pembrolizumab",
    )
    assert response.visualization.type == "network_graph"
    raw = raw_by_id("pembrolizumab")
    data = response.visualization.data
    citations = [c for item in [*data.nodes, *data.edges] for c in item.citations]
    assert citations
    for citation in citations:
        assert citation.excerpt in raw[citation.nct_id]


async def test_sparse_network_explains_empty_graph():
    # 100 NSCLC fixture trials: no sponsor-drug pair repeats, so the graph is
    # empty - and the notes must say why rather than return a blank chart silently.
    response = await ask("Show a network of sponsors and drugs for NSCLC trials", "nsclc", condition="non-small cell lung cancer")
    assert response.visualization.data.edges == []
    assert any("graph is empty" in note for note in response.notes.assumptions)


async def test_count_uses_api_total_not_downloaded_rows():
    client = FakeCTGov("pembrolizumab", total_count=3008)
    response = await run(VisualizationRequest(query="How many pembrolizumab trials are there?", drug_name="pembrolizumab"), client)
    assert response.visualization.type == "stat"
    assert response.visualization.data[0]["trial_count"] == 3008
    assert response.meta.chart_recommended is False


async def test_comparison_fetches_each_group():
    client = FakeCTGov("pembrolizumab")
    request = VisualizationRequest(query="Compare phases for pembrolizumab vs nivolumab", compare_with=["pembrolizumab", "nivolumab"])
    response = await run(request, client)
    assert response.visualization.type == "grouped_bar_chart"
    assert len(client.calls) == 2
    assert set(response.meta.filters_applied) == {"pembrolizumab", "nivolumab"}


async def test_no_matches_returns_zero_stat_with_note():
    response = await ask("Which countries run trials for zzqx?", None, drug_name="zzqx")
    assert response.visualization.type == "stat"
    assert response.visualization.data[0]["trial_count"] == 0
    assert any("No trials matched" in note for note in response.notes.assumptions)


async def test_request_fields_reach_the_api():
    client = FakeCTGov("pembrolizumab")
    await run(VisualizationRequest(query="Trials per year since 2015", drug_name="pembrolizumab", start_year=2015), client)
    params = client.calls[0]
    assert params["query.intr"] == "pembrolizumab"
    assert "2015-01-01" in params["filter.advanced"]


# ---------------------------------------------------------------- HTTP layer


@pytest.fixture
def api(monkeypatch):
    fake = FakeCTGov("breast_cancer")
    with TestClient(main.app) as client:
        monkeypatch.setattr(main.app.state, "ctgov", fake)
        monkeypatch.setattr(main.app.state, "llm", None)
        yield client, fake


def test_visualize_200(api):
    client, _ = api
    response = client.post("/v1/visualize", json={"query": "Breast cancer trials by phase", "condition": "breast cancer"})
    assert response.status_code == 200
    VisualizationResponse.model_validate(response.json())


def test_visualize_422_on_unknown_field(api):
    client, _ = api
    assert client.post("/v1/visualize", json={"query": "valid question", "drug": "x"}).status_code == 422


def test_visualize_502_when_api_fails(api):
    client, fake = api
    request = httpx.Request("GET", "https://clinicaltrials.gov/api/v2/studies")
    fake.error = httpx.HTTPStatusError("boom", request=request, response=httpx.Response(503, request=request))
    response = client.post("/v1/visualize", json={"query": "Breast cancer trials by phase"})
    assert response.status_code == 502
    assert "503" in response.json()["detail"]


def test_schema_and_health(api):
    client, _ = api
    schema = client.get("/v1/schema").json()
    assert {"request", "response"} <= set(schema)
    assert client.get("/health").json()["status"] == "ok"
