"""HTTP layer. Thin on purpose: validation lives in the schemas, logic in the pipeline.

Run: uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException

from app.api.schemas import VisualizationRequest, VisualizationResponse
from app.ctgov.client import CTGovClient
from app.pipeline import run
from app.planner.llm_planner import make_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One HTTP client and one OpenAI client for the whole process.
    async with CTGovClient() as ctgov:
        app.state.ctgov = ctgov
        app.state.llm = make_client()  # None without OPENAI_API_KEY -> rule planner
        yield


app = FastAPI(
    title="ClinicalTrials.gov query-to-visualization",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "llm_planner": app.state.llm is not None}


@app.get("/v1/schema")
async def schema() -> dict:
    """The exact request and response JSON schemas, for frontend engineers."""
    return {
        "request": VisualizationRequest.model_json_schema(),
        "response": VisualizationResponse.model_json_schema(),
    }


@app.post("/v1/visualize", response_model=VisualizationResponse)
async def visualize(request: VisualizationRequest) -> VisualizationResponse:
    try:
        return await run(request, app.state.ctgov, app.state.llm)
    except httpx.HTTPStatusError as error:
        raise HTTPException(
            status_code=502,
            detail=f"ClinicalTrials.gov returned HTTP {error.response.status_code}. Try again shortly.",
        ) from error
    except httpx.HTTPError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach ClinicalTrials.gov ({type(error).__name__}). Try again shortly.",
        ) from error
