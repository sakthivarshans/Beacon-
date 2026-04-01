"""
app.py — FastAPI server for the BEACON reinforcement learning environment.

Exposes the BEACON environment as a REST API so that agents, dashboards,
and evaluation pipelines can interact with it over HTTP.

Endpoints:
    POST /reset      — initialise / reset the environment
    POST /step       — submit an action and advance one period
    GET  /state      — inspect the full current environment state
    GET  /tasks      — list all available evaluation tasks
    POST /grader     — run a specific grader and get a score
    GET  /baseline   — run all graders and return all scores
    GET  /health     — liveness check

Usage:
    python app.py
    # or
    uvicorn beacon_env.app:app --reload
"""

import os
import sys

# ---------------------------------------------------------------------------
# Ensure parent directory (d:/meta) is on the Python path so that
# environment.py, models.py, and graders.py can be imported as top-level
# modules from this subdirectory.
# ---------------------------------------------------------------------------
_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

# ---------------------------------------------------------------------------
# BEACON imports (resolved via sys.path above)
# ---------------------------------------------------------------------------
from beacon_env.environment import BEACONEnvironment          # noqa: E402
from beacon_env.models import Action                           # noqa: E402
from beacon_env.graders import (                               # noqa: E402
    grade_task1,
    grade_task2,
    grade_task3,
    run_all_graders,
)

# ---------------------------------------------------------------------------
# FastAPI imports
# ---------------------------------------------------------------------------
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="BEACON Environment API",
    description=(
        "REST API for the BEACON dual-scale budget management "
        "reinforcement learning environment."
    ),
    version="1.0.0",
)

# Allow all origins so browser-based agents and dashboards can connect freely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global environment instance
# Starts as None; created / replaced on the first POST /reset call.
# A default instance is also created at startup so GET endpoints work
# immediately without requiring a prior reset.
# ---------------------------------------------------------------------------

_env: BEACONEnvironment = BEACONEnvironment(mode="household", seed=42)


def _require_env() -> BEACONEnvironment:
    """Return the global environment, raising 503 if it is uninitialised."""
    if _env is None:
        raise HTTPException(
            status_code=503,
            detail="Environment not initialised. Call POST /reset first.",
        )
    return _env


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    """Request body for POST /reset."""
    mode:          str = Field(default="household", description="'household' or 'corporate'")
    seed:          int = Field(default=42,          description="Random seed for reproducibility")
    total_periods: int = Field(default=6,           description="Number of budget periods per episode")


class GraderRequest(BaseModel):
    """Request body for POST /grader."""
    task_id: str = Field(description="One of: 'task1', 'task2', 'task3'")


# ---------------------------------------------------------------------------
# Task catalogue (static metadata)
# ---------------------------------------------------------------------------

ACTION_SCHEMA = {
    "allocations":          "dict[str, float]",
    "savings_contribution": "float",
}

TASK_CATALOGUE = [
    {
        "task_id":     "task1",
        "name":        "Bill Coverage",
        "difficulty":  "easy",
        "description": "Allocate income to cover all essential bills in a single period.",
        "mode":        "household",
        "periods":     1,
        "seed":        42,
        "action_schema": ACTION_SCHEMA,
    },
    {
        "task_id":     "task2",
        "name":        "Shock Absorption",
        "difficulty":  "medium",
        "description": (
            "Maintain essential spending while absorbing unexpected "
            "financial shocks across 3 periods."
        ),
        "mode":        "household",
        "periods":     3,
        "seed":        99,
        "action_schema": ACTION_SCHEMA,
    },
    {
        "task_id":     "task3",
        "name":        "6-Month Goal Planning",
        "difficulty":  "hard",
        "description": (
            "Manage a corporate budget over 6 periods, covering all "
            "essential categories while reaching the savings goal."
        ),
        "mode":        "corporate",
        "periods":     6,
        "seed":        7,
        "action_schema": ACTION_SCHEMA,
    },
]

# Map task_id → grader function for quick lookup
_GRADER_MAP = {
    "task1": grade_task1,
    "task2": grade_task2,
    "task3": grade_task3,
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Liveness check")
def health():
    """
    Returns a simple status object confirming the service is running.
    """
    return {"status": "ok", "environment": "BEACON"}


@app.post("/reset", summary="Initialise or reset the environment")
def reset(body: ResetRequest = ResetRequest()):
    """
    Create a fresh BEACONEnvironment with the given parameters and call
    reset(). Returns the initial Observation as JSON.

    - **mode**: `"household"` or `"corporate"` (default: `"household"`)
    - **seed**: random seed for reproducibility (default: `42`)
    - **total_periods**: episode length (default: `6`)
    """
    global _env
    try:
        _env = BEACONEnvironment(
            mode=body.mode,
            total_periods=body.total_periods,
            seed=body.seed,
        )
        obs = _env.reset()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return obs.model_dump()


@app.post("/step", summary="Submit an action and advance one period")
def step(action: Action):
    """
    Apply the agent's Action to the current environment and advance by one
    budget period.

    Returns the resulting Observation, Reward, done flag, and info dict.

    - **allocations**: `{category: amount, ...}` — must cover all categories
    - **savings_contribution**: amount added to savings this period
    """
    env = _require_env()
    obs, reward, done, info = env.step(action)

    return {
        "observation": obs.model_dump(),
        "reward":      reward.model_dump(),
        "done":        done,
        "info":        info,
    }


@app.get("/state", summary="Inspect the current environment state")
def state():
    """
    Return the full internal state of the current environment as a plain
    dictionary. Does not advance the episode.
    """
    env = _require_env()
    return env.state()


@app.get("/tasks", summary="List all available evaluation tasks")
def tasks():
    """
    Return metadata for all three BEACON evaluation tasks, including their
    difficulty, mode, episode length, and expected action schema.
    """
    return TASK_CATALOGUE


@app.post("/grader", summary="Run a specific grader and return its score")
def grader(body: GraderRequest):
    """
    Execute the grader for the requested task and return the normalised
    score in [0.0, 1.0].

    - **task_id**: one of `"task1"`, `"task2"`, `"task3"`
    """
    grader_fn = _GRADER_MAP.get(body.task_id)
    if grader_fn is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown task_id '{body.task_id}'. "
                   f"Valid options: {list(_GRADER_MAP.keys())}",
        )

    score = grader_fn()
    return {"task_id": body.task_id, "score": score}


@app.get("/baseline", summary="Run all graders and return all scores")
def baseline():
    """
    Execute all three BEACON graders sequentially and return their scores.

    This endpoint is deterministic — scores are identical on every call.
    """
    scores = run_all_graders()
    return scores


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
