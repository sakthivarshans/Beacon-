"""
models.py — Pydantic v2 data models for the BEACON reinforcement learning environment.

BEACON (Budget Environment for Agent Control and Optimization of Needs) is a dual-scale
budget management environment supporting "household" and "corporate" simulation modes.
"""

from pydantic import BaseModel, Field
from typing import Any


class Observation(BaseModel):
    """
    Represents the observation returned to the agent at each environment step.

    Contains the full state of the current budget period, including income,
    category-level spending, savings progress, any active economic shocks,
    credit health, outstanding debt, and how many periods are left in the episode.
    """

    mode: str
    """Simulation mode — either 'household' or 'corporate'."""

    period: int
    """Current time period, starting from 1."""

    total_income: float
    """Total income available for the current period (may be seasonally adjusted)."""

    category_budgets: dict[str, float]
    """Mapping of category name to the amount allocated for that category."""

    category_spent: dict[str, float]
    """Mapping of category name to the amount already spent this period."""

    savings_balance: float
    """Current accumulated savings balance (includes compound interest)."""

    savings_goal: float
    """Target savings balance the agent should aim to reach."""

    active_shocks: list[str]
    """Names of unexpected financial events currently affecting the environment."""

    periods_remaining: int
    """Number of time periods left before the episode ends."""

    # --- New fields ---

    credit_score: float
    """CIBIL-style credit score in the range [300, 900]."""

    debt_balance: float
    """Outstanding debt principal (grows each period by the interest rate)."""

    debt_interest_rate: float
    """Per-period interest rate applied to debt_balance."""

    episode_history: list[dict[str, Any]]
    """List of per-period summary dicts recorded since episode start."""


class Action(BaseModel):
    """
    Represents the action submitted by the agent for a given time period.

    The agent specifies how much to allocate to each spending category,
    how much to contribute to savings, and optionally how much to pay toward
    outstanding debt from the available income.
    """

    allocations: dict[str, float]
    """Mapping of category name to the amount the agent allocates this period."""

    savings_contribution: float
    """Amount the agent chooses to add to savings this period."""

    debt_payment: float = Field(default=0.0)
    """Optional amount the agent pays toward outstanding debt this period."""


class Reward(BaseModel):
    """
    Represents the reward signal returned to the agent after each step.

    The total reward is a scalar in [-1.0, 1.0] composed of several sub-scores
    that reflect different aspects of budgeting performance: bill coverage,
    savings trajectory, spending efficiency, credit health, and debt management.
    Penalties are subtracted for constraint violations.
    """

    total: float
    """Final scalar reward for the step, in the range [-1.0, 1.0]."""

    bills_paid_score: float
    """Score (0–0.35) reflecting whether all essential bills were covered."""

    savings_progress_score: float
    """Score (0–0.25) reflecting progress toward the savings goal."""

    efficiency_score: float
    """Score (0 or 0.10) reflecting whether the agent stayed within income."""

    credit_health_score: float
    """Score (0–0.15) reflecting credit score health."""

    debt_management_score: float
    """Score (0–0.15) reflecting how well outstanding debt is managed."""

    shock_resilience_bonus: float
    """Retained for backward compatibility — always 0.0 in the expanded formula."""

    penalties: float
    """Cumulative penalty subtracted for constraint violations."""
