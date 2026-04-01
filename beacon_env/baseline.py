"""
baseline.py — Groq LLM baseline agent for the BEACON RL environment.

Runs a Llama 3 model (via Groq) as a zero-shot budget-allocation agent
against all three BEACON tasks and prints reproducible episode scores.

Usage:
    export GROQ_API_KEY="your-key-here"
    python baseline.py

Requirements:
    pip install openai
"""

import json
import os

from openai import OpenAI

from environment import BEACONEnvironment
from models import Action


# ---------------------------------------------------------------------------
# Groq client — OpenAI-compatible endpoint
# ---------------------------------------------------------------------------

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

MODEL       = "llama3-8b-8192"
TEMPERATURE = 0          # deterministic completions for reproducibility


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(obs, step_num: int) -> str:
    """
    Construct a structured natural-language prompt from the current Observation.

    The prompt instructs the LLM to return ONLY a valid JSON object with
    `allocations` and `savings_contribution` fields. No prose, no markdown.

    Args:
        obs:      The current Observation from the environment.
        step_num: 1-indexed step number within the episode (for context).

    Returns:
        A formatted prompt string.
    """
    # Format category budgets and spent as a readable table
    budget_lines = "\n".join(
        f"  {cat}: allocated={obs.category_budgets[cat]:.2f}, "
        f"spent={obs.category_spent[cat]:.2f}"
        for cat in obs.category_budgets
    )

    shocks_text = (
        ", ".join(obs.active_shocks) if obs.active_shocks else "none"
    )

    prompt = f"""You are a financial planning agent managing a {obs.mode} budget.

Current state (Period {obs.period} of {obs.period + obs.periods_remaining - 1}):
  - Periods remaining (including this one): {obs.periods_remaining}
  - Total income available this period: {obs.total_income:.2f}
  - Savings balance: {obs.savings_balance:.2f}
  - Savings goal: {obs.savings_goal:.2f}
  - Active financial shocks: {shocks_text}

Category budgets and spending so far:
{budget_lines}

Your task:
  Allocate this period's income across all categories and decide how much to save.
  The total of all allocations + savings_contribution must NOT exceed {obs.total_income:.2f}.
  Prioritise essential categories first (avoid allocating 0 to any necessary category).
  Try to make progress toward the savings goal each period.

Respond with ONLY a valid JSON object — no explanation, no markdown, no extra text:
{{
  "allocations": {{
    {", ".join(f'"{cat}": <float>' for cat in obs.category_budgets)}
  }},
  "savings_contribution": <float>
}}"""

    return prompt


# ---------------------------------------------------------------------------
# Fallback action
# ---------------------------------------------------------------------------

def _fallback_action(obs) -> Action:
    """
    Build a safe fallback Action using exact minimum required allocations.

    Used when the LLM response cannot be parsed as valid JSON. Allocates
    exactly the minimum fraction of income to each category and puts any
    remaining income into savings.

    Args:
        obs: The current Observation (provides income and mode context).

    Returns:
        A valid Action that satisfies all essential category minimums.
    """
    minimums    = BEACONEnvironment.MIN_REQUIREMENTS[obs.mode]
    income      = obs.total_income

    allocations = {cat: frac * income for cat, frac in minimums.items()}
    total_bills = sum(allocations.values())

    # Sweep remaining income into savings after covering bills
    savings_contribution = max(0.0, income - total_bills)

    return Action(
        allocations=allocations,
        savings_contribution=savings_contribution,
    )


# ---------------------------------------------------------------------------
# LLM action parser
# ---------------------------------------------------------------------------

def _parse_action(response_text: str, obs) -> Action:
    """
    Parse the LLM's JSON response into a valid Action.

    Applies two safety guards after parsing:
      1. Clamps all allocation values to non-negative floats.
      2. Scales the entire action down proportionally if total spend would
         exceed total_income, ensuring the agent never overspends.

    Falls back to minimum allocations if the response is not valid JSON.

    Args:
        response_text: Raw text returned by the LLM.
        obs:           Current Observation (used for income and fallback).

    Returns:
        A valid Action ready to pass to env.step().
    """
    try:
        # Strip surrounding whitespace/newlines before parsing
        data = json.loads(response_text.strip())

        allocations          = {
            cat: max(0.0, float(v))
            for cat, v in data["allocations"].items()
        }
        savings_contribution = max(0.0, float(data["savings_contribution"]))

        # Safety clamp: scale down if total spend exceeds income
        total_requested = sum(allocations.values()) + savings_contribution
        if total_requested > obs.total_income and total_requested > 0:
            scale = obs.total_income / total_requested
            allocations = {cat: amt * scale for cat, amt in allocations.items()}
            savings_contribution *= scale

        return Action(
            allocations=allocations,
            savings_contribution=savings_contribution,
        )

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"  [WARN] Could not parse LLM response ({type(exc).__name__}: {exc}). "
              f"Using fallback minimum allocations.")
        return _fallback_action(obs)


# ---------------------------------------------------------------------------
# Core episode runner
# ---------------------------------------------------------------------------

def run_agent_episode(mode: str, total_periods: int, seed: int) -> float:
    """
    Run a full BEACON episode with the Groq LLM agent and return the
    average reward across all periods.

    At each step the agent receives a natural-language prompt describing
    the current budget state, responds with a JSON allocation plan, and
    the environment returns a structured Reward. If the LLM produces
    unparseable output, a safe minimum-allocation fallback is used.

    Args:
        mode:          BEACON mode — "household" or "corporate".
        total_periods: Number of budget periods in the episode.
        seed:          Random seed for environment reproducibility.

    Returns:
        Mean reward.total across all completed periods (float in [-1.0, 1.0]).
    """
    # --- Initialise environment ----------------------------------------------
    env = BEACONEnvironment(mode=mode, total_periods=total_periods, seed=seed)
    obs = env.reset()

    period_rewards: list[float] = []

    system_prompt = (
        "You are a precise financial planning agent. "
        "You always respond with ONLY valid JSON — no prose, no markdown fences, "
        "no explanation. Every numeric value must be a plain float."
    )

    # --- Episode loop --------------------------------------------------------
    for step_num in range(1, total_periods + 1):
        user_prompt = _build_prompt(obs, step_num)

        # --- Query the LLM ---------------------------------------------------
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            raw_text = response.choices[0].message.content or ""
        except Exception as exc:
            print(f"  [WARN] LLM API call failed (step {step_num}): {exc}. "
                  f"Using fallback action.")
            raw_text = ""  # triggers fallback in _parse_action

        # --- Parse response into an Action -----------------------------------
        action = _parse_action(raw_text, obs)

        # --- Step the environment --------------------------------------------
        obs, reward, done, _info = env.step(action)
        period_rewards.append(reward.total)

        if done:
            break

    # --- Average reward across all periods -----------------------------------
    avg_reward = sum(period_rewards) / len(period_rewards) if period_rewards else 0.0
    return avg_reward


# ---------------------------------------------------------------------------
# Top-level baseline runner
# ---------------------------------------------------------------------------

def run_baseline() -> dict[str, float]:
    """
    Run all three BEACON tasks with the Groq LLM agent and report scores.

    Tasks:
        Task 1 — Easy:   household mode, 1 period,  seed=42
        Task 2 — Medium: household mode, 3 periods, seed=99
        Task 3 — Hard:   corporate mode, 6 periods, seed=7

    Each task returns the mean reward across all periods, printed to 2
    decimal places.

    Returns:
        dict with keys "task1", "task2", "task3" mapping to float scores.
    """
    print("Running BEACON baseline...")
    print(f"  Model : {MODEL}")
    print(f"  Temp  : {TEMPERATURE}")
    print()

    # --- Task 1: Easy — Bill Coverage (1 period, household) ------------------
    print("Task 1 (Easy — Bill Coverage)...")
    score1 = run_agent_episode(mode="household", total_periods=1, seed=42)
    print(f"Task 1: {score1:.2f}")
    print()

    # --- Task 2: Medium — Shock Absorption (3 periods, household) ------------
    print("Task 2 (Medium — Shock Absorption)...")
    score2 = run_agent_episode(mode="household", total_periods=3, seed=99)
    print(f"Task 2: {score2:.2f}")
    print()

    # --- Task 3: Hard — 6-Month Goal Planning (6 periods, corporate) ---------
    print("Task 3 (Hard — 6-Month Goal Planning)...")
    score3 = run_agent_episode(mode="corporate", total_periods=6, seed=7)
    print(f"Task 3: {score3:.2f}")
    print()

    return {
        "task1": score1,
        "task2": score2,
        "task3": score3,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_baseline()
