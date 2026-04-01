"""
graders.py — Evaluation graders for the BEACON reinforcement learning environment.

Each grader runs one complete, fully deterministic episode and returns a
normalised float score in [0.0, 1.0].

Graders:
    grade_task1() — Easy:   Bill Coverage          (household, 1 period)
    grade_task2() — Medium: Shock Absorption       (household, 3 periods)
    grade_task3() — Hard:   6-Month Goal Planning  (corporate, 6 periods)
    grade_task4() — Expert: Debt Recovery          (fresh_graduate scenario)

run_all_graders() runs all four, prints results, and returns a summary dict.
"""

from environment import BEACONEnvironment
from models import Action


# ---------------------------------------------------------------------------
# GRADER 1 — Easy: Bill Coverage
# ---------------------------------------------------------------------------

def grade_task1() -> float:
    """
    Easy grader: tests whether the agent can cover all essential bills in a
    single period by allocating exactly the minimum required amount to each
    essential category and directing remaining income to savings.

    Episode config:
        mode="household", total_periods=1, seed=42

    Scoring:
        score = reward.bills_paid_score / 0.4   → normalised to [0.0, 1.0]

    Returns:
        A float in [0.0, 1.0] representing bill-coverage performance.
    """
    # --- Set up environment ---------------------------------------------------
    env = BEACONEnvironment(mode="household", total_periods=1, seed=42)
    obs = env.reset()

    income      = obs.total_income
    minimums    = BEACONEnvironment._BASE_MIN_REQUIREMENTS["household"]

    # --- Build allocations: exactly the minimum required for each category ----
    # Essential categories have a non-zero minimum fraction; discretionary gets 0.
    allocations: dict[str, float] = {}
    total_bills = 0.0

    for cat, fraction in minimums.items():
        amount = fraction * income          # exact minimum amount
        allocations[cat] = amount
        total_bills += amount

    # Remaining income after meeting all bills goes into savings
    savings_contribution = max(0.0, income - total_bills)

    action = Action(
        allocations=allocations,
        savings_contribution=savings_contribution,
    )

    # --- Run the single step --------------------------------------------------
    _obs, reward, _done, _info = env.step(action)

    # --- Normalise bills_paid_score from [0.0, 0.4] → [0.0, 1.0] ------------
    score = reward.bills_paid_score / 0.4
    return round(score, 4)


# ---------------------------------------------------------------------------
# GRADER 2 — Medium: Shock Absorption
# ---------------------------------------------------------------------------

def grade_task2() -> float:
    """
    Medium grader: tests the agent's ability to maintain essential spending
    while absorbing unexpected financial shocks across 3 periods.

    Episode config:
        mode="household", total_periods=3, seed=99

    Strategy (per step):
        Step 1 — Allocate minimums everywhere; reduce discretionary to help
                 absorb the shock cost. Put any remainder into savings.
        Step 2 — Rebalance after shock: re-allocate minimums and re-check
                 shock costs; discretionary absorbs overflow again.
        Step 3 — Recovery: allocate minimums, maximise savings contribution
                 to push savings_progress_score up.

    Scoring:
        raw_avg = mean(reward.total) across 3 steps   ∈ [-1.0, 1.0]
        score   = (raw_avg + 1.0) / 2.0               ∈ [ 0.0, 1.0]

    Returns:
        A float in [0.0, 1.0] representing shock-resilience performance.
    """
    # --- Set up environment ---------------------------------------------------
    env = BEACONEnvironment(mode="household", total_periods=3, seed=99)
    obs = env.reset()

    # Force at least one shock active at the start if reset produced none
    if not env._active_shocks:
        env._active_shocks = ["medical_emergency"]
        env._shock_costs   = {"medical_emergency": 0.15 * env._total_income}

    minimums = BEACONEnvironment._BASE_MIN_REQUIREMENTS["household"]

    total_rewards: list[float] = []

    for step_num in range(1, 4):  # steps 1, 2, 3
        income     = env._total_income
        shock_cost = sum(env._shock_costs.values()) if env._active_shocks else 0.0

        # Compute baseline essential spend (sum of all minimum fractions × income)
        essential_spend = sum(
            frac * income
            for cat, frac in minimums.items()
            if frac > 0.0
        )

        # Budget headroom after essentials
        headroom = income - essential_spend

        if step_num == 1:
            # Step 1: allocate minimums; let discretionary absorb shock cost
            allocations = {cat: frac * income for cat, frac in minimums.items()}

            # Add shock cost into discretionary so it shows the agent "spent" it
            shock_absorption = min(shock_cost, max(0.0, headroom))
            allocations["discretionary"] = shock_absorption

            savings_contribution = max(0.0, headroom - shock_absorption)

        elif step_num == 2:
            # Step 2: rebalance — refresh shock costs, keep essentials solid
            allocations = {cat: frac * income for cat, frac in minimums.items()}

            current_shock = sum(env._shock_costs.values()) if env._active_shocks else 0.0
            shock_absorption = min(current_shock, max(0.0, headroom))
            allocations["discretionary"] = shock_absorption

            savings_contribution = max(0.0, headroom - shock_absorption)

        else:
            # Step 3: recovery — allocate minimums, maximise savings
            allocations = {cat: frac * income for cat, frac in minimums.items()}
            allocations["discretionary"] = 0.0  # nothing to discretionary

            # Channel all remaining headroom into savings
            savings_contribution = max(0.0, headroom)

        action = Action(
            allocations=allocations,
            savings_contribution=savings_contribution,
        )

        _obs, reward, _done, _info = env.step(action)
        total_rewards.append(reward.total)

    # --- Normalise mean reward from [-1.0, 1.0] → [0.0, 1.0] ----------------
    avg_reward = sum(total_rewards) / len(total_rewards)
    score = (avg_reward + 1.0) / 2.0
    return round(score, 4)


# ---------------------------------------------------------------------------
# GRADER 3 — Hard: 6-Month Goal Planning
# ---------------------------------------------------------------------------

def grade_task3() -> float:
    """
    Hard grader: tests the agent's ability to meet a multi-period savings goal
    while consistently covering all essential spending in a corporate setting.

    Episode config:
        mode="corporate", total_periods=6, seed=7

    Strategy (each of 6 steps):
        - Allocate exactly the minimum required to every category.
        - Contribute 15% of total_income to savings.
        - Keep total spend ≤ total_income (efficiency constraint).

    Scoring:
        goal_reached = min(savings_balance / savings_goal, 1.0)
        no_misses    = 1.0 if no essential category ever had 0 allocation
                       else 0.5
        score = (goal_reached × 0.6) + (no_misses × 0.4)

    Returns:
        A float in [0.0, 1.0] representing long-term planning performance.
    """
    # --- Set up environment ---------------------------------------------------
    env = BEACONEnvironment(mode="corporate", total_periods=6, seed=7)
    obs = env.reset()

    minimums = BEACONEnvironment._BASE_MIN_REQUIREMENTS["corporate"]

    # Track whether any essential category was ever left at zero allocation
    had_zero_essential = False

    for _step in range(6):
        income = env._total_income

        # Allocate exactly the minimum to every category
        allocations: dict[str, float] = {
            cat: frac * income for cat, frac in minimums.items()
        }

        # Contribute a fixed 15% of income to savings each period
        savings_contribution = 0.15 * income

        # Check for zero-allocation on any essential before submitting
        for cat, frac in minimums.items():
            if frac > 0.0 and allocations.get(cat, 0.0) == 0.0:
                had_zero_essential = True

        action = Action(
            allocations=allocations,
            savings_contribution=savings_contribution,
        )

        _obs, _reward, done, _info = env.step(action)

        if done:
            break

    # --- Final score calculation ----------------------------------------------
    savings_balance = env._savings_balance
    savings_goal    = env._savings_goal

    # Fraction of savings goal achieved, capped at 1.0
    goal_reached = min(savings_balance / savings_goal, 1.0) if savings_goal > 0 else 0.0

    # Full credit if every step had non-zero allocation to all essential cats
    no_misses = 0.5 if had_zero_essential else 1.0

    score = (goal_reached * 0.6) + (no_misses * 0.4)
    return round(score, 4)


# ---------------------------------------------------------------------------
# GRADER 4 — Expert: Debt Recovery
# ---------------------------------------------------------------------------

def grade_task4() -> float:
    """Expert grader: Debt Recovery — fresh graduate scenario.

    Tests whether the agent can clear a ₹1,50,000 student-loan debt while
    maintaining a healthy credit score over the course of a full episode
    in the ``"fresh_graduate"`` preset scenario.

    Episode config:
        BEACONEnvironment.load_scenario("fresh_graduate")
        (household mode, 6 periods, seed=101, starting debt=₹1,50,000,
        income=₹35,000, savings_goal=₹50,000)

    Strategy (each period):
        - Allocate exactly the minimum required fraction to every essential
          category (non-essentials receive 0).
        - Contribute 5% of income to savings.
        - Pay 20% of income toward outstanding debt.

    Scoring:
        debt_cleared = 1.0 if debt fully repaid
                       else max(0.0, 1.0 − debt_balance / 150,000)
        survived     = 1.0 if credit_score ≥ 600
                       else credit_score / 900.0
        score = (debt_cleared × 0.6) + (survived × 0.4)

    Returns:
        A float in [0.0, 1.0] representing debt-recovery performance.
    """
    env = BEACONEnvironment.load_scenario("fresh_graduate")
    obs = env.reset()

    for _ in range(env.total_periods):
        allocations: dict[str, float] = {}
        for cat, frac in env._BASE_MIN_REQUIREMENTS["household"].items():
            allocations[cat] = frac * obs.total_income if frac > 0 else 0.0

        action = Action(
            allocations=allocations,
            savings_contribution=obs.total_income * 0.05,
            debt_payment=obs.total_income * 0.20,
        )
        obs, reward, done, info = env.step(action)
        if done:
            break

    starting_debt = 150_000.0
    debt_cleared = (
        1.0 if env._debt_balance == 0
        else max(0.0, 1.0 - env._debt_balance / starting_debt)
    )
    survived = (
        1.0 if env._credit_score >= 600
        else env._credit_score / 900.0
    )
    return round(debt_cleared * 0.6 + survived * 0.4, 4)


# ---------------------------------------------------------------------------
# Aggregate runner
# ---------------------------------------------------------------------------

def run_all_graders() -> dict[str, float]:
    """
    Run all four BEACON graders, print individual scores, and return a
    summary dictionary.

    Each grader is fully deterministic — scores are identical on every run.

    Returns:
        dict with keys "task1", "task2", "task3", "task4" mapping to float
        scores in [0.0, 1.0].
    """
    task1 = grade_task1()
    task2 = grade_task2()
    task3 = grade_task3()
    task4 = grade_task4()

    print(f"Task 1 (Easy):   {task1:.2f}")
    print(f"Task 2 (Medium): {task2:.2f}")
    print(f"Task 3 (Hard):   {task3:.2f}")
    print(f"Task 4 (Expert): {task4:.2f}")

    return {
        "task1": task1,
        "task2": task2,
        "task3": task3,
        "task4": task4,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_all_graders()
