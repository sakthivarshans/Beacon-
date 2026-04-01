"""
environment.py — BEACON reinforcement learning environment.

BEACON (Budget Environment for Agent Control and Optimization of Needs) is a
dual-scale budget management environment with two operating modes:
  - "household": personal finance simulation (income in Indian Rupees)
  - "corporate":  organisational finance simulation

New systems in this version
────────────────────────────
  1.  Credit score (CIBIL-style, 300–900) that gates shock probability.
  2.  Debt balance with per-period compound interest; auto-funded missed bills.
  3.  Compound interest on savings balance.
  4.  Per-category inflation that raises minimum requirements each period.
  5.  Seasonal income variation (bonus / slow periods).
  6.  Expanded shock catalogue (12 per mode) with secondary income / cost effects.
  7.  Episode history recorded after every step.
  8.  Expanded reward formula (bills, savings, credit, debt, efficiency, penalties).
  9.  Four preset load_scenario() class-method shortcuts.
  10. state() extended with all new fields.
"""

import random
from copy import deepcopy
from models import Observation, Action, Reward


# ---------------------------------------------------------------------------
# Module-level configuration constants
# ---------------------------------------------------------------------------

MODES = ("household", "corporate")

# Spending categories available in each mode
CATEGORIES: dict[str, list[str]] = {
    "household": [
        "rent", "food", "utilities", "transport",
        "education", "medical", "discretionary",
    ],
    "corporate": [
        "payroll", "operations", "marketing", "logistics",
        "capex", "reserves", "miscellaneous",
    ],
}

# Income sampling range (inclusive) per mode — household values in Indian Rupees
INCOME_RANGE: dict[str, tuple[float, float]] = {
    "household": (30_000.0,     100_000.0),
    "corporate": (1_000_000.0, 50_000_000.0),
}

# Expanded shock catalogues — 12 per mode
SHOCKS: dict[str, list[str]] = {
    "household": [
        "medical_emergency",
        "appliance_repair",
        "school_fee_spike",
        "utility_surge",
        "job_loss_partial",
        "flood_damage",
        "vehicle_breakdown",
        "wedding_expense",
        "festival_overspend",
        "rent_hike",
        "medical_followup",
        "education_fee_hike",
    ],
    "corporate": [
        "vendor_default",
        "regulatory_fine",
        "equipment_failure",
        "key_employee_exit",
        "cyber_attack",
        "supply_chain_disruption",
        "currency_fluctuation",
        "client_payment_delay",
        "raw_material_spike",
        "legal_dispute",
        "tax_reassessment",
        "market_downturn",
    ],
}

# Each shock's direct cost is sampled in [10%, 25%] of total_income
SHOCK_COST_RANGE: tuple[float, float] = (0.10, 0.25)

# Secondary effects: shock name → ("income" | "cost_category", multiplier_delta)
# E.g. "income" with -0.15 means total_income *= (1 - 0.15)
# E.g. "cost_category" with a dict means that category minimum fraction += delta
SHOCK_SECONDARY_EFFECTS: dict[str, dict] = {
    # Household
    "medical_emergency": {"kind": "income",   "delta": -0.15},
    "job_loss_partial":  {"kind": "income",   "delta": -0.30},
    # Corporate
    "vendor_default":    {"kind": "cost_cat", "category": "operations", "delta": 0.20},
    "key_employee_exit": {"kind": "cost_cat", "category": "payroll",    "delta": 0.15},
}

# Per-category monthly / quarterly inflation multipliers
INFLATION_RATES: dict[str, dict[str, float]] = {
    "household": {
        "rent":          1.0,
        "food":          1.008,
        "utilities":     1.005,
        "transport":     1.004,
        "education":     1.006,
        "medical":       1.010,
        "discretionary": 1.003,
    },
    "corporate": {
        "payroll":       1.015,
        "operations":    1.008,
        "marketing":     1.005,
        "logistics":     1.010,
        "capex":         1.003,
        "reserves":      1.0,
        "miscellaneous": 1.005,
    },
}

# Per-period savings interest rates
SAVINGS_RATE: dict[str, float] = {
    "household": 0.005,   # 0.5% per month
    "corporate":  0.008,  # 0.8% per quarter
}

# Per-period debt interest rates
DEBT_INTEREST_RATE: dict[str, float] = {
    "household": 0.015,   # 18% p.a. → 1.5% per month
    "corporate":  0.010,  # 12% p.a. → 1% per quarter
}

# Seasonal income multipliers: {period: multiplier}
SEASONAL_INCOME: dict[str, dict[int, float]] = {
    "household": {3: 1.30, 6: 0.85},
    "corporate":  {2: 1.40, 4: 0.75},
}

# Credit score → shock probability
CREDIT_SHOCK_PROB: list[tuple[float, float]] = [
    # (min_score_exclusive, probability)  — evaluated top-down
    (700.0, 0.20),  # score > 700
    (500.0, 0.35),  # 500 < score ≤ 700
    (0.0,   0.50),  # score ≤ 500
]


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class BEACONEnvironment:
    """
    BEACON: Budget Environment for Agent Control and Optimization of Needs.

    An OpenEnv-compatible, dual-scale budget management RL environment.
    The agent manages a budget over `total_periods` steps, allocating funds
    across spending categories, growing savings, and weathering random
    financial shocks.

    Episode flow::

        obs = env.reset()
        while True:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            if done:
                break

    New in this version: credit scoring, debt management, compound interest,
    per-category inflation, seasonal income, expanded shocks with secondary
    effects, episode history, expanded reward formula, and preset scenarios.
    """

    # ------------------------------------------------------------------
    # Minimum category allocations as a *fraction* of total_income.
    # These are mutated each period by the inflation engine, so we store
    # live copies on the instance (see reset()).
    # Categories with 0.0 are non-essential (no penalty for zero spend).
    # ------------------------------------------------------------------
    _BASE_MIN_REQUIREMENTS: dict[str, dict[str, float]] = {
        "household": {
            "rent":          0.25,
            "food":          0.20,
            "utilities":     0.08,
            "transport":     0.05,
            "education":     0.10,
            "medical":       0.05,
            "discretionary": 0.00,  # non-essential
        },
        "corporate": {
            "payroll":       0.35,
            "operations":    0.20,
            "marketing":     0.05,
            "logistics":     0.08,
            "capex":         0.05,
            "reserves":      0.10,
            "miscellaneous": 0.00,  # non-essential
        },
    }

    def __init__(
        self,
        mode: str = "household",
        total_periods: int = 6,
        seed: int = 42,
        scenario: str | None = None,
    ) -> None:
        """
        Initialise the BEACON environment.

        Args:
            mode:          Simulation mode — "household" or "corporate".
            total_periods: Number of budget periods in one episode.
            seed:          Random seed for full reproducibility.
            scenario:      Optional preset scenario name. If provided the
                           environment is configured as if load_scenario()
                           had been called (after standard init).

        Raises:
            ValueError: If an unrecognised mode is supplied.
        """
        if mode not in MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Choose one of {MODES}."
            )

        self.mode          = mode
        self.total_periods = total_periods
        self.seed          = seed

        # Isolated RNG — does not pollute global random state
        self._rng = random.Random(seed)

        # ----- Core state fields (all properly initialised in reset()) -----
        self._period:           int               = 1
        self._total_income:     float             = 0.0
        self._base_income:      float             = 0.0   # pre-seasonal income
        self._savings_balance:  float             = 0.0
        self._savings_goal:     float             = 0.0
        self._category_budgets: dict[str, float]  = {}
        self._category_spent:   dict[str, float]  = {}
        self._active_shocks:    list[str]         = []
        self._shock_costs:      dict[str, float]  = {}

        # Live inflated minimum requirements (fractions of total_income)
        self._min_requirements: dict[str, float]  = {}

        # ----- New systems ------------------------------------------------
        self._credit_score:      float            = 750.0
        self._debt_balance:      float            = 0.0
        self._debt_interest_rate: float           = DEBT_INTEREST_RATE[mode]
        self._savings_rate:      float            = SAVINGS_RATE[mode]
        self._history:           list[dict]       = []
        self._current_scenario:  str | None       = scenario

        # Scenario-specific overrides applied before first reset
        self._scenario_forced_shocks:   dict[int, str]   = {}  # period → shock
        self._scenario_income_override: float | None     = None
        self._scenario_income_by_period: dict[int, float] = {}
        self._scenario_starting_debt:   float            = 0.0

        # Apply preset scenario configuration if requested
        if scenario is not None:
            self._configure_scenario(scenario)

        # Start the first episode immediately
        self.reset()

    # ------------------------------------------------------------------
    # Preset scenario factory
    # ------------------------------------------------------------------

    @classmethod
    def load_scenario(cls, scenario_name: str) -> "BEACONEnvironment":
        """
        Return a pre-configured BEACONEnvironment for the named scenario.

        Available scenarios
        -------------------
        ``"fresh_graduate"``
            Household mode, 6 periods. Starts with ₹1,50,000 student-loan
            debt and a modest ₹35,000 monthly income. Savings goal ₹50,000.

        ``"family_crisis"``
            Household mode, 6 periods. Triggers a medical_emergency shock at
            period 2. Income ₹55,000.

        ``"startup_survival"``
            Corporate mode, 4 quarters. Income ₹8,00,000 but reduced by 40%
            in periods 1 and 3 to simulate irregular cash flow.

        ``"growth_phase"``
            Corporate mode, 4 quarters. Income ₹50,00,000, savings goal
            ₹80,00,000.

        Args:
            scenario_name: One of the scenario keys listed above.

        Returns:
            A fully configured and reset BEACONEnvironment instance.

        Raises:
            ValueError: If an unknown scenario name is provided.
        """
        valid = {"fresh_graduate", "family_crisis", "startup_survival", "growth_phase"}
        if scenario_name not in valid:
            raise ValueError(
                f"Unknown scenario '{scenario_name}'. Choose from {valid}."
            )
        return cls(scenario=scenario_name)

    def _configure_scenario(self, name: str) -> None:
        """Map scenario name onto instance overrides (called before reset)."""
        if name == "fresh_graduate":
            self.mode          = "household"
            self.total_periods = 6
            self.seed          = 101
            self._scenario_starting_debt   = 150_000.0
            self._scenario_income_override = 35_000.0
            self._savings_goal_override    = 50_000.0

        elif name == "family_crisis":
            self.mode          = "household"
            self.total_periods = 6
            self.seed          = 202
            self._scenario_income_override  = 55_000.0
            self._scenario_forced_shocks    = {2: "medical_emergency"}
            self._savings_goal_override     = None

        elif name == "startup_survival":
            self.mode          = "corporate"
            self.total_periods = 4
            self.seed          = 303
            self._scenario_income_override  = 800_000.0
            # Periods 1 and 3 income reduced by 40%
            self._scenario_income_by_period = {1: 0.60, 3: 0.60}
            self._savings_goal_override     = None

        elif name == "growth_phase":
            self.mode          = "corporate"
            self.total_periods = 4
            self.seed          = 404
            self._scenario_income_override  = 5_000_000.0
            self._savings_goal_override     = 8_000_000.0

        # Re-derive per-mode constants after possibly changing self.mode
        self._debt_interest_rate = DEBT_INTEREST_RATE[self.mode]
        self._savings_rate       = SAVINGS_RATE[self.mode]

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """
        Reset the environment and begin a new episode.

        Re-seeds the internal RNG so that consecutive reset() calls always
        produce the same starting state (deterministic reproducibility).
        Randomly activates zero or one shock at episode start (probability
        gated by credit score).

        Returns:
            The initial Observation for the new episode.
        """
        # Fresh RNG from the same seed → identical episode starts every call
        self._rng = random.Random(self.seed)

        # --- Sample (or override) income ---------------------------------
        if getattr(self, "_scenario_income_override", None) is not None:
            self._base_income = float(self._scenario_income_override)
        else:
            lo, hi = INCOME_RANGE[self.mode]
            self._base_income = self._rng.uniform(lo, hi)
        self._total_income = self._base_income

        # --- Savings goal ------------------------------------------------
        if getattr(self, "_savings_goal_override", None) is not None:
            self._savings_goal = float(self._savings_goal_override)
        else:
            self._savings_goal = 0.20 * self._base_income * self.total_periods

        # --- Zero-initialise all category tracking -----------------------
        categories = CATEGORIES[self.mode]
        self._category_budgets = {cat: 0.0 for cat in categories}
        self._category_spent   = {cat: 0.0 for cat in categories}

        # --- Live minimum requirements (deep copy so inflation is fresh) -
        self._min_requirements = deepcopy(
            self._BASE_MIN_REQUIREMENTS[self.mode]
        )

        # --- Reset savings, period, and new systems ----------------------
        self._savings_balance = 0.0
        self._period          = 1
        self._credit_score    = 750.0
        self._history         = []

        # Starting debt from scenario (or zero)
        self._debt_balance = getattr(self, "_scenario_starting_debt", 0.0)

        # --- Clear shock state, then optionally seed one starting shock --
        self._active_shocks = []
        self._shock_costs   = {}
        shock_prob = self._credit_score_to_shock_prob()
        if self._rng.random() < shock_prob:
            self._activate_random_shock()

        return self._make_observation()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        """
        Execute one budget period using the agent's action.

        Steps performed (in order):

        1.  Apply seasonal income variation for the current period.
        2.  Apply compound interest to savings balance.
        3.  Apply compound interest to debt balance.
        4.  Apply scenario-forced shocks for this period (if any).
        5.  Apply category allocations → update budgets and spent amounts.
        6.  Process debt payment from action.
        7.  Add savings contribution.
        8.  Apply per-category inflation to minimum requirements.
        9.  Identify missed essential bills → add to debt, update credit.
        10. Update credit score based on period outcomes.
        11. Compute multi-component reward.
        12. Record period to episode history.
        13. Advance period counter.
        14. Randomly activate a new shock (probability gated by credit score).
        15. Determine episode termination.

        Args:
            action: The Action submitted by the agent for this period.

        Returns:
            observation:  New environment state after the step.
            reward:       Structured Reward for this period.
            done:         True when the episode has ended.
            info:         Auxiliary diagnostic data (plain dict).
        """
        # ---- 1. Seasonal income adjustment ------------------------------
        season = SEASONAL_INCOME.get(self.mode, {})
        seasonal_factor = season.get(self._period, 1.0)
        # Also apply scenario per-period income multipliers
        period_override = getattr(self, "_scenario_income_by_period", {})
        scenario_factor = period_override.get(self._period, 1.0)
        self._total_income = self._base_income * seasonal_factor * scenario_factor

        # ---- 2. Compound interest on savings ----------------------------
        if self._savings_balance > 0:
            self._savings_balance *= (1.0 + self._savings_rate)

        # ---- 3. Compound interest on debt -------------------------------
        if self._debt_balance > 0:
            self._debt_balance *= (1.0 + self._debt_interest_rate)

        # ---- 4. Forced shocks (scenario) --------------------------------
        forced_shocks = getattr(self, "_scenario_forced_shocks", {})
        if self._period in forced_shocks:
            shock_name = forced_shocks[self._period]
            if shock_name not in self._active_shocks:
                self._active_shocks.append(shock_name)
            cost_fraction = self._rng.uniform(*SHOCK_COST_RANGE)
            self._shock_costs[shock_name] = cost_fraction * self._total_income
            self._apply_shock_secondary_effect(shock_name)

        # ---- 5. Apply category allocations ------------------------------
        for cat, amount in action.allocations.items():
            if cat in self._category_budgets:
                self._category_budgets[cat] = amount
                self._category_spent[cat]   = amount

        # ---- 6. Process debt payment ------------------------------------
        debt_payment = max(0.0, getattr(action, "debt_payment", 0.0))
        if debt_payment > 0 and self._debt_balance > 0:
            self._debt_balance = max(0.0, self._debt_balance - debt_payment)

        # ---- 7. Savings contribution ------------------------------------
        self._savings_balance += action.savings_contribution

        # ---- 8. Total spending (allocations + savings + debt payment) ---
        total_spent = (
            sum(action.allocations.values())
            + action.savings_contribution
            + debt_payment
        )
        overspent = total_spent > self._total_income

        # ---- 9. Inflation → inflate minimum requirement fractions -------
        inflation = INFLATION_RATES[self.mode]
        for cat in self._min_requirements:
            self._min_requirements[cat] *= inflation.get(cat, 1.0)

        # ---- 10. Identify missed essential bills → add to debt ----------
        minimums = self._min_requirements
        essential_cats = {
            cat: frac for cat, frac in minimums.items() if frac > 0.0
        }
        missed_bills: list[str] = []
        all_essential_paid = True

        for cat, min_frac in essential_cats.items():
            min_required = min_frac * self._total_income
            allocated    = action.allocations.get(cat, 0.0)
            if allocated < 0.80 * min_required:
                # Missed — shortfall is added to debt
                shortfall = min_required - allocated
                self._debt_balance += shortfall
                missed_bills.append(cat)
                all_essential_paid = False

        # ---- 11. Update credit score ------------------------------------
        self._update_credit_score(
            all_essential_paid=all_essential_paid,
            overspent=overspent,
            savings_contributed=action.savings_contribution > 0,
        )

        # ---- 12. Compute reward -----------------------------------------
        reward = self._calculate_reward(action, total_spent, missed_bills)

        # ---- 13. Record period to history --------------------------------
        self._history.append({
            "period":          self._period,
            "reward":          reward.total,
            "credit_score":    self._credit_score,
            "debt_balance":    self._debt_balance,
            "savings_balance": self._savings_balance,
            "shocks":          list(self._active_shocks),
            "overspent":       overspent,
        })

        # ---- 14. Advance time period ------------------------------------
        self._period += 1

        # ---- 15. Randomly activate a new shock (credit-gated prob) ------
        shock_prob = self._credit_score_to_shock_prob()
        if self._rng.random() < shock_prob:
            shock = self._activate_random_shock()
            if shock:
                self._apply_shock_secondary_effect(shock)

        # ---- 16. Episode is done when no periods remain ------------------
        done = self.periods_remaining == 0

        # ---- 17. Diagnostic info dict ------------------------------------
        info: dict = {
            "period_completed":   self._period - 1,
            "total_spent":        total_spent,
            "total_income":       self._total_income,
            "overspent":          overspent,
            "missed_bills":       missed_bills,
            "active_shocks":      list(self._active_shocks),
            "shock_costs":        dict(self._shock_costs),
            "savings_balance":    self._savings_balance,
            "savings_goal":       self._savings_goal,
            "periods_remaining":  self.periods_remaining,
            "credit_score":       self._credit_score,
            "debt_balance":       self._debt_balance,
        }

        return self._make_observation(), reward, done, info

    def state(self) -> dict:
        """
        Return the complete current environment state as a plain dictionary.

        Useful for logging, checkpointing, or external serialisation without
        constructing Pydantic models.

        Returns:
            A flat dict containing all internal state fields, including the
            new credit score, debt, savings rate, inflation rates, and
            episode history.
        """
        return {
            "mode":                self.mode,
            "period":              self._period,
            "total_periods":       self.total_periods,
            "periods_remaining":   self.periods_remaining,
            "total_income":        self._total_income,
            "base_income":         self._base_income,
            "savings_balance":     self._savings_balance,
            "savings_goal":        self._savings_goal,
            "savings_rate":        self._savings_rate,
            "category_budgets":    dict(self._category_budgets),
            "category_spent":      dict(self._category_spent),
            "active_shocks":       list(self._active_shocks),
            "shock_costs":         dict(self._shock_costs),
            "seed":                self.seed,
            # --- New fields ---
            "credit_score":        self._credit_score,
            "debt_balance":        self._debt_balance,
            "debt_interest_rate":  self._debt_interest_rate,
            "inflation_rates":     dict(INFLATION_RATES[self.mode]),
            "min_requirements":    dict(self._min_requirements),
            "episode_history":     list(self._history),
            "current_scenario":    self._current_scenario,
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def periods_remaining(self) -> int:
        """Number of budget periods still remaining in the current episode."""
        return max(0, self.total_periods - self._period + 1)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_observation(self) -> Observation:
        """Build and return an Observation from the current internal state."""
        return Observation(
            mode=self.mode,
            period=self._period,
            total_income=self._total_income,
            category_budgets=dict(self._category_budgets),
            category_spent=dict(self._category_spent),
            savings_balance=self._savings_balance,
            savings_goal=self._savings_goal,
            active_shocks=list(self._active_shocks),
            periods_remaining=self.periods_remaining,
            # New fields
            credit_score=self._credit_score,
            debt_balance=self._debt_balance,
            debt_interest_rate=self._debt_interest_rate,
            episode_history=list(self._history),
        )

    # ------------------------------------------------------------------
    # Credit score helpers
    # ------------------------------------------------------------------

    def _credit_score_to_shock_prob(self) -> float:
        """Return per-period shock probability based on current credit score."""
        score = self._credit_score
        for min_score, prob in CREDIT_SHOCK_PROB:
            if score > min_score:
                return prob
        return 0.50  # fallback (score ≤ 0 edge case)

    def _update_credit_score(
        self,
        *,
        all_essential_paid: bool,
        overspent: bool,
        savings_contributed: bool,
    ) -> None:
        """Adjust credit score for the period's financial behaviour."""
        delta = 0.0

        if all_essential_paid:
            delta += 15.0
        else:
            delta -= 50.0

        if overspent:
            delta -= 25.0

        if savings_contributed:
            delta += 10.0

        self._credit_score = max(300.0, min(900.0, self._credit_score + delta))

    # ------------------------------------------------------------------
    # Shock helpers
    # ------------------------------------------------------------------

    def _activate_random_shock(self) -> str | None:
        """
        Select and activate one random shock from the mode's shock pool.

        Prefers shocks not currently active. If all shocks are already
        active, one is reselected and its cost refreshed.

        Cost is sampled uniformly in [10%, 25%] of total_income.

        Returns:
            The name of the newly activated shock, or None if pool is empty.
        """
        available = SHOCKS[self.mode]
        if not available:
            return None

        # Prefer inactive shocks for variety
        inactive = [s for s in available if s not in self._active_shocks]
        shock = self._rng.choice(inactive if inactive else available)

        cost_fraction = self._rng.uniform(*SHOCK_COST_RANGE)
        shock_cost    = cost_fraction * self._total_income

        if shock not in self._active_shocks:
            self._active_shocks.append(shock)

        # Always refresh the cost (covers re-roll of existing shocks)
        self._shock_costs[shock] = shock_cost
        return shock

    def _apply_shock_secondary_effect(self, shock: str) -> None:
        """
        Apply secondary economic effects for shocks that have them.

        Secondary effects:
            income   → self._total_income reduced by |delta| fraction.
            cost_cat → the named category's minimum requirement fraction
                       is increased by delta (higher mandatory spend).
        """
        effect = SHOCK_SECONDARY_EFFECTS.get(shock)
        if effect is None:
            return

        if effect["kind"] == "income":
            # e.g. delta = -0.15 → income *= 0.85
            self._total_income *= (1.0 + effect["delta"])
            # Also update base_income so seasonal scaling reference stays valid
            self._base_income = self._total_income

        elif effect["kind"] == "cost_cat":
            cat   = effect["category"]
            delta = effect["delta"]
            if cat in self._min_requirements:
                self._min_requirements[cat] = min(
                    1.0, self._min_requirements[cat] + delta
                )

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _calculate_reward(
        self,
        action: Action,
        total_spent: float,
        missed_bills: list[str],
    ) -> Reward:
        """
        Compute the structured Reward for the current period.

        Component breakdown
        -------------------
        bills_paid_score       ∈ [0.0, 0.35]
            Fraction of essential categories with ≥ 80% of inflated
            minimum allocation, scaled by 0.35.

        savings_progress_score ∈ [0.0, 0.25]
            (savings_balance / savings_goal) × 0.25, capped at 0.25.

        credit_health_score    ∈ [0.0, 0.15]
            (credit_score − 300) / 600 × 0.15.

        debt_management_score  ∈ [0.0, 0.15]
            Tiered by debt relative to income:
            0 debt → 0.15 | < 20% income → 0.10 |
            < 50% income → 0.05 | else → 0.0

        efficiency_score       ∈ {0.0, 0.10}
            0.10 if total_spent ≤ total_income, else 0.0.

        penalties              ∈ (−∞, 0.0]
            −0.30 per missed essential bill
            −0.20 if debt_balance > 50% of income
            −0.10 if overspent
            −0.15 if credit_score < 500

        total = sum of all, clipped to [−1.0, 1.0].

        Args:
            action:       Agent's action for this period.
            total_spent:  Total funds deployed.
            missed_bills: List of essential categories underfunded.

        Returns:
            A fully populated Reward model.
        """
        minimums = self._min_requirements
        essential_cats = {
            cat: frac for cat, frac in minimums.items() if frac > 0.0
        }
        total_essential = len(essential_cats)

        # --- bills_paid_score --- (max 0.35) ------------------------------
        categories_covered = 0
        for cat, min_frac in essential_cats.items():
            min_required = min_frac * self._total_income
            allocated    = action.allocations.get(cat, 0.0)
            if allocated >= 0.80 * min_required:
                categories_covered += 1

        bills_paid_score = (
            (categories_covered / total_essential) * 0.35
            if total_essential > 0
            else 0.35
        )

        # --- savings_progress_score --- (max 0.25) -------------------------
        if self._savings_goal > 0:
            raw = (self._savings_balance / self._savings_goal) * 0.25
            savings_progress_score = min(raw, 0.25)
        else:
            savings_progress_score = 0.0

        # --- credit_health_score --- (max 0.15) ---------------------------
        credit_health_score = ((self._credit_score - 300.0) / 600.0) * 0.15

        # --- debt_management_score --- (max 0.15) -------------------------
        if self._debt_balance == 0.0:
            debt_management_score = 0.15
        elif self._debt_balance < 0.20 * self._total_income:
            debt_management_score = 0.10
        elif self._debt_balance < 0.50 * self._total_income:
            debt_management_score = 0.05
        else:
            debt_management_score = 0.0

        # --- efficiency_score --- (0.10 or 0.0) ---------------------------
        overspent       = total_spent > self._total_income
        efficiency_score = 0.10 if not overspent else 0.0

        # --- penalties --- (negative) -------------------------------------
        penalties = 0.0

        # −0.30 per essential category underfunded
        penalties -= 0.30 * len(missed_bills)

        # −0.20 if outstanding debt exceeds 50% of income
        if self._debt_balance > 0.50 * self._total_income:
            penalties -= 0.20

        # −0.10 for overspending income this period
        if overspent:
            penalties -= 0.10

        # −0.15 if credit score is dangerously low
        if self._credit_score < 500.0:
            penalties -= 0.15

        # --- total — clipped to [−1.0, 1.0] ------------------------------
        total = (
            bills_paid_score
            + savings_progress_score
            + credit_health_score
            + debt_management_score
            + efficiency_score
            + penalties
        )
        total = max(-1.0, min(1.0, total))

        return Reward(
            total=total,
            bills_paid_score=bills_paid_score,
            savings_progress_score=savings_progress_score,
            efficiency_score=efficiency_score,
            credit_health_score=credit_health_score,
            debt_management_score=debt_management_score,
            shock_resilience_bonus=0.0,  # retained for backward compat
            penalties=penalties,
        )
