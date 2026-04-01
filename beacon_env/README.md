---
title: BEACON
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
---

# BEACON (Budget Environment for Agent Control and Optimization of Needs)

A dual-scale financial reinforcement learning environment for training and evaluating AI agents on real-world budget management tasks.

## Simulation Modes

| Mode | Target | Currency | Description |
| --- | --- | --- | --- |
| **Household** | Personal Finance | Indian Rupees (INR) | Manages household income, expenses, savings, and personal debt across multiple periods. |
| **Corporate** | Organisational Finance | Indian Rupees (INR) | Manages corporate cash flow, payroll, operations, and capital expenditures. |

## Environment Features

This environment incorporates advanced economic systems to accurately model real-world scenarios:

| Feature | Description |
| --- | --- |
| **Credit Scoring** | CIBIL-style (300-900) score that gates the probability of financial shocks. |
| **Debt Management** | Debt balances incur per-period compound interest; missed essential bills auto-fund to debt. |
| **Inflation Modelling** | Per-category inflation rates that increase minimum requirements over time. |
| **Economic Shocks** | Expanded shock catalog (e.g., medical emergency, vendor default) with direct and secondary effects. |
| **Compound Savings** | Savings accumulate per-period compound interest. |
| **Seasonal Income** | Variable income streams mapping to real-world seasonal bonuses and slumps. |

## Evaluation Tasks

| Task | Difficulty Level | Objective | Latest Baseline Score |
| --- | --- | --- | --- |
| **Task 1** | Easy | Bill Coverage | 0.88 |
| **Task 2** | Medium | Shock Absorption | 0.59 |
| **Task 3** | Hard | 6-Period Goal Planning | 0.87 |
| **Task 4** | Expert | Debt Recovery | 0.49 |

## Core API Endpoints

| Endpoint | HTTP Method | Action |
| --- | --- | --- |
| `/health` | `GET` | Health check for the environment service |
| `/tasks` | `GET` | Load list of defined tasks |
| `/scenarios` | `GET` | Retrieve pre-configured scenarios (e.g., `fresh_graduate`) |
| `/state` | `GET` | Get the full current environment state |
| `/baseline` | `GET` | Fetch baseline agent metrics |
| `/reset` | `POST` | Reset the episode and start a new one |
| `/step` | `POST` | Execute agent actions for the current period |
| `/grader` | `POST` | Run evaluation graders against the current history |

## Pre-configured Scenarios

| Scenario | Mode | Length | Description |
| --- | --- | --- | --- |
| `fresh_graduate` | Household | 6 Periods | Starts with heavy student loan debt and modest income. |
| `family_crisis` | Household | 6 Periods | Forces a severe medical emergency shock early in the episode. |
| `startup_survival` | Corporate | 4 Quarters | Features reduced income in multiple periods to simulate scarce cash flow. |
| `growth_phase` | Corporate | 4 Quarters | High income scenario with extremely aggressive savings goals. |
