<!-- Copilot instructions for ai_assessment_1 -->
# Repository context (short)
- This repository contains a compact, rule-based income classification prototype built around `main.py`.
- Data files live under `data/` (notably `adult.data.csv` and `adult.test.csv`). The code preprocesses the UCI "Adult" dataset and evaluates a hand-crafted rule ensemble.

# What an AI coding agent should know first
- Primary entrypoint: `main.py`. Read it start-to-finish — it contains preprocessing, rule registration, evaluation and plotting in one file.
- The core model is `RuleBasedIncomeModel` (in `main.py`). It supports multiple named rulesets, per-rule stats (TP/FP/TN/FN), and activation counting. Rules are registered via `add_rule(name, func, ruleset)` and optionally via the helper `add_enum_rules`.
- Preprocessing helper: `preprocess_df(...)` normalises categorical values, buckets `education-num`, creates `_was_missing` indicators, and converts the `income` target to 0/1 using ">50K".

# Key patterns and project conventions
- Rules are pure Python callables that accept a DataFrame row and return truthy/falsy values. Many rules are added as lambdas that close over the expected value (see `add_enum_rules` usage for proper capture: lambda r, v=v: r[col] == v).
- Rules are stored as `("{ruleset}::{rulename}", func)` tuples; use the full key pattern when inspecting `rule_stats` or `rule_counts`.
- Imputers: `preprocess_df` returns fitted `num_imputer` and `cat_imputer`. When reusing preprocessors across datasets, pass these into `preprocess_df(..., fit_imputers=False)`.
- The code uses simple thresholds and equality checks as rule logic. When adding numeric comparisons, prefer safe coercion (the preprocess stage casts numeric columns) to avoid exceptions in lambdas.

# Useful files and locations to change
- `main.py` — primary file to edit for model, rules, evaluation, and plotting logic.
- `data/` — dataset CSVs. Code expects `data/adult.data.csv` by default; relative paths are used.
- `notes.txt` — project notes that indicate report structure, not runnable code.

# Developer workflows (how to run & verify)
- Run locally using Python 3.8+ (the code uses pandas, numpy, matplotlib, scikit-learn). There's no lockfile — use a virtualenv and install:
  - pip install pandas numpy matplotlib scikit-learn
- Execute the script from the repo root: `python main.py`. The script loads `data/adult.data.csv`, builds rules, prints evaluation tables to stdout, and displays matplotlib plots.
- When modifying rulesets or adding rules, run the full script to observe updated textual metrics and the plotted rule summaries.

# Typical agent tasks and examples
- Add a new named ruleset: call `model.create_ruleset('myset')` then `model.add_rule('rulename', func, ruleset='myset')`. Switch during evaluation with `model.set_active_ruleset('myset')` or pass `ruleset='myset'` into `evaluate()`.
- To compute a continuous score (ROC/AUC), reuse the commented ROC example at the bottom of `main.py` which counts rule activations per row as a surrogate score.
- When adding many enum-driven rules, prefer `add_enum_rules(model, df, col, prefix=..., name_transform=...)` to keep rule registration consistent.

# Edge-cases & gotchas (observed in code)
- Lambdas that capture loop variables must use default args (see how `add_enum_rules` does `lambda r, v=v: ...`). Avoid late-binding bugs when adding rules in loops.
- `preprocess_df` prints missing-value summaries and returns `missing_rows_df` — use this when debugging unusual dataset rows.
- `classification_report(..., zero_division=0)` is used to avoid exceptions when a class has no predictions; metric keys may be missing — access safely using .get(...) as demonstrated.

# What not to change lightly
- The single-file structure (`main.py`) intentionally keeps all logic together for the assessment. Refactors that split behavior into modules are allowed but please preserve the public function signatures used by `main()` (e.g., `preprocess_df`, `build_rules`, `RuleBasedIncomeModel`).

# If you need more details
- Ask for examples of desired changes (new rules, different evaluation, adding unit tests). If you want a refactor plan, indicate whether you prefer small incremental changes or a full module split.

---
Please review these instructions and tell me any missing conventions, preferred test commands, or CI steps to include.
