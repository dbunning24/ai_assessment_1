# ==============================================
# Rule-Based Income Model (Deterministic Rules)
# ==============================================
# This code was generated with the assistance of ChatGPT (GPT-5)

from __future__ import annotations

from datetime import datetime
from itertools import combinations
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
import os

# ========================
# 1) Rule Engine
# ========================

RuleFunc = Callable[[pd.Series], bool]


class RuleBasedIncomeModel:
    """
    Minimal, transparent rule engine:
    - Multiple named rulesets (for A/B testing).
    - Per-rule TP/FP/TN/FN stats gathered during evaluation.
    - Prediction = 1 (>50K) iff at least `min_fires` rules return True.
    """

    def __init__(self) -> None:
        self.rulesets: Dict[str, List[Tuple[str, RuleFunc]]] = {"default": []}
        self.active_ruleset: str = "default"
        self.rules: List[Tuple[str, RuleFunc]] = self.rulesets[self.active_ruleset]

        # per-rule stats and activation counts
        self.rule_stats: Dict[str, Dict[str, int]] = {}
        self.rule_counts: Dict[str, int] = {}

    # ---- ruleset management ----
    def create_ruleset(self, name: str) -> None:
        if name not in self.rulesets:
            self.rulesets[name] = []

    def set_active_ruleset(self, ruleset: str) -> None:
        if ruleset not in self.rulesets:
            raise ValueError(f"Unknown ruleset '{ruleset}'")
        self.active_ruleset = ruleset
        self.rules = self.rulesets[self.active_ruleset]

    def add_rule(self, name: str, func: RuleFunc, ruleset: str | None = None) -> None:
        """Register a new rule in a named ruleset (default = active)."""
        ruleset = ruleset or self.active_ruleset
        if ruleset not in self.rulesets:
            self.create_ruleset(ruleset)

        key = f"{ruleset}::{name}"
        self.rulesets[ruleset].append((key, func))
        self.rule_stats.setdefault(key, {"TP": 0, "FP": 0, "TN": 0, "FN": 0})
        self.rule_counts.setdefault(key, 0)

        if ruleset == self.active_ruleset:
            self.rules = self.rulesets[self.active_ruleset]

    def reset_ruleset_stats(self, ruleset: str) -> None:
        prefix = f"{ruleset}::"
        for key in list(self.rule_stats.keys()):
            if key.startswith(prefix):
                self.rule_stats[key] = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
                self.rule_counts[key] = 0

    # ---- prediction + evaluation ----
    def predict_row(self, row: pd.Series, min_fires: int = 1) -> int:
        """
        Evaluate all rules on a single row. Predict 1 if at least `min_fires` rules return True.
        Also updates per-rule TP/FP/TN/FN and activation counts.
        """
        fires = 0
        true_label = row["income"]

        for name_key, func in self.rules:
            try:
                fired = bool(func(row))
            except Exception:
                fired = False  # guard against occasional type errors

            if fired:
                self.rule_counts[name_key] = self.rule_counts.get(name_key, 0) + 1
                fires += 1
                if true_label == 1:
                    self.rule_stats[name_key]["TP"] += 1
                else:
                    self.rule_stats[name_key]["FP"] += 1
            else:
                if true_label == 1:
                    self.rule_stats[name_key]["FN"] += 1
                else:
                    self.rule_stats[name_key]["TN"] += 1

        return 1 if fires >= min_fires else 0

    def predict(self, df: pd.DataFrame, min_fires: int = 1) -> pd.Series:
        return df.apply(lambda row: self.predict_row(row, min_fires=min_fires), axis=1)

    def evaluate(self, df: pd.DataFrame, label_col: str = "income",
                 ruleset: str | None = None, min_fires: int = 1) -> None:
        """
        Evaluate current (or provided) ruleset against df.
        Prints overall metrics and a per-rule table sorted by F1.
        """
        if ruleset is not None:
            self.set_active_ruleset(ruleset)

        self.reset_ruleset_stats(self.active_ruleset)
        self.rules = self.rulesets[self.active_ruleset]

        df = df.copy()
        df["predicted"] = self.predict(df, min_fires=min_fires)
        y_true = df[label_col]
        y_pred = df["predicted"]

        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        TP = int(((y_true == 1) & (y_pred == 1)).sum())
        FP = int(((y_true == 0) & (y_pred == 1)).sum())
        TN = int(((y_true == 0) & (y_pred == 0)).sum())
        FN = int(((y_true == 1) & (y_pred == 0)).sum())
        specificity = TN / (TN + FP) if (TN + FP) else 0.0

        p0 = report.get('0', {}).get('precision', 0.0)
        r0 = report.get('0', {}).get('recall', 0.0)
        f0 = report.get('0', {}).get('f1-score', 0.0)
        p1 = report.get('1', {}).get('precision', 0.0)
        r1 = report.get('1', {}).get('recall', 0.0)
        f1 = report.get('1', {}).get('f1-score', 0.0)

        now = datetime.now()
        header = f"\n==== {now} ====\n"
        overall = (
            f"Overall Accuracy: {acc*100:.2f}%\n"
            f"Class 0 Precision/Recall/F1: {p0*100:.2f}/{r0*100:.2f}/{f0*100:.2f}\n"
            f"Class 1 Precision/Recall/F1: {p1*100:.2f}/{r1*100:.2f}/{f1*100:.2f}\n"
            f"Overall Specificity (TNR): {specificity*100:.2f}%\n"
        )

        # per-rule metrics (active ruleset only)
        rule_metrics: List[Dict[str, float]] = []
        prefix = f"{self.active_ruleset}::"
        for key, stats in self.rule_stats.items():
            if not key.startswith(prefix):
                continue
            name = key.split("::", 1)[1]
            TP_r, FP_r, TN_r, FN_r = (stats["TP"], stats["FP"], stats["TN"], stats["FN"])
            prec = TP_r / (TP_r + FP_r) if (TP_r + FP_r) else 0.0
            rec = TP_r / (TP_r + FN_r) if (TP_r + FN_r) else 0.0
            f1_r = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            spec = TN_r / (TN_r + FP_r) if (TN_r + FP_r) else 0.0

            rule_metrics.append({
                "name": name,
                "activations": self.rule_counts.get(key, 0),
                "TP": TP_r, "FP": FP_r, "TN": TN_r, "FN": FN_r,
                "precision": prec, "recall": rec, "f1": f1_r, "specificity": spec
            })

        rule_metrics.sort(key=lambda x: x["f1"], reverse=True)

        # pretty print
        table_header = (
            f"{'Rule':30} {'Act':>5} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5} "
            f"{'Prec':>7} {'Rec':>7} {'F1':>7} {'Spec':>8}\n"
            + "-" * 96 + "\n"
        )
        rows = ""
        for rm in rule_metrics:
            rows += (
                f"{rm['name']:30} {rm['activations']:5} {rm['TP']:5} {rm['FP']:5} "
                f"{rm['TN']:5} {rm['FN']:5} {rm['precision']*100:7.2f} "
                f"{rm['recall']*100:7.2f} {rm['f1']*100:7.2f} {rm['specificity']*100:8.2f}\n"
            )

        footer = "-" * 80 + f"\n==== End {now} ====\n"
        print(header + overall + table_header + rows + footer)
        return df


# ========================
# 2) Data Preprocessing
# ========================

def _combine_workclass(val: str | None) -> str:
    if val in ("Federal-gov", "Local-gov", "State-gov"):
        return "Gov"
    if val in ("Self-emp-inc", "Self-emp-not-inc"):
        return "Self"
    if val in ("Never-worked", "Without-pay", "Unknown", "Missing"):
        return "Other/Unknown"
    return val or "Other"


def _combine_occupation(val: str | None) -> str:
    prof = {"Exec-managerial", "Prof-specialty", "Tech-support"}
    clerical = {"Adm-clerical"}
    manual = {"Craft-repair", "Handlers-cleaners", "Machine-op-inspct", "Transport-moving", "Farming-fishing"}
    service = {"Other-service", "Protective-serv", "Priv-house-serv", "Sales"}
    if val in prof:
        return "Professional"
    if val in clerical:
        return "Clerical"
    if val in manual:
        return "Manual"
    if val in service:
        return "Service"
    if val in {"Armed-Forces"}:
        return "Other"
    return val or "Other"


def _education_num_to_bucket(num) -> str:
    try:
        n = int(num)
    except Exception:
        return "Other"
    if n <= 9:
        return "Elem"
    if 10 <= n <= 12:
        return "HS-grad"
    if n == 13:
        return "Bachelors"
    if n == 14:
        return "Masters"
    if n == 15:
        return "Prof-school"
    if n >= 16:
        return "Doctorate"
    return "Other"


def _combine_marital(val: str | None) -> str:
    if val in {"Married-civ-spouse", "Married-spouse-absent", "Married-AF-spouse"}:
        return "Married"
    if val in {"Divorced", "Widowed", "Separated"}:
        return "Previously-Married"
    return val or "Other"


def _combine_relationship(val: str | None) -> str:
    if val in {"Not-in-family", "Other-relative", "Unmarried"}:
        return "Not-in-family"
    return val or "Other"


def preprocess_df(df_raw: pd.DataFrame, columns: List[str], num_cols: List[str],
                  cat_cols_excluding_target: List[str] | None = None,
                  num_imputer: SimpleImputer | None = None,
                  cat_imputer: SimpleImputer | None = None,
                  fit_imputers: bool = True) -> Tuple[pd.DataFrame, SimpleImputer, SimpleImputer, pd.DataFrame, pd.DataFrame]:
    """
    Clean and preprocess Adult dataset:
    - strip whitespace, coerce numerics
    - impute numerics (median) & categoricals ("Missing")
    - collapse high-cardinality categories
    - map target to {0,1}
    Returns: (df, num_imputer, cat_imputer, missing_rows_df, was_missing_df)
    """
    df = df_raw.copy()
    if list(df.columns) != columns:
        df.columns = columns

    # strip whitespace on object cols
    obj_cols = df.select_dtypes(include=["object"]).columns
    df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())

    # numeric coercion
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # capture missing rows BEFORE imputation
    missing_per_col = df.isna().sum()
    print("Missing values per column:\n", missing_per_col[missing_per_col > 0])
    missing_rows_df = df[df.isna().any(axis=1)].copy()

    # determine categorical columns (excluding target)
    if cat_cols_excluding_target is None:
        cat_cols_all = df.select_dtypes(include=["object"]).columns.tolist()
        cat_cols = [c for c in cat_cols_all if c != "income"]
    else:
        cat_cols = cat_cols_excluding_target

    # record missing indicators for transparency
    was_missing = df[num_cols + cat_cols].isna()
    for c in was_missing.columns:
        df[f"{c}_was_missing"] = was_missing[c].astype(int)

    # imputation
    if fit_imputers:
        num_imputer = SimpleImputer(strategy="median")
        df[num_cols] = num_imputer.fit_transform(df[num_cols])
        cat_imputer = SimpleImputer(strategy="constant", fill_value="Missing")
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
    else:
        if num_imputer is None or cat_imputer is None:
            raise ValueError("When fit_imputers=False you must provide num_imputer and cat_imputer")
        df[num_cols] = num_imputer.transform(df[num_cols])
        df[cat_cols] = cat_imputer.transform(df[cat_cols])

    # apply aggregations / bucketings
    df["workclass"] = df["workclass"].apply(_combine_workclass)
    df["occupation"] = df["occupation"].apply(_combine_occupation)
    df["education"] = df["education-num"].apply(_education_num_to_bucket)
    df["marital-status"] = df["marital-status"].apply(_combine_marital)
    df["relationship"] = df["relationship"].apply(_combine_relationship)
    df["native-country"] = df["native-country"].apply(lambda v: "US" if v == "United-States" else "Non-US")

    # target → {0,1}
    if df["income"].isna().any():
        df["income"].fillna(df["income"].mode().iloc[0], inplace=True)
    df["income"] = df["income"].astype(str).str.contains(">50K").astype(int)

    return df, num_imputer, cat_imputer, missing_rows_df, was_missing


# ========================
# 3) Analytics Utilities
# ========================

def plot_rule_metrics(model: RuleBasedIncomeModel, df: pd.DataFrame, save=False) -> None:
    """
    Plot per-rule F1 vs. activation counts for the active ruleset.
    """
    prefix = f"{model.active_ruleset}::"
    metrics = []
    for key, stats in model.rule_stats.items():
        if not key.startswith(prefix):
            continue
        name = key.split("::", 1)[1]
        TP = stats.get("TP", 0)
        FP = stats.get("FP", 0)
        FN = stats.get("FN", 0)
        prec = TP / (TP + FP) if (TP + FP) else 0.0
        rec = TP / (TP + FN) if (TP + FN) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

        metrics.append({"name": name, "f1": f1, "activations": model.rule_counts.get(key, 0)})

    metrics.sort(key=lambda x: x["f1"], reverse=True)
    if not metrics:
        print("No rules to plot for active ruleset.")
        return

    names = [m["name"] for m in metrics]
    f1_scores = [m["f1"] for m in metrics]
    activations = [m["activations"] for m in metrics]

    y = np.arange(len(names))
    fig, ax1 = plt.subplots(figsize=(12, max(6, len(names) * 0.3)))
    ax1.barh(y - 0.2, f1_scores, height=0.4, label="F1 Score")
    ax1.set_xlabel("F1 Score")
    ax1.set_ylabel("Rule")
    ax1.set_yticks(y)
    ax1.set_yticklabels(names)
    ax1.invert_yaxis()

    ax2 = ax1.twiny()
    ax2.barh(y + 0.2, activations, height=0.4, color="tab:red", label="Activations")
    ax2.set_xlabel("Number of Activations")

    ax1.legend(loc="lower right")
    ax2.legend(loc="upper right")
    plt.title(f"Rule Performance — '{model.active_ruleset}'")
    plt.tight_layout()
    if save:
        _ensure_plot_dir()
        path = f"plots/rule_metrics_{model.active_ruleset}.png"
        plt.savefig(path, dpi=300)
        print(f"Feature usage plot saved to {path}")
    #plt.show()


def compute_rule_metrics(model: RuleBasedIncomeModel,
                         df: pd.DataFrame | None = None,
                         ruleset: str | None = None) -> List[Dict[str, float]]:
    """
    Return per-rule metrics for the active (or provided) ruleset.
    Requires that model.evaluate(...) or model.predict(...) has been run.
    """
    if ruleset is not None:
        model.set_active_ruleset(ruleset)

    prefix = f"{model.active_ruleset}::"
    out = []
    for key, stats in model.rule_stats.items():
        if not key.startswith(prefix):
            continue
        name = key.split("::", 1)[1]
        TP = stats.get("TP", 0)
        FP = stats.get("FP", 0)
        TN = stats.get("TN", 0)
        FN = stats.get("FN", 0)
        activations = model.rule_counts.get(key, 0)

        prec = TP / (TP + FP) if (TP + FP) else 0.0
        rec = TP / (TP + FN) if (TP + FN) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        spec = TN / (TN + FP) if (TN + FP) else 0.0

        out.append({
            "name": name,
            "activations": activations,
            "TP": TP, "FP": FP, "TN": TN, "FN": FN,
            "precision": prec, "recall": rec, "f1": f1, "specificity": spec
        })
    out.sort(key=lambda x: x["f1"], reverse=True)
    return out

def export_all_metrics(model, df, ruleset=None, path_prefix="metrics", min_fires=2):
    """
    Export both overall performance metrics and per-rule details for a ruleset.
    Robust to stale df['predicted']: computes overall metrics from a fresh local y_pred,
    then runs a clean evaluate() to populate per-rule stats that match the console.

    Outputs:
      - {path_prefix}_overall.csv
      - {path_prefix}_rules.csv
    """
    import csv
    from sklearn.metrics import classification_report, accuracy_score

    # pick ruleset
    ruleset = ruleset or model.active_ruleset
    model.set_active_ruleset(ruleset)

    # ===== 1) OVERALL METRICS (no side effects) =====
    # compute a fresh prediction vector to avoid reading stale df['predicted']
    y_true = df["income"].astype(int).values
    y_pred_local = model.predict(df, min_fires=min_fires).astype(int).values

    acc = accuracy_score(y_true, y_pred_local)
    report = classification_report(y_true, y_pred_local, output_dict=True, zero_division=0)

    # safe extraction (keys are '0' and '1')
    p0 = report.get('0', {}).get('precision', 0.0)
    r0 = report.get('0', {}).get('recall', 0.0)
    f0 = report.get('0', {}).get('f1-score', 0.0)
    p1 = report.get('1', {}).get('precision', 0.0)
    r1 = report.get('1', {}).get('recall', 0.0)
    f1 = report.get('1', {}).get('f1-score', 0.0)

    # specificity = TN / (TN + FP)
    TN = int(((y_true == 0) & (y_pred_local == 0)).sum())
    FP = int(((y_true == 0) & (y_pred_local == 1)).sum())
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    # write overall
    overall_path = f"{path_prefix}_overall.csv"
    with open(overall_path, "w", newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        w.writerow(["Accuracy", acc])
        w.writerow(["Class 0 Precision", p0])
        w.writerow(["Class 0 Recall", r0])
        w.writerow(["Class 0 F1", f0])
        w.writerow(["Class 1 Precision", p1])
        w.writerow(["Class 1 Recall", r1])
        w.writerow(["Class 1 F1", f1])
        w.writerow(["Specificity (TNR)", specificity])

    # ===== 2) PER-RULE METRICS (match console exactly) =====
    # reset cumulative stats and re-run a clean evaluation that ALSO writes df['predicted']
    model.reset_ruleset_stats(ruleset)
    model.evaluate(df, ruleset=ruleset, min_fires=min_fires)

    # now pull per-rule metrics (these match the console table)
    rule_metrics = compute_rule_metrics(model, df=df, ruleset=ruleset)

    rule_path = f"{path_prefix}_rules.csv"
    with open(rule_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name","activations","TP","FP","TN","FN",
                "precision","recall","f1","specificity"
            ]
        )
        writer.writeheader()
        for m in rule_metrics:
            writer.writerow({k: m.get(k, 0) for k in writer.fieldnames})

    print(f"✅ exported metrics for '{ruleset}' (min_fires={min_fires})")
    print(f"   • overall → {overall_path}")
    print(f"   • per-rule → {rule_path}")

# ========================
# 4) Final Curated Rules
# ========================

def build_final_rules(model: RuleBasedIncomeModel) -> None:
    """
    Final numeric rule set using education-num thresholds (generalisation)
    and interpretable socio-economic structure. Matches paper methodology.
    """

    model.add_rule("married_professional",
               lambda r: r["marital-status"] == "Married" and r["occupation"] == "Professional")

    model.add_rule("professional_husband",
                lambda r: r["occupation"] == "Professional" and r["relationship"] == "Husband")

    model.add_rule("capital_gain_high",
                lambda r: r["capital-gain"] > 5000)

    model.add_rule("married_high_edu",
                lambda r: r["marital-status"] == "Married" and r["education-num"] >= 13)

    # optional
    model.add_rule("us_capital_gain_high",
                lambda r: r["native-country"] == "US" and r["capital-gain"] > 5000)


# ========================
# 5) Optional: Rule Mining
# ========================

def auto_discover_rules(model: RuleBasedIncomeModel, df: pd.DataFrame,
                        threshold: float = 0.5, min_support: float = 0.02,
                        max_features: int = 3, ruleset_name: str = "auto") -> str:
    """
    Lightweight exploratory rule mining:
    - For 1..max_features categorical combos, compute P(>50K | combo) and support.
    - If mean >= threshold and support >= min_support, emit a rule (exact match).
    """
    model.create_ruleset(ruleset_name)
    model.set_active_ruleset(ruleset_name)

    target = "income"
    total_rows = len(df)
    cat_cols = [c for c in df.columns if df[c].dtype == "object" and c != target]

    rule_count = 0
    for k in range(1, max_features + 1):
        print(f"🔍 mining {k}-feature combos...")
        for cols in combinations(cat_cols, k):
            tab = df.groupby(list(cols))[target].agg(["mean", "count"])
            for values, row in tab.iterrows():
                support = row["count"] / total_rows
                if support >= min_support and row["mean"] > threshold:
                    vals = values if isinstance(values, tuple) else (values,)
                    rule_name = "__".join(f"{c}_{str(v).replace(' ', '_')}" for c, v in zip(cols, vals))
                    model.add_rule(
                        rule_name,
                        lambda r, cols=cols, vals=vals: all(r[c] == v for c, v in zip(cols, vals)),
                        ruleset=ruleset_name
                    )
                    rule_count += 1

    print(f"✅ auto-discovered {rule_count} rules "
          f"(threshold={threshold*100:.0f}%, min_support={min_support*100:.1f}%) "
          f"in ruleset '{ruleset_name}'")
    return ruleset_name


# ========================
# 6) Main
# ========================

def load_and_preprocess(train_path: str, columns: List[str], num_cols: List[str]):
    raw = pd.read_csv(train_path, header=0, skipinitialspace=True, na_values=["?", " ?"])
    return preprocess_df(raw, columns=columns, num_cols=num_cols)


# ===============================
#   Visualization / Diagnostics
# ===============================

def _ensure_plot_dir():
    """Ensure 'plots' directory exists for saving visualizations."""
    os.makedirs("plots", exist_ok=True)


def plot_roc_curve(model, df, save=False):
    """Plot ROC curve using rule activation counts as a score surrogate."""
    y_true = df["income"].values

    def rule_activation_count(row):
        total = 0
        for _, func in model.rules:
            try:
                if func(row):
                    total += 1
            except Exception:
                continue
        return total

    y_scores = df.apply(rule_activation_count, axis=1)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — '{model.active_ruleset}'")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()

    if save:
        _ensure_plot_dir()
        path = f"plots/roc_{model.active_ruleset}.png"
        plt.savefig(path, dpi=300)
        print(f"ROC curve saved to {path}")

    #plt.show()


def plot_precision_recall_curve(model, df, save=False):
    """Plot Precision–Recall curve using rule activations as score."""
    y_true = df["income"].values

    def rule_activation_count(row):
        total = 0
        for _, func in model.rules:
            try:
                if func(row):
                    total += 1
            except Exception:
                continue
        return total

    y_scores = df.apply(rule_activation_count, axis=1)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="darkgreen", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curve — '{model.active_ruleset}'")
    plt.grid(True)
    plt.tight_layout()

    if save:
        _ensure_plot_dir()
        path = f"plots/precision_recall_{model.active_ruleset}.png"
        plt.savefig(path, dpi=300)
        print(f"Precision-Recall curve saved to {path}")

    #plt.show()


def plot_confusion_matrix(df, y_col="income", pred_col="predicted", save=False):
    """Plot confusion matrix of actual vs predicted classes."""
    cm = confusion_matrix(df[y_col], df[pred_col])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["<=50K", ">50K"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    if save:
        _ensure_plot_dir()
        path = "plots/confusion_matrix.png"
        plt.savefig(path, dpi=300)
        print(f"Confusion matrix saved to {path}")

    #plt.show()


def plot_f1_vs_threshold(model, df, max_fires=6, save=False):
    """Plot overall F1 score vs min_fires threshold."""
    y_true = df["income"].values
    f1s = []

    for k in range(1, max_fires + 1):
        df["predicted"] = model.predict(df, min_fires=k)
        report = classification_report(y_true, df["predicted"], output_dict=True, zero_division=0)
        f1s.append(report.get("1", {}).get("f1-score", 0))

    plt.figure(figsize=(7, 5))
    plt.plot(range(1, max_fires + 1), f1s, marker="o", color="purple")
    plt.xlabel("min_fires threshold")
    plt.ylabel("F1 score")
    plt.title(f"F1 Score vs Rule Consensus — '{model.active_ruleset}'")
    plt.grid(True)
    plt.tight_layout()

    if save:
        _ensure_plot_dir()
        path = f"plots/f1_vs_minfires_{model.active_ruleset}.png"
        plt.savefig(path, dpi=300)
        print(f"F1 vs min_fires plot saved to {path}")

    #plt.show()


def plot_precision_recall_tradeoff(model, df, max_fires=6, save=False):
    """Plot how precision and recall change as min_fires increases."""
    precisions, recalls = [], []
    y_true = df["income"].values

    for k in range(1, max_fires + 1):
        df["predicted"] = model.predict(df, min_fires=k)
        report = classification_report(y_true, df["predicted"], output_dict=True, zero_division=0)
        precisions.append(report.get("1", {}).get("precision", 0))
        recalls.append(report.get("1", {}).get("recall", 0))

    plt.figure(figsize=(7, 5))
    plt.plot(range(1, max_fires + 1), precisions, marker="o", label="Precision", color="tab:blue")
    plt.plot(range(1, max_fires + 1), recalls, marker="o", label="Recall", color="tab:orange")
    plt.xlabel("min_fires threshold")
    plt.ylabel("Score")
    plt.title(f"Precision-Recall Tradeoff — '{model.active_ruleset}'")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save:
        _ensure_plot_dir()
        path = f"plots/precision_recall_tradeoff_{model.active_ruleset}.png"
        plt.savefig(path, dpi=300)
        print(f"Precision–Recall tradeoff plot saved to {path}")

    #plt.show()


def plot_rule_scatter(model, save=False):
    """Scatter plot of rule activations vs F1 score."""
    metrics = compute_rule_metrics(model)
    acts = [m["activations"] for m in metrics]
    f1s = [m["f1"] for m in metrics]
    names = [m["name"] for m in metrics]

    plt.figure(figsize=(8, 6))
    plt.scatter(acts, f1s, alpha=0.7, color="teal")
    for i, name in enumerate(names):
        if f1s[i] > 0.5 or acts[i] > np.percentile(acts, 95):
            plt.text(acts[i], f1s[i], name, fontsize=8, ha="left", va="bottom")

    plt.xlabel("Rule activations")
    plt.ylabel("F1 score")
    plt.title(f"Rule Performance Distribution - '{model.active_ruleset}'")
    plt.grid(True)
    plt.tight_layout()

    if save:
        _ensure_plot_dir()
        path = f"plots/rule_scatter_{model.active_ruleset}.png"
        plt.savefig(path, dpi=300)
        print(f"Rule scatter plot saved to {path}")

    #plt.show()


def plot_feature_usage(model, save=False):
    """Approximate 'feature importance' by counting feature mentions in rule names."""
    feature_counts = {}
    for key, _ in model.rules:
        rule_name = key.split("::", 1)[1]
        for feat in ["age", "education", "occupation", "hours", "capital", "marital", "relationship"]:
            if feat in rule_name:
                feature_counts[feat] = feature_counts.get(feat, 0) + 1

    if not feature_counts:
        print("No recognizable feature names in rule identifiers.")
        return

    feats, counts = zip(*sorted(feature_counts.items(), key=lambda x: x[1], reverse=True))
    plt.figure(figsize=(8, 5))
    plt.barh(feats, counts, color="slateblue")
    plt.xlabel("Number of rules referencing feature")
    plt.title(f"Approximate Feature Usage - '{model.active_ruleset}'")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save:
        _ensure_plot_dir()
        path = f"plots/feature_usage_{model.active_ruleset}.png"
        plt.savefig(path, dpi=300)
        print(f"Feature usage plot saved to {path}")

    #plt.show()


def main() -> None:
    # dataset schema
    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]
    train_path = "data/adult.test.csv"
    num_cols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]

    # 1) load & preprocess
    df, num_imputer, cat_imputer, missing_rows_df, was_missing = load_and_preprocess(train_path, columns, num_cols)

    # 2) model + rules
    model = RuleBasedIncomeModel()
    model.create_ruleset("final")
    model.set_active_ruleset("final")
    build_final_rules(model)
       # 3) evaluate (require at least 1 rules)
    min_fires=1
    df = model.evaluate(df, ruleset="final", min_fires=min_fires)

    # 4) visualizations
    plot_rule_metrics(model, df, save=True)
    plot_roc_curve(model, df, save=True)
    plot_precision_recall_curve(model, df, save=True)
    plot_confusion_matrix(df, save=True)
    plot_f1_vs_threshold(model, df, save=True)

    # 5) export metrics
    export_all_metrics(model, df, ruleset="final", path_prefix="final", min_fires=min_fires)

if __name__ == "__main__":
    main()
