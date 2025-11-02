# This code was generated with the assistance of ChatGPT (GPT-5)

# ==============================================
# Rule-Based Income Model (All Rules Evaluated)
# ==============================================
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from datetime import datetime
import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np
import os

class RuleBasedIncomeModel:
    def __init__(self):
        # will point to the active ruleset list
        self.rules = []  
        # TP, FP, TN, FN per rule-key (ruleset::rulename)
        self.rule_stats = {}  
        # Count of how many times each rule-key fires
        self.rule_counts = {}  

        # New: support multiple named rulesets
        self.rulesets = {"default": []}
        self.active_ruleset = "default"
        self.rules = self.rulesets[self.active_ruleset]

    def create_ruleset(self, name):
        """Create an empty named ruleset (if not exists)."""
        if name not in self.rulesets:
            self.rulesets[name] = []

    def add_rule(self, name, func, ruleset=None):
        if ruleset is None:
            ruleset = self.active_ruleset
        """Register a new rule into a named ruleset (default if omitted)."""
        # ensure ruleset exists
        if ruleset not in self.rulesets:
            self.create_ruleset(ruleset)

        key = f"{ruleset}::{name}"
        # store (key, func) in the ruleset list; key used for tracking stats
        self.rulesets[ruleset].append((key, func))

        # initialize stats/counts for this rule-key
        self.rule_stats.setdefault(key, {"TP": 0, "FP": 0, "TN": 0, "FN": 0})
        self.rule_counts.setdefault(key, 0)

        # if this is the currently active ruleset, ensure self.rules points to it
        if ruleset == self.active_ruleset:
            self.rules = self.rulesets[self.active_ruleset]

    def reset_ruleset_stats(self, ruleset):
        """Reset stats and activation counts for a specific ruleset."""
        prefix = f"{ruleset}::"
        for key in list(self.rule_stats.keys()):
            if key.startswith(prefix):
                self.rule_stats[key] = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
                self.rule_counts[key] = 0

    def set_active_ruleset(self, ruleset):
        """Switch the active ruleset used during prediction/evaluation."""
        if ruleset not in self.rulesets:
            raise ValueError(f"Unknown ruleset '{ruleset}'")
        self.active_ruleset = ruleset
        self.rules = self.rulesets[self.active_ruleset]

    def predict_row(self, row, min_fires=1):
        """
        Evaluate all rules for a row and update per-rule stats.
        Predict 1 (>50K) only if at least `min_fires` rules return True.
        """
        fires = 0
        for name_key, func in self.rules:
            try:
                rule_prediction = bool(func(row))
            except Exception:
                rule_prediction = False  # handle occasional type errors gracefully

            true_label = row["income"]

            if rule_prediction:
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

        prediction = 1 if fires >= min_fires else 0
        return prediction


    def predict(self, df, min_fires=1):
        return df.apply(lambda row: self.predict_row(row, min_fires=min_fires), axis=1)


    def evaluate(self, df, label_col="income", ruleset=None, min_fires=1):
        """
        Evaluate model. If ruleset provided, evaluation uses only that named ruleset.
        If ruleset is None, uses the currently active ruleset.
        """
        # choose ruleset
        if ruleset is not None:
            self.set_active_ruleset(ruleset)

        # reset stats for the active ruleset to avoid mixing previous runs
        self.reset_ruleset_stats(self.active_ruleset)

        # ensure self.rules points at the active ruleset
        self.rules = self.rulesets[self.active_ruleset]

        # run predictions (this updates stats for the active ruleset keys)
        df["predicted"] = self.predict(df, min_fires=min_fires)
        y_true = df[label_col]
        y_pred = df["predicted"]

        acc = accuracy_score(y_true, y_pred)
        # avoid UndefinedMetricWarning when a label has no predicted samples
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        # compute overall confusion components so we can report specificity (TNR)
        TP_overall = int(((y_true == 1) & (y_pred == 1)).sum())
        FP_overall = int(((y_true == 0) & (y_pred == 1)).sum())
        TN_overall = int(((y_true == 0) & (y_pred == 0)).sum())
        FN_overall = int(((y_true == 1) & (y_pred == 0)).sum())
        overall_specificity = TN_overall / (TN_overall + FP_overall) if (TN_overall + FP_overall) > 0 else 0

        now = datetime.now()
        header = f"\n==== {now} ====\n"
        # safe access to class metrics in case a label key is absent
        p0 = report.get('0', {}).get('precision', 0)
        r0 = report.get('0', {}).get('recall', 0)
        f0 = report.get('0', {}).get('f1-score', 0)
        p1 = report.get('1', {}).get('precision', 0)
        r1 = report.get('1', {}).get('recall', 0)
        f1 = report.get('1', {}).get('f1-score', 0)

        overall = (
            f"Overall Accuracy: {acc*100:.2f}%\n"
            f"Class 0 Precision/Recall/F1: {p0*100:.2f}/{r0*100:.2f}/{f0*100:.2f}\n"
            f"Class 1 Precision/Recall/F1: {p1*100:.2f}/{r1*100:.2f}/{f1*100:.2f}\n"
            f"Overall Specificity (TNR): {overall_specificity*100:.2f}%\n"
        )

        # Compute metrics only for rules in the active ruleset
        rule_metrics = []
        prefix = f"{self.active_ruleset}::"
        for key, stats in self.rule_stats.items():
            if not key.startswith(prefix):
                continue
            name = key.split("::", 1)[1]
            TP = stats["TP"]
            FP = stats["FP"]
            TN = stats["TN"]
            FN = stats["FN"]

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
            rule_metrics.append({
                "name": name,
                "activations": self.rule_counts.get(key, 0),
                "TP": TP,
                "FP": FP,
                "TN": TN,
                "FN": FN,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "specificity": specificity
            })

        # Sort rules by F1 descending
        rule_metrics.sort(key=lambda x: x["f1"], reverse=True)

        # Build pretty table
        table_header = f"{'Rule':30} {'Act':>5} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Spec':>8}\n"
        table_header += "-" * 96 + "\n"

        table_rows = ""
        for rm in rule_metrics:
            table_rows += (
                f"{rm['name']:30} {rm['activations']:5} {rm['TP']:5} {rm['FP']:5} "
                f"{rm['TN']:5} {rm['FN']:5} {rm['precision']*100:7.2f} "
                f"{rm['recall']*100:7.2f} {rm['f1']*100:7.2f} {rm.get('specificity',0)*100:8.2f}\n"
            )

        footer = "-" * 80 + f"\n==== End {now} ====\n"

        output = header + overall + table_header + table_rows + footer

        print(output)

        #with open("results.txt", "a") as f:
        #    f.write(output)

def preprocess_df(df_raw, columns, num_cols,
                  cat_cols_excluding_target=None,
                  num_imputer=None, cat_imputer=None,
                  fit_imputers=True):
    """
    Clean and preprocess an Adult dataset DataFrame.
    Returns: (df, num_imputer, cat_imputer, missing_rows_df, was_missing)
    If fit_imputers=True, fit and return new imputers; otherwise require provided imputers.
    """
    df = df_raw.copy()
    if list(df.columns) != columns:
        df.columns = columns

    # strip whitespace
    obj_cols = df.select_dtypes(include=['object']).columns
    df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())

    # coerce numeric-like columns
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

    # report / capture missing rows BEFORE imputation
    missing_per_col = df.isna().sum()
    print("Missing values per column:\n", missing_per_col[missing_per_col > 0])
    missing_rows_df = df[df.isna().any(axis=1)].copy()

    # determine categorical columns excluding target
    if cat_cols_excluding_target is None:
        cat_cols_all = df.select_dtypes(include=['object']).columns.tolist()
        cat_cols = [c for c in cat_cols_all if c != "income"]
    else:
        cat_cols = cat_cols_excluding_target

    # record missing indicators for numeric + categorical columns
    was_missing = df[num_cols + cat_cols].isna()
    for c in was_missing.columns:
        df[f"{c}_was_missing"] = was_missing[c].astype(int)

    # small helpers (same logic as before)
    def _combine_workclass(val):
        if val in ("Federal-gov", "Local-gov", "State-gov"):
            return "Gov"
        if val in ("Self-emp-inc", "Self-emp-not-inc"):
            return "Self"
        if val in ("Never-worked", "Without-pay", "Unknown", "Missing"):
            return "Other/Unknown"
        return val

    def _combine_occupation(val):
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
        if val in ("Armed-Forces",):
            return "Other"
        return val or "Other"

    def _education_num_to_bucket(num):
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

    def _combine_marital(val):
        if val in {"Married-civ-spouse", "Married-spouse-absent", "Married-AF-spouse"}:
            return "Married"
        if val in {"Divorced", "Widowed", "Separated"}:
            return "Previously-Married"
        return val or "Other"

    def _combine_relationship(val):
        if val in {"Not-in-family", "Other-relative", "Unmarried"}:
            return "Not-in-family"
        return val or "Other"

    # imputation: fit or reuse provided imputers
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

    # apply aggregations
    df["workclass"] = df["workclass"].apply(_combine_workclass)
    df["occupation"] = df["occupation"].apply(_combine_occupation)
    df["education"] = df["education-num"].apply(_education_num_to_bucket)
    df["marital-status"] = df["marital-status"].apply(_combine_marital)
    df["relationship"] = df["relationship"].apply(_combine_relationship)
    df["native-country"] = df["native-country"].apply(lambda v: "US" if v == "United-States" else "Non-US")

    # target cleaning
    if df["income"].isna().any():
        df["income"].fillna(df["income"].mode().iloc[0], inplace=True)
    df["income"] = df["income"].astype(str).str.contains(">50K").astype(int)

    return df, num_imputer, cat_imputer, missing_rows_df, was_missing

# New helper: add enum-like categorical rules compactly
def add_enum_rules(model, df, col, prefix=None, name_transform=None):
    """
    For each unique non-null value in df[col], register a rule that checks equality.
    name_transform is an optional function to make rule names filesystem-safe.
    """
    vals = sorted(df[col].dropna().unique().tolist())
    prefix = prefix or col
    for v in vals:
        rule_name = f"{prefix}_{str(v)}"
        if name_transform:
            rule_name = name_transform(rule_name)
        # capture current v in default arg to avoid late-binding
        model.add_rule(rule_name, lambda r, v=v: r[col] == v)

def load_and_preprocess(train_path, columns, num_cols):
    raw_train = pd.read_csv(train_path, header=0, skipinitialspace=True, na_values=['?', ' ?'])
    df, num_imputer, cat_imputer, missing_rows_df, was_missing = preprocess_df(
        raw_train, columns=columns, num_cols=num_cols
    )
    return df, num_imputer, cat_imputer, missing_rows_df, was_missing


def build_rules(model, df):
    """
    Register rules in a concise, readable way using helpers.
    """
    # age rules
    model.add_rule("age_over_40", lambda r: r["age"] > 40)
    model.add_rule("age_over_30", lambda r: r["age"] > 30)

    # workclass rules (enum-driven)
    add_enum_rules(model, df, "workclass", prefix="workclass",
                   name_transform=lambda n: n.replace('-', '_').replace(' ', '_').replace('/', '_'))

    # education numeric buckets (explicit)
    model.add_rule("education_num_le_9", lambda r: r["education-num"] <= 9)
    model.add_rule("education_num_10_12", lambda r: 10 <= r["education-num"] <= 12)
    model.add_rule("education_num_13_14", lambda r: 13 <= r["education-num"] <= 14)
    model.add_rule("education_num_15", lambda r: r["education-num"] == 15)
    model.add_rule("education_num_ge_16", lambda r: r["education-num"] >= 16)

    # marital-status, occupation, relationship (enum-driven)
    add_enum_rules(model, df, "marital-status", prefix="marital",
                   name_transform=lambda n: n.replace('-', '_').replace(' ', '_'))
    add_enum_rules(model, df, "occupation", prefix="occupation",
                   name_transform=lambda n: n.replace('-', '_').replace(' ', '_'))
    add_enum_rules(model, df, "relationship", prefix="relationship",
                   name_transform=lambda n: n.replace('-', '_').replace(' ', '_'))

    # race and sex (small lists)
    races = ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
    for race in races:
        model.add_rule(f"race_{race.replace('-', '_')}", lambda r, race=race: r["race"] == race)
    for sex in ["Female", "Male"]:
        model.add_rule(f"sex_{sex}", lambda r, sex=sex: r["sex"] == sex)

    # capital and hours
    model.add_rule("capital_gain_positive", lambda r: r["capital-gain"] > 0)
    model.add_rule("capital_gain_high", lambda r: r["capital-gain"] > 5000)
    model.add_rule("capital_loss_positive", lambda r: r["capital-loss"] > 0)
    model.add_rule("capital_loss_high", lambda r: r["capital-loss"] > 2000)
    model.add_rule("hours_over_40", lambda r: r["hours-per-week"] > 40)
    model.add_rule("hours_over_50", lambda r: r["hours-per-week"] > 50)

    # native-country simplified (preprocessing maps to "US"/"Non-US")
    model.add_rule("country_US", lambda r: r["native-country"] == "US")
    model.add_rule("country_Non_US", lambda r: r["native-country"] == "Non-US")


def run_evaluation(model, df, ruleset=None, min_fires=1):
    model.evaluate(df, ruleset=ruleset, min_fires=min_fires)


def plot_rule_metrics(model, df):
    """
    Plot F1 score and activation counts for all rules in the active ruleset.
    This version does NOT filter rules by f1 or activation count — every rule is included.
    """
    rule_metrics = []
    prefix = f"{model.active_ruleset}::"
    for key, stats in model.rule_stats.items():
        if not key.startswith(prefix):
            continue
        rule_name = key.split("::", 1)[1]
        TP = stats.get("TP", 0)
        FP = stats.get("FP", 0)
        FN = stats.get("FN", 0)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        rule_metrics.append({
            "name": rule_name,
            "f1": f1,
            "activations": model.rule_counts.get(key, 0)
        })

    # sort by f1 descending for presentation
    rule_metrics.sort(key=lambda x: x["f1"], reverse=True)

    names = [rm["name"] for rm in rule_metrics]
    f1_scores = [rm["f1"] for rm in rule_metrics]
    activations = [rm["activations"] for rm in rule_metrics]

    if not names:
        print("No rules found for the active ruleset to plot.")
        return

    y_pos = np.arange(len(names))
    fig, ax1 = plt.subplots(figsize=(12, max(6, len(names)*0.3)))
    color1 = 'tab:blue'
    ax1.barh(y_pos - 0.2, f1_scores, height=0.4, color=color1, label="F1 Score")
    ax1.set_xlabel("F1 Score", color=color1)
    ax1.set_ylabel("Rule")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names)
    ax1.invert_yaxis()
    ax1.tick_params(axis='x', labelcolor=color1)

    ax2 = ax1.twiny()
    color2 = 'tab:red'
    ax2.barh(y_pos + 0.2, activations, height=0.4, color=color2, label="Activations")
    ax2.set_xlabel("Number of Activations", color=color2)
    ax2.tick_params(axis='x', labelcolor=color2)

    ax1.legend(loc='lower right')
    ax2.legend(loc='upper right')

    plt.title(f"Rule Performance for '{model.active_ruleset}' ruleset ")
    plt.tight_layout()
    plt.show()


def compute_rule_metrics(model, df=None, ruleset=None):
    """
    Return a list of per-rule metric dicts for the active (or provided) ruleset.

    Notes:
    - This reads `model.rule_stats` and `model.rule_counts` (they are populated when
      `model.evaluate(...)` or `model.predict(...)` has been run). If you haven't
      run evaluation yet, stats will be zeros.
    - If `df` is provided it's only used to compute dataset-level support if needed.
    """
    if ruleset is not None:
        model.set_active_ruleset(ruleset)

    prefix = f"{model.active_ruleset}::"
    metrics = []
    for key, stats in model.rule_stats.items():
        if not key.startswith(prefix):
            continue
        name = key.split("::", 1)[1]
        TP = stats.get("TP", 0)
        FP = stats.get("FP", 0)
        TN = stats.get("TN", 0)
        FN = stats.get("FN", 0)
        activations = model.rule_counts.get(key, 0)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        metrics.append({
            "name": name,
            "activations": activations,
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity
        })

    # preserve the same ordering as the printed table (by f1 desc)
    metrics.sort(key=lambda x: x["f1"], reverse=True)
    return metrics


def export_rule_metrics_csv(model, df=None, path="rule_metrics.csv", ruleset=None):
    """Export per-rule metrics to a CSV file. This is non-destructive."""
    metrics = compute_rule_metrics(model, df=df, ruleset=ruleset)
    if not metrics:
        print("No metrics to export (make sure you've run model.evaluate(...) first).")
        return

    import csv
    fieldnames = ["name", "activations", "TP", "FP", "TN", "FN", "precision", "recall", "f1", "specificity"]
    with open(path, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in metrics:
            # ensure numeric fields are written as numbers
            row = {k: (m.get(k, 0)) for k in fieldnames}
            writer.writerow(row)

    print(f"Exported {len(metrics)} rule metrics to {path}")

def build_final_rules(model: RuleBasedIncomeModel):
    """
    Final numeric rule set using education-num thresholds for generalisation
    and interpretability. Based on exploratory data analysis of the Adult dataset.
    """

    # === socioeconomic + education/occupation rules ===
    # 13+ = Bachelors or higher
    model.add_rule("married_high_edu",
        lambda r: r["marital-status"] == "Married" and r["education-num"] >= 13)

    model.add_rule("professional_high_edu",
        lambda r: r["occupation"] == "Professional" and r["education-num"] >= 13)

    model.add_rule("high_edu_husband",
        lambda r: r["education-num"] >= 13 and r["relationship"] == "Husband")

    # 15+ roughly Prof-school or higher
    model.add_rule("married_profschool",
        lambda r: r["marital-status"] == "Married" and r["education-num"] >= 15)

    model.add_rule("doctorate_professional",
        lambda r: r["education-num"] >= 16 and r["occupation"] == "Professional")

    # relationship nuance (kept for structure correlation)
    model.add_rule("professional_husband",
        lambda r: r["occupation"] == "Professional" and r["relationship"] == "Husband")

    model.add_rule("professional_wife",
        lambda r: r["occupation"] == "Professional" and r["relationship"] in ["Wife", "Husband"])

    # === capital gain rules ===
    model.add_rule("capital_gain_high",
        lambda r: r["capital-gain"] > 5000)

    model.add_rule("capital_gain_mid",
        lambda r: 3000 <= r["capital-gain"] <= 10000)

    model.add_rule("professional_capgain_pos",
        lambda r: r["occupation"] == "Professional" and r["capital-gain"] > 0)

    model.add_rule("husband_capgain_pos",
        lambda r: r["relationship"] == "Husband" and r["capital-gain"] > 0)

    # === labour intensity & effort ===
    model.add_rule("professional_hours_over_45",
        lambda r: r["occupation"] == "Professional" and r["hours-per-week"] > 45)

    model.add_rule("hours_40plus_capital_gain_mid",
        lambda r: r["hours-per-week"] > 40 and 3000 <= r["capital-gain"] <= 10000)

    # combined high-education professional
    model.add_rule("profschool_professional",
        lambda r: r["education-num"] >= 15 and r["occupation"] == "Professional")

    model.add_rule("married_doctorate",
        lambda r: r["marital-status"] == "Married" and r["education-num"] >= 16)

def main():
    # configuration
    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]
    train_path = "data/adult.data.csv"
    num_cols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]

    # load & preprocess
    df, num_imputer, cat_imputer, missing_rows_df, was_missing = load_and_preprocess(train_path, columns, num_cols)

    # model + rules
    model = RuleBasedIncomeModel()
    """
    rs_name = auto_discover_rules(
        model,
        df,
        threshold=0.6,      # percentage of >50K rows to qualify
        min_support=0.01,   # 1% of data minimum support
        max_features=2,     # 1, 2, and 3-way combos
        ruleset_name="auto"
    )
    best = max(range(1,4), key=lambda n: model.evaluate(df, ruleset=rs_name, min_fires=n))
    print(best)
    """
    model.create_ruleset("final")
    model.set_active_ruleset("final")
    build_final_rules(model)
    run_evaluation(model, df, "final", min_fires=2)
    plot_rule_metrics(model, df)
    export_rule_metrics_csv(model, df, ruleset="final")

def auto_discover_rules(model, df, threshold=0.5, min_support=0.02,
                        max_features=3, ruleset_name="auto"):
    """
    Automatically discover (feature=value) combos of size 1..max_features
    whose share of >50K rows exceeds `threshold` and appear in at least
    `min_support` fraction of the data.
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
                    # Ensure tuple when k==1
                    vals = values if isinstance(values, tuple) else (values,)
                    parts = [f"{c}_{str(v).replace(' ', '_')}" for c, v in zip(cols, vals)]
                    rule_name = "__".join(parts)
                    # capture columns/values safely for lambda
                    model.add_rule(rule_name,
                        lambda r, cols=cols, vals=vals: all(r[c] == v for c, v in zip(cols, vals)),
                        ruleset=ruleset_name)
                    rule_count += 1

    print(f"✅ auto-discovered {rule_count} rules "
          f"(threshold={threshold*100:.0f}%, min_support={min_support*100:.1f}%) "
          f"in ruleset '{ruleset_name}'")
    return ruleset_name


if __name__ == "__main__":
    main()

# --- Commented-out ROC / AUC plotting example (enable when needed) ---
# The block below shows two options for obtaining a continuous score for ROC:
# 1) If you have probability scores (e.g., from a probabilistic classifier), use y_scores = proba[:,1]
# 2) For this rule-based model, you can use the number of rules that fire per row as a surrogate continuous score.
#
# To enable: remove the leading '#' characters from the lines you want to run.
#
# Requirements: from sklearn.metrics import roc_curve, auc
#
# Example:
# # from sklearn.metrics import roc_curve, auc
# #
# # # Option A: use probabilistic scores if available (uncomment and supply `proba`)
# # # y_true = df["income"]
# # # y_scores = proba[:, 1]  # probability for positive class
# # #
# # # Option B (rule-based score): count how many rules in the active ruleset fire for each row
# # y_true = df["income"]
# # def rule_activation_count(row):
# #     # ensure we're evaluating the active ruleset
# #     total = 0
# #     for name_key, func in model.rules:
# #         try:
# #             if func(row):
# #                 total += 1
# #         except Exception:
# #             # if rule function expects numeric types, ensure row values are correct
# #             continue
# #     return total
# #
# # y_scores = df.apply(rule_activation_count, axis=1)
# #
# # # Compute ROC curve and AUC
# # fpr, tpr, thresholds = roc_curve(y_true, y_scores)
# # roc_auc = auc(fpr, tpr)
# #
# # # Plot ROC
# # plt.figure(figsize=(8, 6))
# # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
# # plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
# # plt.xlim([0.0, 1.0])
# # plt.ylim([0.0, 1.05])
# # plt.xlabel('False Positive Rate')
# # plt.ylabel('True Positive Rate')
# # plt.title(f'ROC Curve for ruleset \"{model.active_ruleset}\"')
# # plt.legend(loc="lower right")
# # plt.grid(True)
# # plt.show()
