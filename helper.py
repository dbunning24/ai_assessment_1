# ==============================================
# Rule-Based Income Model (Hybrid + Buckets + Weighted Voting)
# ==============================================
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from datetime import datetime
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt

class RuleBasedIncomeModel:
    def __init__(self):
        self.rulesets = {"default": []}
        self.active_ruleset = "default"
        self.rules = self.rulesets[self.active_ruleset]
        self.rule_stats = {}
        self.rule_counts = {}

    def create_ruleset(self, name):
        if name not in self.rulesets:
            self.rulesets[name] = []

    def add_rule(self, name, func, ruleset=None):
        ruleset = ruleset or self.active_ruleset
        if ruleset not in self.rulesets:
            self.create_ruleset(ruleset)
        key = f"{ruleset}::{name}"
        self.rulesets[ruleset].append((key, func))
        self.rule_stats.setdefault(key, {"TP": 0, "FP": 0, "TN": 0, "FN": 0})
        self.rule_counts.setdefault(key, 0)
        if ruleset == self.active_ruleset:
            self.rules = self.rulesets[self.active_ruleset]

    def set_active_ruleset(self, ruleset):
        if ruleset not in self.rulesets:
            raise ValueError(f"Unknown ruleset '{ruleset}'")
        self.active_ruleset = ruleset
        self.rules = self.rulesets[ruleset]

    def reset_ruleset_stats(self, ruleset):
        prefix = f"{ruleset}::"
        for key in list(self.rule_stats.keys()):
            if key.startswith(prefix):
                self.rule_stats[key] = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
                self.rule_counts[key] = 0

    # -------- Prediction modes --------
    def predict_row(self, row, min_fires=1, weighted=False, weight_threshold=1.0):
        fires = 0
        score = 0
        for name_key, func in self.rules:
            try:
                rule_prediction = bool(func(row))
            except Exception:
                rule_prediction = False
            true_label = row["income"]

            if rule_prediction:
                self.rule_counts[name_key] += 1
                fires += 1
                prec = self.rule_stats[name_key]["TP"] / max(1, self.rule_stats[name_key]["TP"] + self.rule_stats[name_key]["FP"])
                score += prec if weighted else 1
                if true_label == 1:
                    self.rule_stats[name_key]["TP"] += 1
                else:
                    self.rule_stats[name_key]["FP"] += 1
            else:
                if true_label == 1:
                    self.rule_stats[name_key]["FN"] += 1
                else:
                    self.rule_stats[name_key]["TN"] += 1

        if weighted:
            prediction = 1 if score >= weight_threshold else 0
        else:
            prediction = 1 if fires >= min_fires else 0
        return prediction

    def predict(self, df, min_fires=1, weighted=False, weight_threshold=1.0):
        return df.apply(lambda r: self.predict_row(r, min_fires=min_fires, weighted=weighted, weight_threshold=weight_threshold), axis=1)

    def predict_row_hybrid(self, row):
        fires_precise = sum(bool(f(row)) for _, f in self.rulesets.get("precise", []))
        fires_broad = sum(bool(f(row)) for _, f in self.rulesets.get("broad", []))
        
        # weighted hybrid: precise counts as 2 votes
        score = fires_precise * 2 + fires_broad
        return 1 if score >= 3 else 0

    # -------- Evaluation --------
    def evaluate(self, df, label_col="income", ruleset=None, min_fires=1, weighted=False, weight_threshold=1.0, hybrid=False):
        if hybrid:
            df["predicted"] = df.apply(lambda r: self.predict_row_hybrid(r), axis=1)
        else:
            if ruleset:
                self.set_active_ruleset(ruleset)
            self.reset_ruleset_stats(self.active_ruleset)
            df["predicted"] = self.predict(df, min_fires=min_fires, weighted=weighted, weight_threshold=weight_threshold)

        y_true, y_pred = df[label_col], df["predicted"]
        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        TP = int(((y_true == 1) & (y_pred == 1)).sum())
        FP = int(((y_true == 0) & (y_pred == 1)).sum())
        TN = int(((y_true == 0) & (y_pred == 0)).sum())
        FN = int(((y_true == 1) & (y_pred == 0)).sum())
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        now = datetime.now()
        print(f"\n==== {now} ====")
        print(f"Overall Accuracy: {acc*100:.2f}%")
        print(f"Class 0 Precision/Recall/F1: {report['0']['precision']*100:.2f}/{report['0']['recall']*100:.2f}/{report['0']['f1-score']*100:.2f}")
        print(f"Class 1 Precision/Recall/F1: {report['1']['precision']*100:.2f}/{report['1']['recall']*100:.2f}/{report['1']['f1-score']*100:.2f}")
        print(f"Specificity (TNR): {specificity*100:.2f}%")
        print(f"==== End {now} ====\n")

# -------- Preprocessing --------
def preprocess_df(df_raw, columns, num_cols):
    df = df_raw.copy()
    df.columns = columns
    obj_cols = df.select_dtypes(include=['object']).columns
    df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="constant", fill_value="Missing")
    cat_cols = [c for c in df.select_dtypes(include=['object']).columns if c != "income"]
    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

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
        if val in prof: return "Professional"
        if val in clerical: return "Clerical"
        if val in manual: return "Manual"
        if val in service: return "Service"
        return "Other"

    def _education_num_to_bucket(num):
        n = int(num)
        if n <= 9: return "Elem"
        if 10 <= n <= 12: return "HS-grad"
        if n == 13: return "Bachelors"
        if n == 14: return "Masters"
        if n == 15: return "Prof-school"
        if n >= 16: return "Doctorate"
        return "Other"

    def _combine_marital(val):
        if val in {"Married-civ-spouse", "Married-spouse-absent", "Married-AF-spouse"}: return "Married"
        if val in {"Divorced", "Widowed", "Separated"}: return "Previously-Married"
        return val or "Other"

    def _combine_relationship(val):
        if val in {"Not-in-family", "Other-relative", "Unmarried"}: return "Not-in-family"
        return val or "Other"

    df["workclass"] = df["workclass"].apply(_combine_workclass)
    df["occupation"] = df["occupation"].apply(_combine_occupation)
    df["education"] = df["education-num"].apply(_education_num_to_bucket)
    df["marital-status"] = df["marital-status"].apply(_combine_marital)
    df["relationship"] = df["relationship"].apply(_combine_relationship)
    df["native-country"] = df["native-country"].apply(lambda v: "US" if v == "United-States" else "Non-US")

    df["income"] = df["income"].astype(str).str.contains(">50K").astype(int)
    return df

def add_numeric_buckets(df):
    bins = {
        "age": [0, 25, 35, 45, 55, 65, 100],
        "hours-per-week": [0, 30, 40, 50, 60, 100],
        "capital-gain": [-1, 0, 1000, 5000, 10000, 99999],
    }
    for col, edges in bins.items():
        labels = [f"{col}_b{i}" for i in range(len(edges) - 1)]
        df[f"{col}_bucket"] = pd.cut(df[col], bins=edges, labels=labels, include_lowest=True)
    return df

# -------- Auto Rule Discovery --------
def auto_discover_rules(model, df, threshold=0.5, min_support=0.02, max_features=2, ruleset_name="auto"):
    model.create_ruleset(ruleset_name)
    model.set_active_ruleset(ruleset_name)
    target = "income"
    total_rows = len(df)
    cat_cols = [c for c in df.columns if (df[c].dtype == "object" or "_bucket" in c) and c != target]

    rule_count = 0
    for k in range(1, max_features + 1):
        print(f"🔍 mining {k}-feature combos for {ruleset_name}...")
        for cols in combinations(cat_cols, k):
            tab = df.groupby(list(cols))[target].agg(["mean", "count"])
            for values, row in tab.iterrows():
                support = row["count"] / total_rows
                if support >= min_support and row["mean"] > threshold:
                    vals = values if isinstance(values, tuple) else (values,)
                    rule_name = "__".join([f"{c}_{v}" for c, v in zip(cols, vals)])
                    model.add_rule(rule_name, lambda r, cols=cols, vals=vals: all(r[c] == v for c, v in zip(cols, vals)), ruleset=ruleset_name)
                    rule_count += 1
    print(f"✅ {rule_count} rules added to '{ruleset_name}' (thresh={threshold}, support={min_support})")
    return ruleset_name

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

# -------- Main --------
def main():
    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]
    num_cols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    df = pd.read_csv("data/adult.data.csv", header=0, skipinitialspace=True, na_values=["?", " ?"])
    df = preprocess_df(df, columns, num_cols)
    df = add_numeric_buckets(df)

    model = RuleBasedIncomeModel()

    # mine two rule sets
    auto_discover_rules(model, df, threshold=0.65, min_support=0.01, max_features=2, ruleset_name="precise")
    auto_discover_rules(model, df, threshold=0.5, min_support=0.02, max_features=2, ruleset_name="broad")

    # evaluate different voting styles
    print("=== Strict (min_fires=2) ===")
    model.evaluate(df, ruleset="precise", min_fires=2)
    print("=== Weighted Voting ===")
    model.evaluate(df, ruleset="precise", weighted=True, weight_threshold=1.2)
    print("=== Hybrid (precise + broad) ===")
    model.evaluate(df, hybrid=True)

    metrics = compute_rule_metrics(model, df, ruleset="precise")
    pd.DataFrame(metrics).sort_values("f1", ascending=False).head(30)

    metrics_broad = compute_rule_metrics(model, df, ruleset="broad")
    pd.DataFrame(metrics_broad).sort_values("f1", ascending=False).head(30)

    export_rule_metrics_csv(model, df, path="precise_rules.csv", ruleset="precise")
    export_rule_metrics_csv(model, df, path="broad_rules.csv", ruleset="broad")



if __name__ == "__main__":
    main()

