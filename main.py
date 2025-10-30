# code developed by both a human and chatgpt

# ==============================================
# Rule-Based Income Model (All Rules Evaluated)
# ==============================================
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from datetime import datetime

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

    def add_rule(self, name, func, ruleset="default"):
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

    def predict_row(self, row):
        """Evaluate all rules for a row and update stats for each rule-key."""
        prediction = 0
        for name_key, func in self.rules:
            # name_key is "{ruleset}::{rulename}"
            rule_prediction = 1 if func(row) else 0
            true_label = row["income"]

            # Update stats for this rule-key
            if rule_prediction == 1:
                self.rule_counts[name_key] = self.rule_counts.get(name_key, 0) + 1
                if true_label == 1:
                    self.rule_stats[name_key]["TP"] += 1
                else:
                    self.rule_stats[name_key]["FP"] += 1
            else:
                if true_label == 1:
                    self.rule_stats[name_key]["FN"] += 1
                else:
                    self.rule_stats[name_key]["TN"] += 1

            # Overall prediction: 1 if any rule fires
            if rule_prediction == 1:
                prediction = 1

        return prediction

    def predict(self, df):
        return df.apply(self.predict_row, axis=1)

    def evaluate(self, df, label_col="income", ruleset=None):
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
        df["predicted"] = self.predict(df)
        y_true = df[label_col]
        y_pred = df["predicted"]

        acc = accuracy_score(y_true, y_pred)
        # avoid UndefinedMetricWarning when a label has no predicted samples
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
 
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

            rule_metrics.append({
                "name": name,
                "activations": self.rule_counts.get(key, 0),
                "TP": TP,
                "FP": FP,
                "TN": TN,
                "FN": FN,
                "precision": precision,
                "recall": recall,
                "f1": f1
            })

        # Sort rules by F1 descending
        rule_metrics.sort(key=lambda x: x["f1"], reverse=True)

        # Build pretty table
        table_header = f"{'Rule':30} {'Act':>5} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5} {'Prec':>7} {'Rec':>7} {'F1':>7}\n"
        table_header += "-" * 80 + "\n"

        table_rows = ""
        for rm in rule_metrics:
            table_rows += (
                f"{rm['name']:30} {rm['activations']:5} {rm['TP']:5} {rm['FP']:5} "
                f"{rm['TN']:5} {rm['FN']:5} {rm['precision']*100:7.2f} "
                f"{rm['recall']*100:7.2f} {rm['f1']*100:7.2f}\n"
            )

        footer = "-" * 80 + f"\n==== End {now} ====\n"

        output = header + overall + table_header + table_rows + footer

        print(output)

        #with open("results.txt", "a") as f:
        #    f.write(output)

# ==============================================
# 1. Load and Clean Data
# ==============================================
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

train_path = "data/adult.data.csv"

# Read file: treat '?' as missing, strip spaces after delimiters
df = pd.read_csv(train_path, header=0, skipinitialspace=True, na_values=['?', ' ?'])

# If file doesn't already have the expected header, enforce column names
if list(df.columns) != columns:
    df.columns = columns

# Strip whitespace in all object/string columns
obj_cols = df.select_dtypes(include=['object']).columns
df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())

# Convert numeric-like columns to numeric dtypes (coerce errors -> NaN)
num_cols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

# --- report missing values so you can inspect ---
missing_per_col = df.isna().sum()
print("Missing values per column:\n", missing_per_col[missing_per_col > 0])
total_missing_rows = df.isna().any(axis=1).sum()
print(f"Total rows with any missing: {total_missing_rows} / {len(df)}")

# --- capture rows that have missing values BEFORE imputation ---
missing_rows_df = df[df.isna().any(axis=1)].copy()

# --- record which cells were missing (indicator features) BEFORE imputation ---
# exclude target from categorical imputation/indicators for now
cat_cols_all = df.select_dtypes(include=['object']).columns.tolist()
cat_cols = [c for c in cat_cols_all if c != "income"]

was_missing = df[num_cols + cat_cols].isna()
for c in was_missing.columns:
    df[f"{c}_was_missing"] = was_missing[c].astype(int)

# --- imputation ---
# Numeric: median
num_imputer = SimpleImputer(strategy="median")
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Categorical: preserve missingness as its own category
cat_imputer = SimpleImputer(strategy="constant", fill_value="Missing")
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# --- combine sparse categories for 'workclass' to reduce granularity ---
def _combine_workclass(val):
    if val in ("Federal-gov", "Local-gov", "State-gov"):
        return "Gov"
    if val in ("Self-emp-inc", "Self-emp-not-inc"):
        return "Self"
    if val in ("Never-worked", "Without-pay", "Unknown", "Missing"):
        return "Other/Unknown"
    return val

df["workclass"] = df["workclass"].apply(_combine_workclass)

# --- aggregate occupation into broader groups (clerical/manual/service/professional/other) ---
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

df["occupation"] = df["occupation"].apply(_combine_occupation)

# --- aggregate education into simplified buckets ---
def _combine_education(val):
    if val in {"1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th"}:
        return "Elem"
    if val in {"12th", "HS-grad"}:
        return "HS-grad"
    if val in {"Assoc-acdm", "Assoc-voc"}:
        return "Assoc"
    if val in {"Bachelors", "Masters", "Doctorate", "Prof-school", "Some-college", "Preschool"}:
        return val
    return val or "Other"

df["education"] = df["education"].apply(_combine_education)

# --- aggregate marital-status into Married / Previously-Married / Never-married ---
def _combine_marital(val):
    if val in {"Married-civ-spouse", "Married-spouse-absent", "Married-AF-spouse"}:
        return "Married"
    if val in {"Divorced", "Widowed", "Separated"}:
        return "Previously-Married"
    return val or "Other"

df["marital-status"] = df["marital-status"].apply(_combine_marital)

# --- simplify relationship into Not-in-family vs others ---
def _combine_relationship(val):
    if val in {"Not-in-family", "Other-relative", "Unmarried"}:
        return "Not-in-family"
    return val or "Other"

df["relationship"] = df["relationship"].apply(_combine_relationship)

# --- collapse native-country into US vs Non-US ---
df["native-country"] = df["native-country"].apply(lambda v: "US" if v == "United-States" else "Non-US")

# Optionally: if you use the `workclasses` list later to add rules,
# replace the original granular entries with the aggregated labels (e.g. "Gov","Self","Other/Unknown").

# If target had missing values, fill with most frequent label (rare)
if df["income"].isna().any():
    df["income"].fillna(df["income"].mode().iloc[0], inplace=True)

# Robust target mapping to 0/1
df["income"] = df["income"].astype(str).str.contains(">50K").astype(int)

# --- print a few rows that originally had missing values (saved before imputation) ---
print("\nRows with missing values — BEFORE imputation:")
if len(missing_rows_df) == 0:
    print("No rows had missing values.")
else:
    # show original dataset columns in the expected order
    cols_to_show = columns.copy()
    print(missing_rows_df[cols_to_show].head(10))

# --- locate and print the same rows AFTER imputation ---
imputed_rows_df = df.loc[missing_rows_df.index, cols_to_show].copy()

print("\nSame rows — AFTER imputation:")
if imputed_rows_df.empty:
    print("No rows to show after imputation.")
else:
    print(imputed_rows_df.head(10))

# --- optionally list which columns were missing in those rows ---
if 'was_missing' in globals() or 'was_missing' in locals():
    cols_missing_in_sample = was_missing.loc[missing_rows_df.index].any()
    filled_cols = cols_missing_in_sample[cols_missing_in_sample].index.tolist()
    if filled_cols:
        print("\nColumns that contained missing values in those rows:", filled_cols)

# ==============================================
# 2. Create and Add Rules
# ==============================================
model = RuleBasedIncomeModel()

"""
# 1. High education + moderate/high hours
model.add_rule("education_hours",
               lambda r: int(r["education-num"]) >= 13 and int(r["hours-per-week"]) > 50)

# 2. Married + high-status occupation + relationship
model.add_rule("married_occupation_relationship",
               lambda r: r["marital-status"] == "Married-civ-spouse"
               and r["occupation"] in ["Exec-managerial", "Prof-specialty", "Sales", "Tech-support"]
               and r["relationship"] in ["Husband", "Wife"])

# 3. Technical / professional jobs + higher hours
model.add_rule("technical_professional",
               lambda r: r["occupation"] in ["Tech-support", "Prof-specialty", "Exec-managerial", "Sales"]
               and int(r["hours-per-week"]) >= 50)

# 4. High capital gain + moderate hours
model.add_rule("capital_gain",
               lambda r: int(r["capital-gain"]) > 5000 and int(r["hours-per-week"]) > 40)

# 5. High education + technical 
model.add_rule("education_occupation",
               lambda r: int(r["education-num"]) >= 14 
               and r["occupation"] in ["Tech-support", "Prof-specialty", "Exec-managerial", "Sales"])
"""
# =========================
# Age-based rules (continuous)
# =========================
model.add_rule("age_over_40", lambda r: r["age"] > 40)
model.add_rule("age_over_30", lambda r: r["age"] > 30)

# =========================
# Workclass rules (categorical)
# =========================
workclasses = sorted(df["workclass"].dropna().unique().tolist())
for wc in workclasses:
    model.add_rule(f"workclass_{wc.replace('-', '_').replace(' ','_').replace('/','_')}", lambda r, wc=wc: r["workclass"] == wc)

# =========================
# Education rules (use education-num ranges to cover all values)
# =========================
# Use contiguous buckets so every education-num value is caught by at least one rule.
model.add_rule("education_num_le_9", lambda r: r["education-num"] <= 9)
model.add_rule("education_num_10_12", lambda r: 10 <= r["education-num"] <= 12)
model.add_rule("education_num_13_14", lambda r: 13 <= r["education-num"] <= 14)
model.add_rule("education_num_15", lambda r: r["education-num"] == 15)
model.add_rule("education_num_ge_16", lambda r: r["education-num"] >= 16)

# =========================
# Marital-status rules
# =========================
maritals = sorted(df["marital-status"].dropna().unique().tolist())
for ms in maritals:
    model.add_rule(f"marital_{str(ms).replace('-', '_').replace(' ','_')}", lambda r, ms=ms: r["marital-status"] == ms)

# =========================
# Occupation rules (aggregated groups)
# =========================
occupations = sorted(df["occupation"].dropna().unique().tolist())
for occ in occupations:
    model.add_rule(f"occupation_{str(occ).replace('-', '_').replace(' ','_')}", lambda r, occ=occ: r["occupation"] == occ)

# =========================
# Relationship rules
# =========================
relationships = sorted(df["relationship"].dropna().unique().tolist())
for rel in relationships:
    model.add_rule(f"relationship_{str(rel).replace('-', '_').replace(' ','_')}", lambda r, rel=rel: r["relationship"] == rel)

# =========================
# Race rules
# =========================
races = ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
for race in races:
    model.add_rule(f"race_{race.replace('-', '_')}", lambda r, race=race: r["race"] == race)

# =========================
# Sex rules
# =========================
for sex in ["Female", "Male"]:
    model.add_rule(f"sex_{sex}", lambda r, sex=sex: r["sex"] == sex)

# =========================
# Capital-gain / capital-loss (continuous)
# =========================
model.add_rule("capital_gain_positive", lambda r: r["capital-gain"] > 0)
model.add_rule("capital_gain_high", lambda r: r["capital-gain"] > 5000)
model.add_rule("capital_loss_positive", lambda r: r["capital-loss"] > 0)
model.add_rule("capital_loss_high", lambda r: r["capital-loss"] > 2000)

# =========================
# Hours-per-week (continuous)
# =========================
model.add_rule("hours_over_40", lambda r: r["hours-per-week"] > 40)
model.add_rule("hours_over_50", lambda r: r["hours-per-week"] > 50)

# =========================
# Native-country rules
# =========================
countries = ["United-States","Cambodia","England","Puerto-Rico","Canada","Germany","Outlying-US(Guam-USVI-etc)",
             "India","Japan","Greece","South","China","Cuba","Iran","Honduras","Philippines","Italy","Poland",
             "Jamaica","Vietnam","Mexico","Portugal","Ireland","France","Dominican-Republic","Laos","Ecuador",
             "Taiwan","Haiti","Columbia","Hungary","Guatemala","Nicaragua","Scotland","Thailand","Yugoslavia",
             "El-Salvador","Trinadad&Tobago","Peru","Hong","Holand-Netherlands"]
for country in countries:
    model.add_rule(f"country_{country.replace('-', '_').replace(' ','_').replace('(','').replace(')','')}",
                   lambda r, country=country: r["native-country"] == country)

# ==============================================
# 3. Evaluate on Test Set
# ==============================================
model.create_ruleset("default")  # harmless if already exists

def run_evaluation(ruleset="default"):
    """
    Run evaluation using the specified ruleset name (default is 'default').
    Example: run_evaluation("default") or run_evaluation("my_ruleset")
    """
    model.evaluate(df, ruleset=ruleset)

# ========
# combined ruleset
# ========
model.create_ruleset("combined")

#model.add_rule("")

run_evaluation("default")

# visualisation
import matplotlib.pyplot as plt
import numpy as np

# Prepare rule metrics for the active ruleset
rule_metrics = []
prefix = f"{model.active_ruleset}::"
for key, stats in model.rule_stats.items():
    # only include rules from the active ruleset
    if not key.startswith(prefix):
        continue
    rule_name = key.split("::", 1)[1]
    TP = stats["TP"]
    FP = stats["FP"]
    FN = stats["FN"]
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    rule_metrics.append({
        "name": rule_name,
        "f1": f1,
        "activations": model.rule_counts.get(key, 0)
    })

# Filter rules by thresholds
f1_threshold = 0.1           # minimum F1 to include
activation_threshold = 50    # minimum activations to include
filtered_rules = [
    rm for rm in rule_metrics 
    if rm["f1"] >= f1_threshold and rm["activations"] >= activation_threshold
]

# Sort by F1 descending
filtered_rules.sort(key=lambda x: x["f1"], reverse=True)

names = [rm["name"] for rm in filtered_rules]
f1_scores = [rm["f1"] for rm in filtered_rules]
activations = [rm["activations"] for rm in filtered_rules]

y_pos = np.arange(len(names))
# Create combined figure
fig, ax1 = plt.subplots(figsize=(12, max(6, len(names)*0.3)))

color1 = 'tab:blue'
ax1.barh(y_pos - 0.2, f1_scores, height=0.4, color=color1, label="F1 Score")
ax1.set_xlabel("F1 Score", color=color1)
ax1.set_ylabel("Rule")
ax1.set_yticks(y_pos)
ax1.set_yticklabels(names)
ax1.invert_yaxis()  # Highest F1 at top
ax1.tick_params(axis='x', labelcolor=color1)

# Second axis for activations
ax2 = ax1.twiny()
color2 = 'tab:red'
ax2.barh(y_pos + 0.2, activations, height=0.4, color=color2, label="Activations")
ax2.set_xlabel("Number of Activations", color=color2)
ax2.tick_params(axis='x', labelcolor=color2)

# Legends
ax1.legend(loc='lower right')
ax2.legend(loc='upper right')

plt.title(f"Rule Performance for '{model.active_ruleset}' ruleset (F1 ≥ {f1_threshold}, Activations ≥ {activation_threshold})")
plt.tight_layout()
plt.show()

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
