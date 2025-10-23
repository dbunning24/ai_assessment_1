import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from datetime import datetime

# ----------------------------------------------
# 1. Load Training + Test Datasets
# ----------------------------------------------
train_path = "data/adult.data.csv"  
test_path  = "data/adult.test.csv"    

columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

train = pd.read_csv(train_path, header=0, names=columns, na_values=" ?", skipinitialspace=True)
test  = pd.read_csv(test_path,  header=0, names=columns, na_values=" ?", skipinitialspace=True)

# ----------------------------------------------
# 2. Basic Cleaning
# ----------------------------------------------
train.dropna(inplace=True)
test.dropna(inplace=True)

train["income"] = train["income"].apply(lambda x: 1 if ">50K" in str(x) else 0)
test["income"]  = test["income"].apply(lambda x: 1 if ">50K" in str(x) else 0)

# ----------------------------------------------
# 3. Define Rules
# ----------------------------------------------
rule_counts = {"education_hours": 0, "married_occupation": 0, "capital_gain": 0}

def predict_rule_based(row):
    if int(row["education-num"]) >= 15 and int(row["hours-per-week"]) > 40:
        rule_counts["education_hours"] += 1
        return 1
    if row["marital-status"] == "Married-civ-spouse" and row["occupation"] in ["Exec-managerial", "Prof-specialty"]:
        rule_counts["married_occupation"] += 1
        return 1
    if int(row["capital-gain"]) > 5000:
        rule_counts["capital_gain"] += 1
        return 1
    return 0

# ----------------------------------------------
# 4. Apply Rules to Test Set
# ----------------------------------------------
test["predicted"] = test.apply(predict_rule_based, axis=1)

# ----------------------------------------------
# 5. Evaluate
# ----------------------------------------------
accuracy = accuracy_score(test["income"], test["predicted"]) * 100
report = classification_report(test["income"], test["predicted"], output_dict=True)

# Build readable string output
output_string = f"Overall Accuracy: {accuracy:.2f}%\n\n"
output_string += "Per-class metrics:\n"

class_labels = {"0": "<=50K", "1": ">50K"}

for label, name in class_labels.items():
    metrics = report[label]
    output_string += (
        f"  Class {label} ({name}): "
        f"Precision={metrics['precision']*100:.2f}%, "
        f"Recall={metrics['recall']*100:.2f}%, "
        f"F1={metrics['f1-score']*100:.2f}%\n"
    )

output_string += (
    f"\nMacro F1: {report['macro avg']['f1-score']*100:.2f}%\n"
    f"Weighted F1: {report['weighted avg']['f1-score']*100:.2f}%\n"
    f"\nRule Activation Counts: {rule_counts}\n"
)

print(output_string)

# Save to file
with open("results.txt", "a") as f:
    now = str(datetime.now())
    f.write(f"==== {now} ====\n")
    f.write(output_string)
    f.write(f"==== {now} ====\n\n")
