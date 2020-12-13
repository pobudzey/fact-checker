import csv
from naive_bayes import GaussianNB
from pathlib import Path
from sklearn.metrics import classification_report

clf = GaussianNB()

# Trace file
trace_data = []
expected_categories = []
predicted_categories = []
test_set = Path.cwd() / "datasets" / "covid_test_public.tsv"
if test_set.exists():
    with open(test_set, encoding="utf8") as tsv_file:
        reader = csv.reader(tsv_file, delimiter="\t")
        for row in reader:
            tweet_id = row[0]
            text = row[1].lower()
            predicted_category, predicted_category_score = clf.predict(text)
            predicted_category_score = "{:1.2E}".format(predicted_category_score)
            expected_category = row[2]
            expected_categories.append(expected_category)
            predicted_categories.append(predicted_category)
            if predicted_category == expected_category:
                label = "correct"
            else:
                label = "wrong"
            trace_data.append(
                [
                    tweet_id,
                    predicted_category,
                    predicted_category_score,
                    expected_category,
                    label,
                ]
            )
else:
    raise FileNotFoundError("Training set missing.")

# Generate trace and evaluation files
output_dir = Path.cwd().parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)
if output_dir.exists():
    with open(output_dir / "trace_NB-BOW-OV.txt", "w") as f:
        for row in trace_data:
            f.write(f"{row[0]}  {row[1]}  {row[2]}  {row[3]} {row[4]}\n")
    with open(output_dir / "eval_NB-BOW-OV.txt", "w") as f:
        report = classification_report(
            expected_categories, predicted_categories, output_dict=True
        )
        f.write(f"{round(report['accuracy'], 4)}\n")
        f.write(
            f"{round(report['yes']['precision'], 4)}  {round(report['no']['precision'], 4)}\n"
        )
        f.write(
            f"{round(report['yes']['recall'], 4)}  {round(report['no']['recall'], 4)}\n"
        )
        f.write(
            f"{round(report['yes']['f1-score'], 4)}  {round(report['no']['f1-score'], 4)}\n"
        )
else:
    raise FileNotFoundError("Error creating output directory.")
