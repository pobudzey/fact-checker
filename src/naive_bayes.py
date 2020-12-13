import csv
from pathlib import Path
from collections import defaultdict


class GaussianNB:
    def __init__(self, vocabulary="original"):
        self.vocabulary = (
            self.build_original_vocabulary()
            if vocabulary == "original"
            else self.build_filtered_vocabulary()
        )

    def build_original_vocabulary(self):
        original_vocabulary = []
        training_set = Path.cwd() / "datasets" / "covid_training.tsv"
        if training_set.exists():
            with open(training_set, encoding="utf8") as tsv_file:
                reader = csv.reader(tsv_file, delimiter="\t")
                for row in reader:
                    text = row[1].lower()
                    for token in text.split():
                        if token not in original_vocabulary:
                            original_vocabulary.append(token)
            return original_vocabulary
        else:
            raise FileNotFoundError("Training set missing.")

    def build_filtered_vocabulary(self):
        filtered_vocabulary = []
        count = defaultdict(int)
        training_set = Path.cwd() / "datasets" / "covid_training.tsv"
        if training_set.exists():
            with open(training_set, encoding="utf8") as tsv_file:
                reader = csv.reader(tsv_file, delimiter="\t")
                for row in reader:
                    text = row[1].lower()
                    for token in text.split():
                        count[token] += 1
                for token in count:
                    if count[token] > 1:
                        filtered_vocabulary.append(token)
            return filtered_vocabulary
        else:
            raise FileNotFoundError("Training set missing.")