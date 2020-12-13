import csv
from pathlib import Path
from collections import defaultdict
from math import log10


class GaussianNB:
    def __init__(self, vocabulary="original"):
        self.vocabulary = (
            self.build_original_vocabulary()
            if vocabulary == "original"
            else self.build_filtered_vocabulary()
        )
        self.conditionals = None
        self.priors = None
        self.fit()

    def build_original_vocabulary(self):
        print("Building original vocabulary...")
        original_vocabulary = []
        training_set = Path.cwd() / "datasets" / "covid_training.tsv"
        if training_set.exists():
            with open(training_set, encoding="utf8") as tsv_file:
                reader = csv.reader(tsv_file, delimiter="\t")
                next(reader)
                for row in reader:
                    text = row[1].lower()
                    for word in text.split():
                        if word not in original_vocabulary:
                            original_vocabulary.append(word)
            print("Done!")
            return original_vocabulary
        else:
            raise FileNotFoundError("Training set missing.")

    def build_filtered_vocabulary(self):
        print("Building filtered vocabulary...")
        filtered_vocabulary = []
        count = defaultdict(int)
        training_set = Path.cwd() / "datasets" / "covid_training.tsv"
        if training_set.exists():
            with open(training_set, encoding="utf8") as tsv_file:
                reader = csv.reader(tsv_file, delimiter="\t")
                next(reader)
                for row in reader:
                    text = row[1].lower()
                    for word in text.split():
                        count[word] += 1
                for word in count:
                    if count[word] > 1:
                        filtered_vocabulary.append(word)
            print("Done!")
            return filtered_vocabulary
        else:
            raise FileNotFoundError("Training set missing.")

    def fit(self):
        print("Fitting model...")
        conditionals = {
            "yes": {word: 0 for word in self.vocabulary},
            "no": {word: 0 for word in self.vocabulary},
        }
        priors = {"yes": 0, "no": 0}
        number_of_tweets = 0
        total_yes = 0
        total_no = 0
        # Counting word occurences
        training_set = Path.cwd() / "datasets" / "covid_training.tsv"
        if training_set.exists():
            with open(training_set, encoding="utf8") as tsv_file:
                reader = csv.reader(tsv_file, delimiter="\t")
                next(reader)
                for row in reader:
                    text = row[1].lower()
                    category = row[2]
                    for word in text.split():
                        if word in self.vocabulary:
                            conditionals[category][word] += 1
                            if category == "yes":
                                total_yes += 1
                            else:
                                total_no += 1
                    priors[category] += 1
                    number_of_tweets += 1
            # Applying additive smoothing
            for category in conditionals:
                for word in conditionals[category]:
                    conditionals[category][word] += 0.01
            total_yes += 0.01 * len(self.vocabulary)
            total_no += 0.01 * len(self.vocabulary)
            # Calculating conditional and prior probabilities
            for category in conditionals:
                for word in conditionals[category]:
                    if category == "yes":
                        divisor = total_yes
                    else:
                        divisor = total_no
                    conditionals[category][word] /= divisor
            for category in priors:
                priors[category] /= number_of_tweets
            self.conditionals = conditionals
            self.priors = priors
            print("Done!")
        else:
            raise FileNotFoundError("Training set missing.")

    def predict(self, text):
        score_yes = log10(self.priors["yes"])
        score_no = log10(self.priors["no"])
        for word in text.lower().split():
            if word in self.vocabulary:
                score_yes += log10(self.conditionals["yes"][word])
                score_no += log10(self.conditionals["no"][word])
        if score_yes > score_no:
            return "yes", score_yes
        else:
            return "no", score_no
