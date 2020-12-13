from naive_bayes import GaussianNB

clf = GaussianNB()

category, score = clf.predict(
    "You do. I hear it affects pigs in a big way. https://t.co/ZCMTJzKb35"
)

print(category)
print(score)
