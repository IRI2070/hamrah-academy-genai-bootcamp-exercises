from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

dataset = load_dataset("IRI2070/snappfood-refined-sentiment-dataset")

train_texts = dataset["train"]["comment"]
train_labels = dataset["train"]["label_id"]

test_texts = dataset["test"]["comment"]
test_labels = dataset["test"]["label_id"]

model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("clf", LogisticRegression(max_iter=1000))
])

model.fit(train_texts, train_labels)

y_pred = model.predict(test_texts)
print(classification_report(test_labels, y_pred))
#               precision    recall  f1-score   support
#
#          0.0       0.99      0.98      0.99      3084
#          1.0       0.97      0.99      0.98      2436
#
#     accuracy                           0.98      5520
#    macro avg       0.98      0.99      0.98      5520
# weighted avg       0.98      0.98      0.98      5520

print(model.predict(["غذا شور بود"]))  # [1.]
print(model.predict(["شیرینی تازه بود"]))  # [0.]
