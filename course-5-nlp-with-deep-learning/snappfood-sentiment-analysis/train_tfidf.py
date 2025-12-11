from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, log_loss

dataset = load_dataset("ParsiAI/snappfood-refined-sentiment-dataset")

train_texts = dataset["train"]["comment"]
train_labels = dataset["train"]["label_id"]

test_texts = dataset["test"]["comment"]
test_labels = dataset["test"]["label_id"]

val_texts = dataset["validation"]["comment"]
val_labels = dataset["validation"]["label_id"]

model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("clf", LogisticRegression(max_iter=1000, verbose=1))
])

model.fit(train_texts, train_labels)

train_pred = model.predict(train_texts)
train_proba = model.predict_proba(train_texts)
train_acc = accuracy_score(train_labels, train_pred)
train_loss = log_loss(train_labels, train_proba)

val_pred = model.predict(val_texts)
val_proba = model.predict_proba(val_texts)
val_acc = accuracy_score(val_labels, val_pred)
val_loss = log_loss(val_labels, val_proba)

test_pred = model.predict(test_texts)
test_proba = model.predict_proba(test_texts)
test_acc = accuracy_score(test_labels, test_pred)
test_loss = log_loss(test_labels, test_proba)

print(f"Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
print(f"Val:   loss={val_loss:.4f}, acc={val_acc:.4f}")
print(f"Test:  loss={test_loss:.4f}, acc={test_acc:.4f}")

# IRI2070/snappfood-refined-sentiment-dataset
# Train: loss=0.0929, acc=0.9813
# Val:   loss=0.0816, acc=0.9846
# Test:  loss=0.0835, acc=0.9848

# ParsiAI/snappfood-sentiment-analysis
# Train: loss=0.3159, acc=0.8733
# Val:   loss=0.3447, acc=0.8555
# Test:  loss=0.3528, acc=0.8504

y_pred = model.predict(test_texts)
print(classification_report(test_labels, y_pred))

print(model.predict(["غذا شور بود"]))  # [1.]
print(model.predict(["شیرینی تازه بود"]))  # [0.]
