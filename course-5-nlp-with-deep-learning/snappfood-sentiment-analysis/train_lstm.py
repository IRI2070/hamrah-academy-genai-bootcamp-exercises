import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from datasets import load_dataset

dataset = load_dataset("ParsiAI/snappfood-refined-sentiment-dataset")

train_texts, train_labels = dataset["train"]["comment"], dataset["train"]["label_id"]
val_texts, val_labels = dataset["validation"]["comment"], dataset["validation"]["label_id"]
test_texts, test_labels = dataset["test"]["comment"], dataset["test"]["label_id"]

vocab_size = 10000
max_len = 512

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)

X_train = pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=max_len)
X_val = pad_sequences(tokenizer.texts_to_sequences(val_texts), maxlen=max_len)
X_test = pad_sequences(tokenizer.texts_to_sequences(test_texts), maxlen=max_len)

y_train = tf.convert_to_tensor(train_labels)
y_val = tf.convert_to_tensor(val_labels)
y_test = tf.convert_to_tensor(test_labels)

model = Sequential([
    Embedding(vocab_size, 128, input_length=max_len),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

history = model.fit(
    X_train, y_train,
    epochs=3,
    batch_size=64,
    validation_data=(X_val, y_val)
)

print(history.history.keys())

for epoch in range(len(history.history['loss'])):
    print(f"Epoch {epoch + 1}: "
          f"loss={history.history['loss'][epoch]:.4f}, "
          f"acc={history.history['accuracy'][epoch]:.4f}, "
          f"val_loss={history.history['val_loss'][epoch]:.4f}, "
          f"val_acc={history.history['val_accuracy'][epoch]:.4f}")

# ParsiAI/snappfood-sentiment-analysis
# Epoch 1: loss=0.3919, acc=0.8286, val_loss=0.3402, val_acc=0.8573
# Epoch 2: loss=0.3144, acc=0.8700, val_loss=0.3600, val_acc=0.8531
# Epoch 3: loss=0.2798, acc=0.8842, val_loss=0.3505, val_acc=0.8589

# IRI2070/snappfood-refined-sentiment-dataset
# Epoch 1: loss=0.3358, acc=0.8588, val_loss=0.0681, val_acc=0.9796
# Epoch 2: loss=0.0845, acc=0.9733, val_loss=0.0430, val_acc=0.9844
# Epoch 3: loss=0.0478, acc=0.9844, val_loss=0.0392, val_acc=0.9860

loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.2f}")

# ParsiAI/snappfood-sentiment-analysis
# accuracy: 0.8573 - loss: 0.344

# IRI2070/snappfood-refined-sentiment-dataset
# accuracy: 0.9851 - loss: 0.0442

sample = ["غذا شور بود", "شیرینی تازه بود"]
X_sample = pad_sequences(tokenizer.texts_to_sequences(sample), maxlen=max_len)

print(model.predict(X_sample))

# ParsiAI/snappfood-sentiment-analysis
# 0.7232165, 0.11191478

# IRI2070/snappfood-refined-sentiment-dataset
# 0.9754438, 0.04563862
