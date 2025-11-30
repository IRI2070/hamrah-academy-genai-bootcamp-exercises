from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate

dataset = load_dataset("ParsiAI/snappfood-sentiment-analysis")
tokenizer = AutoTokenizer.from_pretrained("PartAI/TookaBERT-Large")
model = AutoModelForSequenceClassification.from_pretrained("PartAI/TookaBERT-Large", num_labels=2)
accuracy = evaluate.load("accuracy")


def tokenize_function(example):
    return tokenizer(example["comment"], truncation=True, padding="max_length", max_length=512)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.evaluate(tokenized_datasets["test"])

trainer.save_model("./tooka-bert-large-snappfood-sentiment")
