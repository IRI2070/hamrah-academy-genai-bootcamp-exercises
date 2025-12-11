from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

raw_datasets = load_dataset("IRI2070/snappfood-refined-sentiment-dataset")


def fix_labels(example):
    label = example["label_id"]
    example["labels"] = int(label)
    return example


raw_datasets = raw_datasets.map(fix_labels, remove_columns=["label_id", "label"])

tokenizer = AutoTokenizer.from_pretrained("PartAI/TookaBERT-Large")


def tokenize_function(examples):
    return tokenizer(
        examples["comment"],
        truncation=True,
        padding=False,
        max_length=512,
    )


tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
)

model = AutoModelForSequenceClassification.from_pretrained(
    "PartAI/TookaBERT-Large",
    num_labels=2,
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


training_args = TrainingArguments(
    output_dir="./bert-sentiment-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./bert-sentiment-finetuned/logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True,
    seed=42,
    report_to="none",
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

trainer.save_model("./tooka-bert-large-snappfood-sentiment")
tokenizer.save_pretrained("./tooka-bert-large-snappfood-sentiment")

# IRI2070/snappfood-refined-sentiment-dataset
# ***** train metrics *****
#   epoch                    =        3.0
#   total_flos               = 88054840GF
#   train_loss               =     0.0406
#   train_runtime            = 5:40:32.94
#   train_samples            =      33818
#   train_samples_per_second =      4.965
#   train_steps_per_second   =      0.497


# ***** eval metrics *****
#   epoch                   =        3.0
#   eval_accuracy           =     0.9996
#   eval_loss               =     0.0033
#   eval_runtime            = 0:04:30.54
#   eval_samples            =       5057
#   eval_samples_per_second =     18.692
#   eval_steps_per_second   =       2.34
