import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
import math

df1 = pd.read_csv('data_unique.csv', index_col=None, encoding='utf-8')
print(df1.shape)

df1 = df1.dropna().drop_duplicates()
# df1 = df1[(df1['label'] != "abbrevation_shortening") & (df1['label'] != "no rule")]
no_rule_df = df1[df1['label'] == 'no_rule']
other_df = df1[df1['label'] != 'no_rule']

no_rule_df = no_rule_df.sample(n=9984, random_state=42)
df1 = pd.concat([other_df, no_rule_df])
print(df1.shape)
print(df1['label'].value_counts())

all_classes = sorted(set(df1["label"].tolist()))

print(all_classes)

class_to_id = {cls: idx for idx, cls in enumerate(all_classes)}

ds = Dataset.from_pandas(df1, preserve_index=False)

print(ds)


def map_classes(example):
    return {"label": class_to_id[example["label"]]}


ds = ds.map(map_classes)
print(ds.shape)
print(ds)

columns_to_check = ["candidate", "registered", "label"]


def is_empty_value(value):
    if value is None:
        return True
    elif isinstance(value, str) and value.strip() == "":
        return True
    elif isinstance(value, float) and math.isnan(value):
        return True
    elif isinstance(value, list) and len(value) == 0:
        return True
    elif isinstance(value, dict) and len(value) == 0:
        return True
    return False


def filter_empty_rows(example):
    for col in columns_to_check:
        if col in example and is_empty_value(example[col]):
            return False
    return True


ds = ds.filter(filter_empty_rows)
print(ds.shape)


def filter_by_word_count(example):
    text = example["candidate"]

    if not isinstance(text, str):
        return False

    words = text.strip().split()
    word_count = len(words)

    return word_count > 2


ds = ds.filter(filter_by_word_count)
print(ds.shape)
print(ds.column_names)

print(sorted(set(ds["label"])))

train_df, temp_df = train_test_split(
    ds.to_pandas(), test_size=0.2, stratify=ds['label'], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42
)

train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))
test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

print(train_ds.shape)
print(val_ds.shape)
print(test_ds.shape)

dataset = DatasetDict({
    "train": train_ds,
    "validation": val_ds,
    "test": test_ds
})

from collections import Counter

for split in dataset.keys():
    labels = dataset[split]["label"]
    counter = Counter(labels)
    print(f"{split} split label counts:")
    print(dict(counter))

dataset.push_to_hub("<user_name>/legal-entity-name-validation", token="hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")