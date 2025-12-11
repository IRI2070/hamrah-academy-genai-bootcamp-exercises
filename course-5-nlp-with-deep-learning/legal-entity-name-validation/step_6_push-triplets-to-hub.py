import json
import random
from datasets import Dataset, DatasetDict

with open("triplets.jsonl", "r", encoding="utf-8") as f:
    triplets = [json.loads(line.strip()) for line in f]

random.shuffle(triplets)

train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
n = len(triplets)
train_end = int(train_ratio * n)
val_end = train_end + int(val_ratio * n)

train_data = triplets[:train_end]
val_data = triplets[train_end:val_end]
test_data = triplets[val_end:]

dataset = DatasetDict({
    "train": Dataset.from_list(train_data),
    "validation": Dataset.from_list(val_data),
    "test": Dataset.from_list(test_data),
})

print(dataset)

dataset.push_to_hub("<user_name>/legal-entity-name-validation-bi-encoder-dataset", token='hf_XXXXXXXXXXXXXXXXX')
