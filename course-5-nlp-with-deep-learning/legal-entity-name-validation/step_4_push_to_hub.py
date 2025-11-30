import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

df = pd.read_csv('data_unique.csv', index_col=None, encoding='utf-8')

df = df.drop_duplicates()

train_df, temp_df = train_test_split(
    df, test_size=0.2, stratify=df['rule'], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df['rule'], random_state=42
)

train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))
test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

dataset = DatasetDict({
    "train": train_ds,
    "validation": val_ds,
    "test": test_ds
})

dataset.push_to_hub("<your-huggingface-username>/legal-entity-name-validation", token="hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXX")
