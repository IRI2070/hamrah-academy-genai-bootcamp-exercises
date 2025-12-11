import pandas as pd
from dadmatools.normalizer import Normalizer

df = pd.read_csv('data_sample.csv')
print(df.shape)

normalizer = Normalizer(full_cleaning=True)


def clean_name(name: str) -> str | None:
    if not isinstance(name, str):
        return None

    name = normalizer.normalize(name.strip())

    if name == "":
        return None

    parts = name.split(" ")
    if len(parts) <= 2:
        return None

    return name


df['name'] = df['name'].apply(lambda x: clean_name(x))
print(df.shape)
print(df.head())

df = df.dropna(axis="rows", subset=["name"]).drop_duplicates(subset=["name"])
print(df.shape)

names = list(set(df['name'].tolist()))
print(len(names))

with open(file='legal_names.txt', encoding='utf-8', mode='w') as f:
    for name in names:
        f.write(f'{name}\n')
