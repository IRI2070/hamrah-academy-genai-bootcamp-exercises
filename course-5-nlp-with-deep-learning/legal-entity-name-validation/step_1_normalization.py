import pandas as pd
from dadmatools.normalizer import Normalizer

df = pd.read_csv('data_sample.csv')

normalizer = Normalizer(full_cleaning=True)

df['name'] = df['name'].apply(lambda x: normalizer.normalize(x))
names = df['name'].tolist()

with open(file='legal_names.txt', encoding='utf-8', mode='w') as f:
    for name in names:
        f.write(f'{name}\n')
