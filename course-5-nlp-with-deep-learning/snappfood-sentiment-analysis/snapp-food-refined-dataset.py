from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import load_dataset
from dadmatools.normalizer import Normalizer
from datasets import Dataset
from datasets import DatasetDict
from datasets import concatenate_datasets
from collections import Counter

model_name = "IRI2070/tooka-bert-large-snappfood-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

nlp = pipeline(task="sentiment-analysis", model=model, tokenizer=tokenizer, device='cuda')

dataset = load_dataset('ParsiAI/snappfood-sentiment-analysis')
print(dataset)

happy_datasets = {}
for split_name in dataset.keys():
    happy_datasets[f"{split_name}"] = dataset[split_name].filter(lambda x: x["label"] == "HAPPY")
    print(f"Filtered {split_name}_happy dataset: {happy_datasets[f'{split_name}']}")

happy_sentiment_results = {}
for split_name in happy_datasets.keys():
    comments = list(happy_datasets[split_name]['comment'])
    print(f"Performing sentiment analysis on {len(comments)} comments in the {split_name} split...")
    results = nlp(comments, batch_size=32)
    happy_sentiment_results[split_name] = results
    print(f"Number of sentiment analysis results for {split_name}: {len(results)}")
    print(f"First 5 sentiment analysis results for {split_name}:")
    print(results[:5])

refined_happy_datasets = {}

for split_name in happy_datasets.keys():
    sentiment_data = happy_sentiment_results[split_name]

    indices_to_keep = [
        i for i, result in enumerate(sentiment_data)
        if result['label'] == 'HAPPY' and result['score'] > 0.99
    ]

    refined_split = happy_datasets[split_name].select(indices_to_keep)

    refined_happy_datasets[split_name] = refined_split

    print(f"Original {split_name} happy comments: {len(happy_datasets[split_name])}")
    print(f"Refined {split_name} happy comments (score > 0.90): {len(refined_happy_datasets[split_name])}")
    print(f"First 5 comments in refined {split_name} dataset: {refined_happy_datasets[split_name]['comment'][:5]}\n")

sad_datasets = {}
for split_name in dataset.keys():
    sad_datasets[f"{split_name}"] = dataset[split_name].filter(lambda x: x["label"] == "SAD")
    print(f"Filtered {split_name}_sad dataset: {sad_datasets[f'{split_name}']}")

sad_sentiment_results = {}
for split_name in sad_datasets.keys():
    comments = list(sad_datasets[split_name]['comment'])
    print(f"Performing sentiment analysis on {len(comments)} comments in the {split_name} split...")
    results = nlp(comments, batch_size=32)
    sad_sentiment_results[split_name] = results
    print(f"Number of sentiment analysis results for {split_name}: {len(results)}")
    print(f"First 5 sentiment analysis results for {split_name}:\n{results[:5]}")

refined_sad_datasets = {}

for split_name in sad_datasets.keys():
    sentiment_data = sad_sentiment_results[split_name]

    indices_to_keep = [
        i for i, result in enumerate(sentiment_data)
        if result['label'] == 'SAD' and result['score'] > 0.97
    ]

    refined_split = sad_datasets[split_name].select(indices_to_keep)

    refined_sad_datasets[split_name] = refined_split

    print(f"Original {split_name} sad comments: {len(sad_datasets[split_name])}")
    print(f"Refined {split_name} sad comments (score > 0.99): {len(refined_sad_datasets[split_name])}")
    print(f"First 5 comments in refined {split_name} dataset: {refined_sad_datasets[split_name]['comment'][:5]}\n")

combined_datasets = {}

for split_name in happy_datasets.keys():
    happy_comments_refined = refined_happy_datasets[split_name]

    sad_comments_original = refined_sad_datasets[split_name]

    combined_split = concatenate_datasets([happy_comments_refined, sad_comments_original])

    combined_datasets[f'{split_name}'] = combined_split

    print(f"\n--- {split_name} Combined Dataset ---")
    print(f"Total comments in {split_name}: {len(combined_datasets[f'{split_name}'])}")

    label_counts = Counter(combined_datasets[f'{split_name}']['label'])
    print(f"Label distribution in {split_name}: {label_counts}")

# full cleaning
normalizer = Normalizer(
    full_cleaning=False,
    unify_chars=True,
    refine_punc_spacing=True,
    remove_extra_space=True,
    remove_puncs=True,
    remove_html=True,
    remove_stop_word=False,
    replace_email_with="<EMAIL>",
    replace_number_with=None,
    replace_url_with="",
    replace_mobile_number_with=None,
    replace_emoji_with=None,
    replace_home_number_with=None
)


def clean_comment(text):
    return normalizer.normalize(text)


cleaned_deduplicated_datasets = {}
for split_name, dataset_split in combined_datasets.items():
    cleaned_split = dataset_split.map(lambda example: {"comment": clean_comment(example["comment"])})

    df = cleaned_split.to_pandas()

    original_rows = len(df)
    deduplicated_df = df.drop_duplicates(subset=['comment'])
    removed_rows = original_rows - len(deduplicated_df)

    cleaned_deduplicated_split = Dataset.from_pandas(deduplicated_df, preserve_index=False)

    cleaned_deduplicated_datasets[split_name] = cleaned_deduplicated_split

final_dataset_dict = DatasetDict(cleaned_deduplicated_datasets)

for split_name in final_dataset_dict.keys():
    final_dataset_dict[split_name] = final_dataset_dict[split_name].shuffle(seed=42)

final_dataset_dict.save_to_disk("snappfood-refined-sentiment-dataset-v6")

loaded_dataset = DatasetDict.load_from_disk("/content/drive/MyDrive/snappfood-refined-sentiment-dataset-v6")
repo_id = "<user_name>/snappfood-refined-sentiment-dataset"

loaded_dataset.push_to_hub(repo_id, token="hf_XXXXXXXXXXXXXXXXXXXXXXXX")
