from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_name = "IRI2070/tooka-bert-large-snappfood-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

nlp = pipeline("text-classification", model=model, tokenizer=tokenizer)

print(nlp("غذا شور بود"))  # [{'label': 'SAD', 'score': 0.9999865293502808}]
print(nlp("شیرینی تازه بود"))  # [{'label': 'HAPPY', 'score': 0.9999866485595703}]
