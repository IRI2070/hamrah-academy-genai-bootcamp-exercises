import json
from collections import defaultdict

input_file = "company_examples.jsonl"
output_file = "triplets.jsonl"

dataset = defaultdict(lambda: {"positives": [], "negatives": []})

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line.strip())
        anchor = record.get("original_word")

        if "hard_negative_examples" in record:
            dataset[anchor]["positives"].extend([ex["example"] for ex in record["hard_negative_examples"]])

        if "hard_positive_examples" in record:
            dataset[anchor]["negatives"].extend(record["hard_positive_examples"])

triplets = []
for anchor, values in dataset.items():
    if len(anchor.split()) < 3:
        continue

    positives = values["positives"]
    negatives = values["negatives"]

    if negatives:
        for pos in positives:
            for neg in negatives:
                triplets.append({
                    "anchor": anchor,
                    "positive": pos,
                    "negative": neg
                })

with open(output_file, "w", encoding="utf-8") as f:
    for t in triplets:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")

print(f"✅ تعداد {len(triplets)} نمونه سه‌تایی ساخته شد و در {output_file} ذخیره گردید.")
