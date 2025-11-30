import json
import csv

input_file = "company_examples.jsonl"
output_file = "data_unique.csv"

seen = set()

with open(output_file, "w", newline="", encoding="utf-8") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["original_word", "example", "rule"])

    with open(input_file, "r", encoding="utf-8") as infile:
        for line in infile:
            record = json.loads(line.strip())
            original_word = record["original_word"]

            for neg in record.get("hard_negative_examples", []):
                row = (original_word, neg["example"], neg["rule"])
                if row not in seen:
                    seen.add(row)
                    writer.writerow(row)

            for pos in record.get("hard_positive_examples", []):
                row = (original_word, pos, "no_rule")
                if row not in seen:
                    seen.add(row)
                    writer.writerow(row)
