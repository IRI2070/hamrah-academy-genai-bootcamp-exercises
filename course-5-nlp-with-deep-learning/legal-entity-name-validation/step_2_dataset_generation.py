import os
from typing import List
from pydantic import BaseModel
from openai import OpenAI
import json


class NegativeExample(BaseModel):
    rule: str
    example: str


class CompanyExamples(BaseModel):
    original_word: str
    hard_negative_examples: List[NegativeExample]
    hard_positive_examples: List[str]


client = OpenAI(
    api_key=os.getenv('AVAL_AI_API_KEY'),
    base_url="https://api.avalai.ir/v1"
)

with open(file='legal_names.txt', encoding='utf-8', mode='r') as f:
    words = f.read().splitlines()

print(len(words))

output_file = "company_examples.jsonl"

system_prompt = """
You are an AI system that generates hard nagetives and positive training examples for a multi-class classifier.
You will receive one keyword (brand name or company name).
Your task is to produce a JSON object with the following structure:
- "original_word": the input keyword
- "hard_negative_examples": an array of objects, each with:
   - "rule": the type of hard negative rule or no rule
   - "example": a company name generated according to that rule
- "hard_positive_examples": an array of at least 10 company names, Which does not belong to any negative examples rules.

### Negative Examples Rules:
1. synonym → استفاده از مترادف‌ها (تجارت ↔ بازرگانی، دانش ↔ علم، فولاد ↔ آهن)
2. word_order → جابجایی ترتیب کلمات (صنایع غذایی میهن ↔ میهن صنایع غذایی)
3. singular_plural → تغییر جمع/مفرد (پیشگام ↔ پیشگامان)
4. generic_word → اضافه/حذف کلمات عمومی (شرکت، گروه، هولدینگ، مهندسی)
5. domain_similarity → تغییر حوزه مشابه (غذایی ↔ خوراکی، الکترونیک ↔ الکتریک)
6. prefix_suffix → اضافه/حذف پسوند یا پیشوند (نوین، شرق، البرز)
7. activity_change → تغییر جزئی حوزه فعالیت ولی حفظ برند (غذایی ↔ لبنی، الکترونیک ↔ مخابرات)
8. adjective_removal → حذف صفت یا توصیف‌کننده از نام شرکت (صنایع غذایی میهن ↔ صنایع میهن)
9. morphological_variation → تفاوت صرفی یا تغییر کوچک در شکل واژه (غذایی ↔ غذای، میهن ↔ مهین)
10. word_removal → حذف کامل یک یا چند کلمه از نام شرکت (صنایع غذایی میهن ↔ غذایی میهن)

### Example:

Input keyword: "صنایع غذایی میهن"

Output:
```json
{
  "original_word": "صنایع غذایی میهن",
  "hard_negative_examples": [
    { "rule": "synonym", "example": "صنایع خوراکی میهن" },
    { "rule": "word_order", "example": "میهن صنایع غذایی" },
    { "rule": "singular_plural", "example": "صنعت غذایی میهن" },
    { "rule": "generic_word", "example": "شرکت صنایع غذایی میهن" },
    { "rule": "domain_similarity", "example": "صنایع خوراکی میهن" },
    { "rule": "prefix_suffix", "example": "صنایع غذایی میهن نوین" },
    { "rule": "activity_change", "example": "صنایع لبنی میهن" },
    { "rule": "adjective_removal", "example": "صنایع میهن" },
    { "rule": "morphological_variation", "example": "صنایع غذای میهن" },
    { "rule": "word_removal", "example": "غذایی میهن" }
  ],
  "hard_positive_examples": [
    "صنایع دارویی میهن",
    "شرکت سهامی صنایع نساجی میهن",
    "میهن کشت و صنعت",
    "مهندسی معدنی میهن",
    "مهندسی میهن آریا",
    "شرکت میهن نور آریا",
    "صنعت شیمیایی میهن",
    "میهن صنعت شرق",
    "مهندسی کشت و صنعت میهن",
    "میهن خوراک شرق",
  ]
}
```
"""

with open(output_file, "a", encoding="utf-8") as f:
    for i, keyword in enumerate(words):
        print(i)
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": f"Generate negative examples for keyword: {keyword}"
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "company_examples",
                        "schema": CompanyExamples.model_json_schema()
                    }
                }
            )

            raw_content = response.choices[0].message.content
            parsed_dict = json.loads(raw_content)

            examples = CompanyExamples.model_validate(parsed_dict)

            f.write(json.dumps(examples.model_dump(), ensure_ascii=False) + "\n")

            print(f"✅ ذخیره شد برای کلمه: {keyword}")
        except Exception as e:
            print('error in: ', keyword)
