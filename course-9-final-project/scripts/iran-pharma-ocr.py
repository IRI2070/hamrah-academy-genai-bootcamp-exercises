import argparse
import base64
import os
from src.config import config
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv('AVALAI_API_KEY'),
    base_url=config.BASE_URL,
)

parser = argparse.ArgumentParser(description="Extract drug monographs from PDF files in a folder.")
parser.add_argument("folder_path", help="Path to the folder containing PDF files")
args = parser.parse_args()

pdf_folder = args.folder_path

system_prompt = """
You are a high-precision transcription assistant. Your task is to convert the provided PDF page into Markdown format with absolute literal accuracy. You must not change, correct, or infer even a single character. Transcribe exactly what you see, character for character, maintaining the original formatting.

The only exception is to EXCLUDE all headers and footers (such as page numbers, document titles, or chapter names at the very top or bottom of the page).

Strict Instructions:
1. Formatting & Bolding: You must maintain the exact formatting of the original text. ANYTHING that appears in bold in the PDF must be rendered as bold in your Markdown output.
   - Pay special attention to section headers such as موارد مصرف, مکانیسم اثر, موارد منع مصرف, فارماکوکینتیک, هشدارها, عوارض جانبی, تداخل‌های دارویی, نکات قابل توصیه, مقدار مصرف, اشکال دارویی and any other bolded terms.
   - Use appropriate Markdown tags (#, ##, etc.) for titles and headings exactly as they appear.
2. Literal Accuracy: Do not add, remove, or modify any words. Do not use placeholders or add explanations. Maintain the exact wording, spelling, and sequence of the original text without any corrections.
3. Start Point: You must start transcribing from the very first word at the top of the right column. This includes any fragments of sentences or leftover information from the previous page. DO NOT skip any text at the beginning of the page.
4. Column Layout (Right-to-Left): The page is organized in two columns. You must transcribe the text in this exact order: 
   a) Start from the absolute top of the Right Column and transcribe everything down to its bottom.
   b) Then, continue from the absolute top of the Left Column and transcribe everything down to its bottom.
5. Provide ONLY the transcribed content in the output.
"""

failed_files = []


def encode_pdf_to_base64(pdf_path):
    with open(pdf_path, "rb") as pdf_file:
        return base64.b64encode(pdf_file.read()).decode('utf-8')


for filename in sorted(os.listdir(pdf_folder)):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)

        print(pdf_path)

        base64_pdf = encode_pdf_to_base64(pdf_path)
        data_url = f"data:application/pdf;base64,{base64_pdf}"

        try:
            response = client.responses.create(
                model="gpt-5-mini",
                input=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text",
                             "text": "Now, convert provided pdf file into markdown"},
                            {"type": "input_file", "filename": filename, "file_data": data_url},
                        ],
                    },
                ]
            )

            output_text = response.output_text

            txt_name = os.path.splitext(filename)[0] + ".txt"
            txt_path = os.path.join(pdf_folder, txt_name)

            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(output_text)
            print(f"✅ فایل {txt_name} ساخته شد")
        except Exception as e:
            print(f"⚠️ خطا در پردازش خروجی فایل {filename}: {e}")
            failed_files.append(pdf_path)

with open(file=os.path.join(pdf_folder, 'failed_files.txt'), mode='w', encoding='utf-8') as f:
    for file_path in failed_files:
        f.write(f'{file_path}\n')
