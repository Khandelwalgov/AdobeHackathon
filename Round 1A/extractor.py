# main.py
import os
import fitz  # PyMuPDF
import json
from utils import extract_headings

INPUT_DIR = "input"
OUTPUT_DIR = "output"

def process_pdfs():
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(INPUT_DIR, filename)
            doc = fitz.open(pdf_path)
            title, outline = extract_headings(doc)
            output_json = {
                "title": title,
                "outline": outline
            }
            output_filename = filename.replace(".pdf", ".json")
            with open(os.path.join(OUTPUT_DIR, output_filename), "w") as f:
                json.dump(output_json, f, indent=2)

if __name__ == "__main__":
    process_pdfs()
