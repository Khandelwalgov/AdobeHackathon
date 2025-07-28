import os
import json
from datetime import datetime
from utils import process_collection

BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLLECTIONS = ["Collection 1", "Collection 2", "Collection 3"]

for collection in COLLECTIONS:
    input_path = os.path.join(BASE_DIR, collection, "challenge1b_input.json")
    pdf_dir = os.path.join(BASE_DIR, collection, "PDFs")
    output_path = os.path.join(OUTPUT_DIR, f"{collection}.json")

    with open(input_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    print(f"[INFO] Processing {collection}...")
    result = process_collection(input_data, pdf_dir)

    result["metadata"]["processing_timestamp"] = datetime.utcnow().isoformat()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[✓] Saved to {output_path}")
