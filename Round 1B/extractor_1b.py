import os, json, datetime
from heading_utils import extract_headings_and_text
from semantic_utils import rank_sections_by_similarity

INPUT_DIR = "input"
OUTPUT_DIR = "output"

def load_persona_job():
    with open(os.path.join(INPUT_DIR, "persona.json"), "r") as f:
        data = json.load(f)
    return data["persona"], data["job_to_be_done"]

def main():
    persona, job = load_persona_job()
    combined_query = f"{persona}. Task: {job}"

    all_sections = []
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".pdf"):
            doc_path = os.path.join(INPUT_DIR, filename)
            doc_sections = extract_headings_and_text(doc_path, filename)
            all_sections.extend(doc_sections)

    ranked = rank_sections_by_similarity(all_sections, combined_query)

    output = {
        "metadata": {
            "documents": [f for f in os.listdir(INPUT_DIR) if f.endswith(".pdf")],
            "persona": persona,
            "job_to_be_done": job,
            "timestamp": datetime.datetime.now().isoformat()
        },
        "sections": [],
        "subsection_analysis": []
    }

    for i, item in enumerate(ranked[:10]):
        output["sections"].append({
            "document": item["document"],
            "page": item["page"],
            "section_title": item["title"],
            "importance_rank": i+1
        })
        output["subsection_analysis"].append({
            "document": item["document"],
            "page": item["page"],
            "refined_text": item["text"],
            "importance_rank": i+1
        })

    with open(os.path.join(OUTPUT_DIR, "challenge1b_output.json"), "w") as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
