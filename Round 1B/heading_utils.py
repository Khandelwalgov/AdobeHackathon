import fitz  # PyMuPDF

def extract_headings_and_text(path, filename):
    doc = fitz.open(path)
    sections = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        lines = text.split("\n")
        for line in lines:
            if len(line.split()) <= 12 and line.istitle():
                sections.append({
                    "title": line.strip(),
                    "text": page.get_text().strip(),
                    "page": page_num,
                    "document": filename
                })
    return sections
