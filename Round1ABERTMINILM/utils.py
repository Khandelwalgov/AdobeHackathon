# utils.py

import fitz  # PyMuPDF
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer


# Load SBERT model once
model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")


def extract_outline(pdf_path):
    doc = fitz.open(pdf_path)
    lines = []

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                text = " ".join(span["text"] for span in line["spans"]).strip()
                if not text or len(text) < 4:
                    continue

                font_size = max(span["size"] for span in line["spans"])
                y = line["bbox"][1]
                cap_ratio = sum(c.isupper() for c in text if c.isalpha()) / max(1, len(text))
                word_count = len(text.split())
                ends_with_punct = text[-1] in ".:;"

                lines.append({
                    "text": text,
                    "font_size": font_size,
                    "y": y,
                    "page": page_num + 1,
                    "word_count": word_count,
                    "cap_ratio": cap_ratio,
                    "ends_with_punct": ends_with_punct
                })

    if not lines:
        raise ValueError("No lines detected in PDF.")

    # ---------- Title detection ----------
    page1_top = [l for l in lines if l["page"] == 1 and l["y"] < 250]
    max_font = max((l["font_size"] for l in page1_top), default=0)
    title_candidates = [l for l in page1_top if abs(l["font_size"] - max_font) < 1 and l["word_count"] <= 12]
    title_line = title_candidates[0] if title_candidates else {"text": "Unknown Title"}
    title = title_line["text"]

    # ---------- Filter candidate headings ----------
    candidates = [
        l for l in lines
        if l["font_size"] >= 8 and
        l["word_count"] <= 15 and
        not l["ends_with_punct"]
    ]

    if not candidates:
        return title, []

    # ---------- SBERT Embeddings ----------
    texts = [l["text"] for l in candidates]
    embeddings = model.encode(texts, normalize_embeddings=True)

    # ---------- Combine features ----------
    font_sizes = np.array([[l["font_size"]] for l in candidates])
    y_positions = np.array([[l["y"] / 1000.0] for l in candidates])
    cap_ratios = np.array([[l["cap_ratio"]] for l in candidates])
    X = np.hstack([font_sizes, y_positions, cap_ratios, embeddings])

    # ---------- KMeans for H1/H2/H3 ----------
    try:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto").fit(X)
    except:
        kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto").fit(X)

    cluster_labels = kmeans.labels_

    # Sort clusters by average font size
    cluster_to_level = {}
    cluster_fonts = {}
    for i in range(kmeans.n_clusters):
        cluster_fonts[i] = np.mean([candidates[j]["font_size"] for j in range(len(candidates)) if cluster_labels[j] == i])

    sorted_clusters = sorted(cluster_fonts.items(), key=lambda x: -x[1])
    for rank, (cluster_id, _) in enumerate(sorted_clusters):
        cluster_to_level[cluster_id] = f"H{rank + 1}"

    # ---------- Final outline ----------
    outline = []
    for i, line in enumerate(candidates):
        if line["text"] == title:
            continue  # avoid duplication of title
        outline.append({
            "level": cluster_to_level[cluster_labels[i]],
            "text": line["text"],
            "page": line["page"]
        })

    # Sort by page and y-position for consistency
    outline = sorted(outline, key=lambda l: (l["page"], l["text"]))
    return title, outline
