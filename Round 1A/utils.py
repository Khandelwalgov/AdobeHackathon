# utils.py  – revised
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
import warnings
from sklearn.exceptions import ConvergenceWarning

def extract_headings(doc):
    line_rows = []          # (text, max_font_size, page_num)
    font_sizes = []         # list[float]

    # ---------- 1. Gather line‑level text + max font size ----------
    for page_idx, page in enumerate(doc, start=1):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                text = " ".join(
                    span["text"] for span in line["spans"] if span["text"].strip()
                ).strip()
                if not text:
                    continue
                size = max(span["size"] for span in line["spans"])
                line_rows.append((text, size, page_idx))
                font_sizes.append(size)

    if not line_rows:                                 # empty doc guard
        return "Unknown Title", []

    # ---------- 2. Work out thresholds for H1/H2/H3 ----------
    uniq_sizes = sorted(set(font_sizes), reverse=True)
    headings = []

    if len(uniq_sizes) <= 3:
        # Few distinct sizes → assume top 3 are H1/H2/H3
        thresholds = uniq_sizes
        def classify(sz):
            for i, thr in enumerate(thresholds):
                if sz >= thr - 0.1:                   # small tolerance
                    return f"H{i+1}"
            return None
    else:
        # Use KMeans, but adapt cluster count to avoid the warning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            n_clusters = min(4, len(uniq_sizes))
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit(
                np.array(font_sizes).reshape(-1, 1)
            )
        centers = sorted(km.cluster_centers_.flatten(), reverse=True)

        def classify(sz):
            for i, c in enumerate(centers):
                if sz >= c - 0.5:
                    return f"H{i+1}" if i < 3 else None
            return None

    # ---------- 3. Build outline ----------
    for text, size, page in line_rows:
        level = classify(size)
        if level:
            headings.append({"level": level, "text": text, "page": page})

    # ---------- 4. Title = largest heading on first page ----------
    first_page_headings = [
        h for h in headings if h["page"] == 1 and h["level"] == "H1"
    ]
    title = first_page_headings[0]["text"] if first_page_headings else "Unknown Title"

    return title, headings
