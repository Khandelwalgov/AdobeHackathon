# Adobe Hackathon Round 1A – PDF Outline Extractor

## Approach

- Uses PyMuPDF to parse PDFs and extract all line-level text and font sizes.
- Clusters font sizes (KMeans or top-3 heuristics) to detect heading levels H1, H2, H3.
- Assembles a hierarchical outline (with page numbers).
- Title is chosen as largest H1 on the first page.

## Build & Run

**Build:**
```bash
docker build --platform linux/amd64 -t mysolution:yourid .
