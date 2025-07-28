# Adobe Hackathon Round 1B – Persona-Driven Document Intelligence

## Approach

- Generates semantic queries and boost/penalize keywords for each persona/task using a local TinyLlama LLM (offline).
- Extracts chunks/sections from all PDFs, assigns heading context, and computes embeddings.
- Ranks chunks using semantic similarity, keyword boosting/penalizing, and MMR for diversity.
- Outputs a ranked JSON with sections and refined sub-sections, as per challenge requirements.
- Runs entirely on CPU, no internet, and under the model size limit (ensure model is <1GB).

## Build & Run

**Build:**
```bash
docker build --platform linux/amd64 -t mysolution1b:uniqueid .
