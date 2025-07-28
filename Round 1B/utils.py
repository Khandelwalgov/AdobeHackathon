import os
import re
import fitz
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from llama_cpp import Llama
from typing import List, Dict, Tuple
from dataclasses import dataclass

# ── Models ────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
LLM_PATH = os.path.join("models", "tinyllama-1.1b-chat-v1.0.Q2_K.gguf")

print("Loading models...")
model = SentenceTransformer(EMBEDDING_MODEL)
llm = Llama(model_path=LLM_PATH, n_ctx=1024, n_threads=4, verbose=False)
print("Models loaded.")

# ── Data Structure ────────────────────────────────────────────────────
@dataclass
class Chunk:
    """A dataclass to hold chunk information for cleaner code."""
    text: str
    page: int
    document: str
    section_title: str = "General Information"
    score: float = 0.0

# ── Patched Few-Shot Prompts ───────────────────────────────────────────
# These examples drastically improve the reliability of the small LLM.

GENERATE_QUERIES_PROMPT = """
You are a smart assistant that generates thematic search queries based on a persona and task.
Focus on actionable themes. Output a numbered list.

### Example 1
Persona: HR professional
Task: Create and manage fillable forms for onboarding and compliance.
1. Creating interactive PDF forms
2. How to add and edit form fields
3. Distributing and collecting PDF forms
4. E-signature workflows for compliance documents
5. Securing and protecting sensitive form data

### Example 2
Persona: Food Contractor
Task: Prepare a vegetarian buffet-style dinner menu for a corporate gathering, including gluten-free items.
1. Gluten-free vegetarian main courses
2. Batch-prepped vegetable side dishes
3. Elegant vegetarian appetizers for a crowd
4. Vegan and gluten-free dessert options
5. Buffet presentation and food holding tips

### Example 3
Persona: {persona}
Task: {task}
1. """

LLM_KEYWORDS_PROMPT = """
You are a retrieval AI that lists keywords to help a search. You must follow preferences strictly. For a vegetarian, you must penalize all animal-based foods. For a budget trip, you must penalize expensive options.
Provide a comma-separated list of 5-6 items. Lowercase, no explanations.

### Example 1
Persona: HR professional
Task: Create and manage fillable forms for onboarding and compliance.
Intent: boost
Keywords: fillable form, e-signature, form field, compliance, onboarding, adobe sign

### Example 2
Persona: HR professional
Task: Create and manage fillable forms for onboarding and compliance.
Intent: penalize
Keywords: content marketing, pdf editing, page layout, graphic design, image conversion

### Example 3
Persona: Travel Planner
Task: Plan a trip of 4 days for a group of 10 college friends.
Intent: boost
Keywords: nightlife, budget-friendly, group activities, beaches, bars, affordable eats

### Example 4
Persona: Travel Planner
Task: Plan a trip of 4 days for a group of 10 college friends.
Intent: penalize
Keywords: family-friendly, luxury travel, fine dining, kids activities, historical tours, museums

### Example 5
Persona: Food Contractor
Task: Prepare a vegetarian buffet-style dinner menu for a corporate gathering, including gluten-free items.
Intent: penalize
Keywords: meat, chicken, beef, fish, pork, seafood

### Example 6
Persona: {persona}
Task: {task}
Intent: {intent}
Keywords: """

# ── LLM Helpers ────────────────────────────────────────────────────────

def generate_semantic_queries(persona: str, task: str) -> List[str]:
    """Uses a robust few-shot prompt to generate reliable search queries."""
    prompt = GENERATE_QUERIES_PROMPT.format(persona=persona, task=task)
    response = llm(prompt, max_tokens=150, stop=["###", "6."])["choices"][0]["text"]
    queries = re.findall(r'\d+\.\s*(.*)', "1. " + response)
    return [q.strip() for q in queries if q.strip()]

def llm_keywords(persona: str, task: str, intent: str) -> List[str]:
    """Uses a robust few-shot prompt to generate reliable keywords."""
    prompt = LLM_KEYWORDS_PROMPT.format(persona=persona, task=task, intent=intent)
    response = llm(prompt, max_tokens=60, stop=["\n", "###"])["choices"][0]["text"]
    return [w.strip().lower() for w in response.split(',') if w.strip()]

def llm_generate_title(chunk_text: str) -> str:
    """Generates a concise title for a text chunk as a fallback."""
    prompt = f"Summarize the main topic of this text into a 4-6 word title.\n\nText: \"{chunk_text[:400]}...\"\n\nTitle:"
    response = llm(prompt, max_tokens=20, stop=["\n"])["choices"][0]["text"].strip()
    return re.sub(r'[^a-zA-Z0-9\s]', '', response) or "Key Information"

def truncate_text(text: str, limit: int = 800) -> str:
    """Truncates text to a specified limit, trying to end on a sentence or paragraph break."""
    if len(text) <= limit:
        return text
    part = text[:limit]
    if "\n\n" in part:
        return part.rsplit("\n\n", 1)[0]
    if "." in part:
        return part.rsplit(".", 1)[0] + "."
    return part

# ── PDF Processing (Optimized & More Accurate) ────────────────────────

def extract_headings_fast(doc: fitz.Document) -> List[Dict]:
    """
    PERFORMANCE: A faster, percentile-based method to identify headings.
    ACCURACY: Ignores lines that look like regular sentences.
    """
    headings = []
    font_sizes = []
    for pno, page in enumerate(doc, start=1):
        # FIX: Use "dict" to get the block structure with lines and spans
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                if not line['spans']: continue
                text = " ".join(s['text'] for s in line['spans']).strip()
                # Headings are short and don't end with a period.
                if not text or len(text.split()) > 12 or text.endswith('.'):
                    continue
                size = round(line['spans'][0]['size'], 2)
                headings.append({'text': text, 'size': size, 'page': pno})
                font_sizes.append(size)

    if not font_sizes: return []
    
    p95 = np.percentile(font_sizes, 95)
    p85 = np.percentile(font_sizes, 85)
    
    for h in headings:
        if h['size'] >= p95: h['level'] = 'H1'
        elif h['size'] >= p85: h['level'] = 'H2'
        else: h['level'] = None
    
    final_headings = [{'text': h['text'], 'page': h['page'], 'level': h['level']} for h in headings if h['level']]
    return final_headings

def get_nearest_heading(headings: List[Dict], page_number: int) -> str:
    """Finds the most specific and recent heading for a given page."""
    candidates = [h for h in headings if h["page"] <= page_number]
    if not candidates: return "General Information"
    best_match = min(candidates, key=lambda h: (page_number - h['page'], int(h['level'][1:])))
    return best_match['text']

def extract_chunks_from_pdf(doc: fitz.Document, filename: str) -> List[Chunk]:
    """RELIABILITY: Extracts chunks block-by-block to maintain semantic coherence."""
    chunks = []
    for pno, page in enumerate(doc, start=1):
        blocks = [b[4].replace('\n', ' ').strip() for b in page.get_text("blocks") if len(b[4].split()) > 20]
        for text in blocks:
            chunks.append(Chunk(text=text, page=pno, document=filename))
    return chunks

# ── Re-ranking and Selection ──────────────────────────────────────────

def mmr_re_rank(query_emb: np.ndarray, candidates: List[Chunk], candidate_embs: np.ndarray, top_k: int, lambda_val: float) -> List[Chunk]:
    """PERFORMANCE: Accepts pre-computed embeddings to avoid re-calculating."""
    if not candidates or len(candidates) <= top_k: return candidates

    sim_to_query = cosine_similarity(query_emb.reshape(1, -1), candidate_embs)[0]
    sim_between_items = cosine_similarity(candidate_embs)
    
    selected_indices = [np.argmax(sim_to_query)]
    
    for _ in range(top_k - 1):
        remaining_indices = [i for i in range(len(candidates)) if i not in selected_indices]
        if not remaining_indices: break

        mmr_scores = [
            lambda_val * sim_to_query[i] - (1 - lambda_val) * np.max(sim_between_items[i, selected_indices])
            for i in remaining_indices
        ]
        
        best_next_idx = remaining_indices[np.argmax(mmr_scores)]
        selected_indices.append(best_next_idx)
        
    return [candidates[i] for i in selected_indices]

# ── Main Pipeline ─────────────────────────────────────────────────────

def process_collection(input_data: Dict, pdf_dir: str, top_k: int = 8) -> Dict:
    persona = input_data["persona"]["role"]
    task = input_data["job_to_be_done"]["task"]
    
    # 1. Generate reliable queries and keywords with few-shot prompts
    queries = generate_semantic_queries(persona, task)
    boost_kw = llm_keywords(persona, task, "boost")
    penal_kw = llm_keywords(persona, task, "penalize")
    print(f"Generated Queries: {queries}\nBoosting: {boost_kw}\nPenalizing: {penal_kw}")

    # 2. PERFORMANCE: Process PDFs once to get all chunks and headings
    all_chunks, headings_map = [], {}
    for doc_info in input_data["documents"]:
        filename = doc_info["filename"]
        filepath = os.path.join(pdf_dir, filename)
        if not os.path.exists(filepath): continue
        
        with fitz.open(filepath) as doc:
            headings = extract_headings_fast(doc)
            headings_map[filename] = headings
            doc_chunks = extract_chunks_from_pdf(doc, filename)
            for chunk in doc_chunks:
                chunk.section_title = get_nearest_heading(headings, chunk.page)
                all_chunks.append(chunk)
    
    # 3. PERFORMANCE: Embed all chunks ONCE
    if not all_chunks: return {"metadata": {}, "extracted_sections": [], "subsection_analysis": []}
    
    query_emb = model.encode(queries or [task]).mean(axis=0)
    all_chunk_embs = model.encode([c.text for c in all_chunks], show_progress_bar=False)
    
    # 4. Score all chunks with keyword boosting/penalizing
    sims = cosine_similarity(query_emb.reshape(1,-1), all_chunk_embs)[0]
    for i, chunk in enumerate(all_chunks):
        text_lower = chunk.text.lower()
        boost_score = 0.05 * sum(1 for kw in boost_kw if kw in text_lower)
        penalty_score = -0.1 * sum(1 for kw in penal_kw if kw in text_lower)
        chunk.score = float(sims[i] + boost_score + penalty_score)

    # 5. Re-rank top candidates with MMR for relevance and diversity
    all_chunks.sort(key=lambda x: x.score, reverse=True)
    candidate_pool = all_chunks[:100]
    candidate_indices = [i for i, c in enumerate(all_chunks) if c in candidate_pool]
    candidate_embs_pool = np.array([all_chunk_embs[i] for i in candidate_indices])
    
    top_chunks = mmr_re_rank(query_emb, candidate_pool, candidate_embs_pool, top_k=top_k, lambda_val=0.7)

    # 6. Final safety net filter for hard constraints (like "vegetarian")
    if any(kw in str(penal_kw) for kw in ["meat", "chicken", "fish", "pork", "beef", "seafood"]):
        top_chunks = [c for c in top_chunks if not any(kw in c.text.lower() for kw in penal_kw)]
        
    # 7. Build final JSON output
    extracted_sections, subsection_analysis = [], []
    for rank, chunk in enumerate(top_chunks, start=1):
        section_title = chunk.section_title
        if section_title == "General Information" or len(section_title.split()) <= 2:
            section_title = llm_generate_title(chunk.text)
        
        extracted_sections.append({
            "document": chunk.document, "section_title": section_title,
            "importance_rank": rank, "page_number": chunk.page
        })
        subsection_analysis.append({
            "document": chunk.document, "refined_text": truncate_text(chunk.text),
            "page_number": chunk.page
        })

    return {
        "metadata": {
            "input_documents": [d["filename"] for d in input_data["documents"]],
            "persona": persona, "job_to_be_done": task,
            "processing_timestamp": datetime.utcnow().isoformat()
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }