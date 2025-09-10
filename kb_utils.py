import os, uuid, re
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from config import PERSIST_DIR, OPENAI_API_KEY, MAX_CONTEXT_CHARS

# === Vector DB Setup ===
chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
embedding_func = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY, model_name="text-embedding-3-small"
)
# Delete old collection
chroma_client.delete_collection("global_kb")

# Global KB
try:
    global_kb = chroma_client.get_collection("global_kb")
except:
    global_kb = chroma_client.create_collection(
        name="global_kb", embedding_function=embedding_func
    )

# Per-user KB cache
user_kbs = {}

def get_user_kb(user_id: int):
    if user_id not in user_kbs:
        user_kbs[user_id] = chroma_client.get_or_create_collection(
            name=f"user_{user_id}", embedding_function=embedding_func
        )
    return user_kbs[user_id]

# === Helpers ===
def clean_text(text):
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'\b(\w+)\1\b', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def looks_garbled(text):
    if not text: return True
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return (non_ascii / max(len(text), 1)) > 0.3

def extract_text_from_pdf(file_path, kb, source_name=None):
    source_name = source_name or Path(file_path).name
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if not text or looks_garbled(text):
                    images = convert_from_path(file_path, first_page=page_num+1, last_page=page_num+1, dpi=300)
                    text = pytesseract.image_to_string(images[0], lang="eng")
                text = clean_text(text)
                for chunk in chunk_text(text):
                    kb.add(
                        documents=[chunk],
                        ids=[str(uuid.uuid4())],
                        metadatas=[{"source": "pdf", "file": source_name, "page": page_num}]
                    )
    except Exception as e:
        print(f"❌ Failed to process {file_path}: {e}")

def retrieve_from_kbs(query, user_id, top_k=3):
    user_kb = get_user_kb(user_id)
    results_global = global_kb.query(query_texts=[query], n_results=top_k)
    results_user = user_kb.query(query_texts=[query], n_results=top_k)

    docs = []
    for d in results_global["documents"]:
        if d: docs.append("🌍 Global KB:\n" + d[0])
    for d in results_user["documents"]:
        if d: docs.append("👤 Your KB:\n" + d[0])

    text = "\n\n".join(docs)
    return text[:MAX_CONTEXT_CHARS]

def init_global_kb(folder=Path("./knowledge_base_files")):
    folder.mkdir(exist_ok=True)
    existing = global_kb.get(include=["metadatas"])
    indexed_files = set(meta.get("file") for meta in existing["metadatas"]) if existing else set()
    for file in folder.glob("*.pdf"):
        if file.name in indexed_files: continue
        extract_text_from_pdf(str(file), global_kb, source_name=file.name)
