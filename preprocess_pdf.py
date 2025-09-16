#preprocess_pdf.py
import pdfplumber
from pathlib import Path
import pytesseract
from pdf2image import convert_from_path
import json
from concurrent.futures import ThreadPoolExecutor
from kb_utils import clean_text
import logging
from logging_config import setup_logging
from config import OPENAI_API_KEY, EMBEDDING_MODEL_NAME, EMBEDDINGS_DIR
import traceback
from chromadb.utils import embedding_functions
import tiktoken
import numpy as np

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def preprocess_pdf(file_path, output_path=None):
    file_path = Path(file_path)
    output_path = output_path or file_path.with_suffix(".json")

    extracted_texts = []

    # --- Extract text from all pages ---
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if not text or looks_garbled(text):
                extracted_texts.append(None)
            else:
                extracted_texts.append(clean_text(text))

    # --- OCR only missing pages ---
    if any(t is None for t in extracted_texts):
        images = convert_from_path(file_path, dpi=200)
        def ocr_image(args):
            idx, img = args
            return clean_text(pytesseract.image_to_string(img, lang="eng"))
        with ThreadPoolExecutor() as executor:
            ocr_results = list(executor.map(ocr_image, enumerate(images)))

        for i, text in enumerate(extracted_texts):
            if text is None:
                extracted_texts[i] = ocr_results[i]

    # --- Save to JSON ---
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"file_name": file_path.name, "pages": extracted_texts}, f, ensure_ascii=False, indent=2)

    print(f"Preprocessed PDF saved to: {output_path}")

    return output_path

if __name__ == "__main__":
    import sys
    preprocess_pdf(sys.argv[1])

def embed_json(json_path):
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    file_name = json_path.stem
    embeddings_file = Path(EMBEDDINGS_DIR) / f"{file_name}.embeddings.json"

    all_chunks = []
    all_metadata = []

    for page_num, text in enumerate(data["pages"]):
        if not text:
            continue
        # chunking must be done outside or here
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        all_metadata.extend([
            {"source": data["file_name"], "page": page_num, "chunk_index": i}
            for i in range(len(chunks))
        ])

    # --- Compute embeddings ---
    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY, model_name=EMBEDDING_MODEL_NAME
    )
    vectors = embedding_function(all_chunks)

    # --- Convert NumPy arrays to lists before dumping ---
    records = []
    for text, meta, vec in zip(all_chunks, all_metadata, vectors):
        if isinstance(vec, np.ndarray):
            vec = vec.tolist()
        records.append({
            "text": text,
            "metadata": meta,
            "embedding": vec
        })

    embeddings_file.parent.mkdir(parents=True, exist_ok=True)
    with open(embeddings_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved embeddings to {embeddings_file}")
    return str(embeddings_file)


def looks_garbled(text):
    if not text: return True
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return (non_ascii / max(len(text), 1)) > 0.3

def chunk_text(text: str,
               max_tokens: int = 800,
               overlap: int = 100,
               model: str = "text-embedding-3-small"):
    """Split text into overlapping chunks with logging"""
    logger.info(f"Chunking text of length {len(text)} with max_chunk_size={max_tokens}, overlap={overlap}")

    if not text or len(text) == 0:
        logger.warning("Empty text provided for chunking")
        return []

    try:
        enc = tiktoken.encoding_for_model(model)
        tokens = enc.encode(text)

        chunks = []
        start = 0

        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = enc.decode(chunk_tokens)

            if chunk_text.strip():
                chunks.append(chunk_text)

            if end >= len(tokens):
                break

            start = end - overlap  # move back by overlap for context continuity

        logger.info(f"Text chunked successfully into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error chunking text: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


