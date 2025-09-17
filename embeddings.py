import json
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from chromadb.utils import embedding_functions
from logging_config import setup_logging
from config import EMBEDDINGS_DIR, EMBEDDING_MODEL_NAME, OPENAI_API_KEY
import tiktoken
import traceback

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def embed_json_smart(json_path: str,
                     memory_threshold_mb: float = 50.0,
                     small_batch_size: int = 100,
                     large_batch_size: int = 25,
                     skip_existing: bool = True) -> str:
    """
    Smart embedding function that chooses processing method based on file size.

    Args:
        json_path: Path to the JSON file
        memory_threshold_mb: File size threshold in MB to switch to memory-efficient mode
        small_batch_size: Batch size for standard processing (smaller files)
        large_batch_size: Batch size for memory-efficient processing (larger files)
        skip_existing: Whether to skip if embeddings already exist

    Returns:
        Path to the generated embeddings file
    """
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    # Check file size
    file_size_mb = json_path.stat().st_size / (1024 * 1024)

    print(f"📁 File size: {file_size_mb:.2f} MB")

    if file_size_mb > memory_threshold_mb:
        print(f"🔄 Large file detected (>{memory_threshold_mb}MB), using memory-efficient processing")
        return embed_json_memory_efficient(
            str(json_path),
            batch_size=large_batch_size,
            skip_existing=skip_existing
        )
    else:
        print(f"⚡ Small file detected (<={memory_threshold_mb}MB), using standard processing")
        return embed_json_standard(
            str(json_path),
            batch_size=small_batch_size,
            skip_existing=skip_existing
        )


def embed_json_standard(json_path: str, batch_size: int = 100, skip_existing: bool = True) -> str:
    """
    Optimized function to embed JSON text data with batching and caching.

    Args:
        json_path: Path to the JSON file containing page data
        batch_size: Number of chunks to process in each embedding batch
        skip_existing: Whether to skip if embeddings file already exists

    Returns:
        Path to the generated embeddings file
    """
    json_path = Path(json_path)
    file_name = json_path.stem
    embeddings_file = Path(EMBEDDINGS_DIR) / f"{file_name}.embeddings.json"

    # Skip if embeddings already exist
    if skip_existing and embeddings_file.exists():
        print(f"⏭️  Embeddings already exist: {embeddings_file}")
        return str(embeddings_file)

    # Load and validate JSON data
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        raise ValueError(f"Failed to load JSON file {json_path}: {e}")

    if "pages" not in data:
        raise ValueError("JSON file must contain 'pages' key")

    # Pre-allocate lists with estimated size
    estimated_chunks = len(data["pages"]) * 5  # rough estimate
    all_chunks = []
    all_chunks.reserve(estimated_chunks) if hasattr(all_chunks, 'reserve') else None
    all_metadata = []

    # Process pages and build chunks/metadata
    for page_num, text in enumerate(data["pages"]):
        if not text or not text.strip():  # Skip empty or whitespace-only pages
            continue

        chunks = chunk_text(text)
        if not chunks:  # Skip if chunking produced no results
            continue

        all_chunks.extend(chunks)
        # More efficient metadata creation
        page_metadata = [
            {
                "source": data["file_name"],
                "page": page_num,
                "chunk_index": i
            }
            for i in range(len(chunks))
        ]
        all_metadata.extend(page_metadata)

    if not all_chunks:
        print(f"⚠️  No text chunks found in {json_path}")
        return str(embeddings_file)

    print(f"📝 Processing {len(all_chunks)} chunks from {len(data['pages'])} pages")

    # Initialize embedding function once
    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBEDDING_MODEL_NAME
    )

    # Process embeddings in batches
    all_vectors = []
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i + batch_size]
        print(f"🔄 Processing batch {i // batch_size + 1}/{(len(all_chunks) + batch_size - 1) // batch_size}")

        try:
            batch_vectors = embedding_function(batch_chunks)
            all_vectors.extend(batch_vectors)
        except Exception as e:
            logging.error(f"Failed to generate embeddings for batch starting at {i}: {e}")
            raise

    # Prepare records efficiently
    records = []
    for text, meta, vec in zip(all_chunks, all_metadata, all_vectors):
        # Convert numpy array to list if needed (more efficient check)
        if hasattr(vec, 'tolist'):
            vec = vec.tolist()

        records.append({
            "text": text,
            "metadata": meta,
            "embedding": vec
        })

    # Ensure directory exists
    embeddings_file.parent.mkdir(parents=True, exist_ok=True)

    # Write with error handling
    try:
        with open(embeddings_file, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved {len(records)} embeddings to {embeddings_file}")
    except IOError as e:
        raise IOError(f"Failed to write embeddings file: {e}")

    return str(embeddings_file)


def embed_json_memory_efficient(json_path: str,
                                batch_size: int = 25,
                                skip_existing: bool = True) -> str:
    """
    Memory-efficient version that streams processing for very large files.

    Args:
        json_path: Path to the JSON file
        batch_size: Number of chunks to process in each batch
        skip_existing: Whether to skip if embeddings already exist

    Returns:
        Path to the generated embeddings file
    """
    json_path = Path(json_path)
    file_name = json_path.stem
    embeddings_file = Path(EMBEDDINGS_DIR) / f"{file_name}.embeddings.json"

    # Skip if embeddings already exist
    if skip_existing and embeddings_file.exists():
        print(f"⏭️  Embeddings already exist: {embeddings_file}")
        return str(embeddings_file)

    # Load JSON data
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBEDDING_MODEL_NAME
    )

    embeddings_file.parent.mkdir(parents=True, exist_ok=True)

    # Stream processing - write records as we generate them
    with open(embeddings_file, "w", encoding="utf-8") as f:
        f.write("[\n")
        first_record = True

        batch_chunks = []
        batch_metadata = []

        for page_num, text in enumerate(data["pages"]):
            if not text or not text.strip():
                continue

            chunks = chunk_text(text)
            if not chunks:
                continue

            # Add to current batch
            batch_chunks.extend(chunks)
            batch_metadata.extend([
                {"source": data["file_name"], "page": page_num, "chunk_index": i}
                for i in range(len(chunks))
            ])

            # Process batch when it reaches batch_size
            if len(batch_chunks) >= batch_size:
                vectors = embedding_function(batch_chunks)

                # Write batch records
                for text, meta, vec in zip(batch_chunks, batch_metadata, vectors):
                    if not first_record:
                        f.write(",\n")
                    first_record = False

                    record = {
                        "text": text,
                        "metadata": meta,
                        "embedding": vec.tolist() if hasattr(vec, 'tolist') else vec
                    }
                    json.dump(record, f, ensure_ascii=False)

                # Clear batch
                batch_chunks.clear()
                batch_metadata.clear()

        # Process remaining chunks
        if batch_chunks:
            vectors = embedding_function(batch_chunks)
            for text, meta, vec in zip(batch_chunks, batch_metadata, vectors):
                if not first_record:
                    f.write(",\n")
                first_record = False

                record = {
                    "text": text,
                    "metadata": meta,
                    "embedding": vec.tolist() if hasattr(vec, 'tolist') else vec
                }
                json.dump(record, f, ensure_ascii=False)

        f.write("\n]")

    print(f"✅ Saved embeddings to {embeddings_file} (streaming mode)")
    return str(embeddings_file)


# Additional utility functions for advanced file size analysis

def analyze_json_complexity(json_path: str) -> Dict[str, Any]:
    """
    Analyze JSON file to provide more intelligent routing decisions.

    Returns:
        Dictionary with file analysis metrics
    """
    json_path = Path(json_path)
    file_size_mb = json_path.stat().st_size / (1024 * 1024)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Calculate text complexity
    total_text_length = sum(len(str(page)) for page in data.get("pages", []))
    non_empty_pages = sum(1 for page in data.get("pages", []) if page and str(page).strip())
    avg_page_length = total_text_length / max(non_empty_pages, 1)

    # Estimate chunks (rough calculation)
    estimated_chunks = total_text_length // 500  # assuming ~500 chars per chunk

    analysis = {
        "file_size_mb": file_size_mb,
        "total_pages": len(data.get("pages", [])),
        "non_empty_pages": non_empty_pages,
        "total_text_length": total_text_length,
        "avg_page_length": avg_page_length,
        "estimated_chunks": estimated_chunks,
        "complexity_score": (file_size_mb * 0.3 + estimated_chunks * 0.0001 + avg_page_length * 0.00001)
    }

    return analysis


def embed_json_adaptive(json_path: str, skip_existing: bool = True) -> str:
    """
    Most intelligent routing based on multiple file characteristics.

    Args:
        json_path: Path to the JSON file
        skip_existing: Whether to skip if embeddings already exist

    Returns:
        Path to the generated embeddings file
    """
    analysis = analyze_json_complexity(json_path)

    print(f"📊 File Analysis:")
    print(f"   Size: {analysis['file_size_mb']:.2f} MB")
    print(f"   Pages: {analysis['non_empty_pages']}/{analysis['total_pages']}")
    print(f"   Estimated chunks: {analysis['estimated_chunks']}")
    print(f"   Complexity score: {analysis['complexity_score']:.3f}")

    # Adaptive thresholds based on multiple factors
    if (analysis["file_size_mb"] > 100 or
            analysis["estimated_chunks"] > 5000 or
            analysis["complexity_score"] > 50):

        print("🚀 High complexity detected - using streaming mode with small batches")
        return embed_json_memory_efficient(json_path, batch_size=10, skip_existing=skip_existing)

    elif (analysis["file_size_mb"] > 25 or
          analysis["estimated_chunks"] > 2000):

        print("⚖️  Medium complexity - using memory-efficient mode")
        return embed_json_memory_efficient(json_path, batch_size=25, skip_existing=skip_existing)

    else:
        print("⚡ Low complexity - using standard processing")
        return embed_json_standard(json_path, batch_size=100, skip_existing=skip_existing)


# Convenience wrapper - recommended main function
def embed_json(json_path: str,
               mode: str = "adaptive",
               skip_existing: bool = True,
               **kwargs) -> str:
    """
    Main embedding function with multiple processing modes.

    Args:
        json_path: Path to the JSON file
        mode: Processing mode - "adaptive", "smart", "standard", or "memory_efficient"
        skip_existing: Whether to skip if embeddings already exist
        **kwargs: Additional arguments passed to the specific processing function

    Returns:
        Path to the generated embeddings file
    """
    if mode == "adaptive":
        return embed_json_adaptive(json_path, skip_existing=skip_existing)
    elif mode == "smart":
        return embed_json_smart(json_path, skip_existing=skip_existing, **kwargs)
    elif mode == "standard":
        return embed_json_standard(json_path, skip_existing=skip_existing, **kwargs)
    elif mode == "memory_efficient":
        return embed_json_memory_efficient(json_path, skip_existing=skip_existing, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from 'adaptive', 'smart', 'standard', 'memory_efficient'")


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