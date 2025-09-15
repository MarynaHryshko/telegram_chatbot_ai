# kb_utils.py
import os, uuid, re
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from config import PERSIST_DIR, OPENAI_API_KEY, MAX_CONTEXT_CHARS
import logging
from logging_config import setup_logging
import traceback


# Setup logging
setup_logging()
logger = logging.getLogger(__name__)
# Initialize components in correct order
chroma_client = None
embedding_function = None
global_kb = None
# Per-user KB cache
user_kbs = {}

# === Vector DB Setup ===
# chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
# embedding_func = embedding_functions.OpenAIEmbeddingFunction(
#     api_key=OPENAI_API_KEY, model_name="text-embedding-3-small"
# )


# Initialize ChromaDB client
try:
    logger.info("Initializing ChromaDB client...")
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    logger.info("ChromaDB client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize ChromaDB client: {str(e)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    raise

# Initialize embedding function
try:
    logger.info("Setting up embedding function...")
    # embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    #     model_name="all-MiniLM-L6-v2"
    # )
    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY, model_name="text-embedding-3-small"
    )
    logger.info("Embedding function setup completed")
except Exception as e:
    logger.error(f"Failed to setup embedding function: {str(e)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    raise

###!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Delete old collection
chroma_client.delete_collection("global_kb")

# Initialize global knowledge base
try:
    logger.info("Initializing global knowledge base...")
    global_kb = chroma_client.get_or_create_collection(
        name="global_kb",
        embedding_function=embedding_function
    )
    kb_embedding = global_kb._embedding_function
    logger.info(f"Global KB initialized successfully. Current count: {global_kb.count()}")
    logger.info(f"Global KB initialized successfully. Embedding function: {kb_embedding}")
except Exception as e:
    logger.error(f"Failed to initialize global knowledge base: {str(e)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    raise


# def get_user_kb(user_id: int):
#     if user_id not in user_kbs:
#         user_kbs[user_id] = chroma_client.get_or_create_collection(
#             name=f"user_{user_id}", embedding_function=embedding_func
#         )
#     return user_kbs[user_id]
def get_user_kb(user_id: str):
    """Get or create user-specific knowledge base with logging"""
    kb_name = f"user_kb_{user_id}"
 ####### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    chroma_client.delete_collection(kb_name)

    logger.info(f"Getting KB for user {user_id} (collection: {kb_name})")

    try:
        user_kb = chroma_client.get_or_create_collection(
            name=kb_name,
            embedding_function=embedding_function
        )
        current_count = user_kb.count()
        logger.info(f"User KB for {user_id} retrieved successfully. Current count: {current_count}")
        return user_kb
    except Exception as e:
        logger.error(f"Failed to get/create user KB for {user_id}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


# === Helpers ===
def clean_text(text):
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'\b(\w+)\1\b', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    logger.info(f"Text is cleaned")
    return text.strip()

def chunk_text(text, chunk_size=800, overlap=100):
    # chunks = []
    # start = 0
    # while start < len(text):
    #     end = min(start + chunk_size, len(text))
    #     chunks.append(text[start:end])
    #     start += chunk_size - overlap
    # return chunks
    """Split text into overlapping chunks with logging"""
    logger.info(f"Chunking text of length {len(text)} with chunk_size={chunk_size}, overlap={overlap}")

    if not text or len(text) == 0:
        logger.warning("Empty text provided for chunking")
        return []

    try:
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()

            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)

            if end >= len(text):
                break

            # Move start position back by overlap amount for next chunk
            start = end - overlap
            if start < 0:
                start = 0

        logger.info(f"Text chunked successfully into {len(chunks)} chunks")
        return chunks

    except Exception as e:
        logger.error(f"Error chunking text: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def looks_garbled(text):
    if not text: return True
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return (non_ascii / max(len(text), 1)) > 0.3

def extract_text_from_pdf(file_path, kb, source_name=None):
    logger.info(f"Starting PDF text extraction from: {file_path}")

    # if not os.path.exists(file_path):
    #     logger.error(f"PDF file not found: {file_path}")
    #     raise FileNotFoundError(f"PDF file not found: {file_path}")

    source_name = source_name or Path(file_path).name

    try:
        logger.info(f"Opening PDF document: {file_path}")
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if not text or looks_garbled(text):
                    images = convert_from_path(file_path, first_page=page_num+1, last_page=page_num+1, dpi=300)
                    text = pytesseract.image_to_string(images[0], lang="eng")
                text = clean_text(text)
                logger.info(f"PDF text extraction completed. Total cleaned text length: {len(text)} characters")
                # kb.add(
                #     documents=chunks,
                #     metadatas=metadata,
                #     ids=chunk_ids
                # )
                # Prepare metadata

                chunks = chunk_text(text)
                metadata = [{"source": file_path, "chunk_index": i} for i in range(len(chunks))]

                # Verify KB has embedding function
                try:
                    # Test that the KB can handle embeddings by checking its configuration
                    kb_metadata = kb._embedding_function
                    if kb_metadata is None:
                        logger.error("Knowledge base does not have an embedding function configured")
                        raise RuntimeError("Knowledge base missing embedding function")
                except AttributeError:
                    logger.warning("Could not verify KB embedding function, proceeding with add operation...")

                if not chunks:
                    logger.warning("No chunks created from PDF text")
                    return 0
                logger.info(f"Adding {len(chunks)} chunks to knowledge base...")
                chunk_ids = [f"{os.path.basename(file_path)}_chunk_{i}" for i in range(len(chunks))]
                try:
                    # for chunk in chunk_text(text):
                    #  kb.add(
                    #      documents=[chunk],
                    #      ids=[str(uuid.uuid4())],
                    #      metadatas=[{"source": "pdf", "file": source_name, "page": page_num}]
                    #     )
                    chunks = chunk_text(text)
                    kb.add(
                        documents=chunks,
                        metadatas=metadata,
                        ids=chunk_ids
                    )
                    logger.info(f"Successfully added {len(chunks)} chunks to KB. New total: {kb.count()}")
                    return len(chunks)

                except Exception as e:
                    logger.error(f"Failed to add chunks to knowledge base: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    raise

    except Exception as e:
        print(f"❌ Failed to process {file_path}: {e}")
        logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

def retrieve_from_kbs(query, user_id, top_k=3):
    """Retrieve relevant context from both user and global knowledge bases with logging"""
    logger.info(f"Starting KB retrieval for user {user_id}. Query: {query[:100]}... (n_results={top_k})")
    try:
        user_kb = get_user_kb(user_id)
        # Check if KBs have content
        user_count = user_kb.count()
        global_count = global_kb.count()
        logger.info(f"KB stats - User KB: {user_count} docs, Global KB: {global_count} docs")
        if user_count == 0 and global_count == 0:
            logger.info("No documents in any knowledge base")
            return ""
        #
        # results_global = global_kb.query(query_texts=[query], n_results=top_k)
        # results_user = user_kb.query(query_texts=[query], n_results=top_k)
        #
        # docs = []
        # for d in results_global["documents"]:
        #     if d: docs.append("🌍 Global KB:\n" + d[0])
        # for d in results_user["documents"]:
        #     if d: docs.append("👤 Your KB:\n" + d[0])
        #
        # text = "\n\n".join(docs)
        # return text[:MAX_CONTEXT_CHARS]
        context_parts = []

        # Search user KB if it has content
        if user_count > 0:
            logger.info(f"Searching user KB ({user_count} documents)...")
            try:
                user_results = user_kb.query(
                    query_texts=[query],
                    n_results=min(top_k, user_count)
                )

                if user_results['documents'] and user_results['documents'][0]:
                    user_docs = user_results['documents'][0]
                    logger.info(f"Found {len(user_docs)} relevant documents in user KB")

                    context_parts.append("From your personal knowledge base:")
                    for i, doc in enumerate(user_docs[:3]):  # Limit to top 3
                        context_parts.append(f"- {doc[:200]}...")
                        logger.debug(f"User KB result {i + 1}: {doc[:50]}...")
                else:
                    logger.info("No relevant documents found in user KB")

            except Exception as e:
                logger.warning(f"Error querying user KB: {str(e)}")

        # Search global KB if it has content
        if global_count > 0:
            logger.info(f"Searching global KB ({global_count} documents)...")
            try:
                global_results = global_kb.query(
                    query_texts=[query],
                    n_results=min(top_k
                                  , global_count)
                )

                if global_results['documents'] and global_results['documents'][0]:
                    global_docs = global_results['documents'][0]
                    logger.info(f"Found {len(global_docs)} relevant documents in global KB")

                    if context_parts:  # Add separator if we already have user results
                        context_parts.append("\nFrom the global knowledge base:")
                    else:
                        context_parts.append("From the global knowledge base:")

                    for i, doc in enumerate(global_docs[:3]):  # Limit to top 3
                        context_parts.append(f"- {doc[:200]}...")
                        logger.debug(f"Global KB result {i + 1}: {doc[:50]}...")
                else:
                    logger.info("No relevant documents found in global KB")

            except Exception as e:
                logger.warning(f"Error querying global KB: {str(e)}")

        final_context = "\n".join(context_parts)
        logger.info(f"KB retrieval completed. Context length: {len(final_context)} characters")

        return final_context
    except Exception as e:
        logger.error(f"Error retrieving from knowledge bases for user {user_id}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Return empty context instead of raising to prevent breaking the main flow
        return ""
def init_global_kb(folder=Path("./knowledge_base_files")):
    folder.mkdir(exist_ok=True)
    existing = global_kb.get(include=["metadatas"])
    indexed_files = set(meta.get("file") for meta in existing["metadatas"]) if existing else set()
    for file in folder.glob("*.pdf"):
        if file.name in indexed_files: continue
        extract_text_from_pdf(str(file), global_kb, source_name=file.name)
