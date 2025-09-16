# kb_utils.py
import json
import re
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from config import PERSIST_DIR, OPENAI_API_KEY, EMBEDDING_MODEL_NAME#, TELEGRAM_TOKEN, MAX_CONTEXT_CHARS
import logging
from logging_config import setup_logging
import traceback
import tiktoken


# Setup logging
setup_logging()
logger = logging.getLogger(__name__)
# Initialize components in correct order
chroma_client = None
embedding_function = None
global_kb = None
# Per-user KB cache
user_kbs = {}


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
    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY, model_name=EMBEDDING_MODEL_NAME
    )
    logger.info("Embedding function setup completed")
except Exception as e:
    logger.error(f"Failed to setup embedding function: {str(e)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    raise

# Delete old collection
#chroma_client.delete_collection("global_kb")

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


def get_user_kb(user_id: str):
    """Get or create user-specific knowledge base with logging"""
    kb_name = f"user_kb_{user_id}"

 #   chroma_client.delete_collection(kb_name)

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


def add_embeddings_to_kb(embeddings_file, kb_type, user_id, source_name=None):
     with open(embeddings_file, "r", encoding="utf-8") as f:
         data = json.load(f)

     kb = global_kb if kb_type == "global" else get_user_kb(user_id)

     ids = [f"{d['metadata']['source']}_p{d['metadata']['page']}_c{d['metadata']['chunk_index']}" for d in data]
     texts = [d["text"] for d in data]
     metadatas = [d["metadata"] for d in data]
     embeddings = [d["embedding"] for d in data]
     logger.info(f"Adding {len(texts)} chunks from {embeddings_file} to KB")
     kb.add(
         ids=ids,
         documents=texts,
         metadatas=metadatas,
         embeddings=embeddings
     )
     logger.info(f"✅ Added {len(ids)} docs. Collection now has {kb.count()} docs.")

# def process_pdf_json(preprocessed_json_path, kb_type, user_id):
#     """
#     Fully safe: reads preprocessed JSON, computes embeddings, adds to KB,
#     and sends Telegram notification. No Celery heavy tasks needed.
#     """
#     try:
#         # --- Load preprocessed JSON ---
#         with open(preprocessed_json_path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#
#         all_chunks = []
#         all_metadata = []
#         all_ids = []
#
#         for page_num, text in enumerate(data["pages"]):
#             if not text:
#                 continue
#             chunks = chunk_text(text)
#             all_chunks.extend(chunks)
#             all_metadata.extend([{"source": data["file_name"], "page": page_num, "chunk_index": i}
#                                  for i in range(len(chunks))])
#             all_ids.extend([f"{data['file_name']}_p{page_num}_c{i}" for i in range(len(chunks))])
#
#         if not all_chunks:
#             logger.warning(f"No text found in {preprocessed_json_path}")
#             return 0
#
#         # --- Compute embeddings (safe in main thread) ---
#         embedding_function = embedding_functions.OpenAIEmbeddingFunction(
#             api_key=OPENAI_API_KEY,
#             model_name=EMBEDDING_MODEL_NAME
#         )
#         embeddings = embedding_function(all_chunks)
#
#         # --- Add to KB ---
#         kb = global_kb if kb_type == "global" else get_user_kb(user_id)
#         kb.add(
#             ids=all_ids,
#             documents=all_chunks,
#             metadatas=all_metadata,
#             embeddings=embeddings
#         )
#
#         chunks_added = len(all_chunks)
#         logger.info(f"✅ Added {chunks_added} chunks from {data['file_name']} to KB")
#
#         # --- Notify user on Telegram ---
#         msg = (f"✅ PDF successfully added to your personal KB ({chunks_added} chunks)."
#                if kb_type == "user" else
#                f"✅ PDF added to the global KB ({chunks_added} chunks).")
#         requests.post(
#             f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
#             json={"chat_id": user_id, "text": msg},
#             timeout=10
#         )
#
#         return chunks_added
#
#     except Exception as e:
#         logger.error(f"Failed to process PDF JSON {preprocessed_json_path}: {e}")
#         logger.error(traceback.format_exc())
#         # Notify user of error
#         try:
#             requests.post(
#                 f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
#                 json={"chat_id": user_id, "text": "⚠️ Failed to process PDF. Please try again."},
#                 timeout=10
#             )
#         except Exception as send_error:
#             logger.error(f"Failed to send Telegram error message: {send_error}")
#         return 0

# def test_kb_add(kb):
#     try:
#         # Test with minimal data
#         logger.info("Starting KB add test ...")
#         test_chunks = ["test document"]
#         test_metadata = [{"source": "test", "page": 0, "chunk_index": 0}]
#         test_ids = ["test_id_1"]
#
#         kb.add(documents=test_chunks, metadatas=test_metadata, ids=test_ids)
#         logger.info("KB add test successful")
#         return True
#     except Exception as e:
#         logger.error(f"KB add test failed: {e}")
#         return False


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


def init_global_kb(folder=Path("./knowledge_base_files")):
    folder.mkdir(exist_ok=True)
    existing = global_kb.get(include=["metadatas"])
    indexed_files = set(meta.get("file") for meta in existing["metadatas"]) if existing else set()
    for file in folder.glob("*.pdf"):
        if file.name in indexed_files: continue
        # extract_text_from_pdf(str(file), global_kb, source_name=file.name)
