import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Access control
# List of admin user IDs (example values only)
ADMINS = [000000001]

# List of allowed user IDs (example values only)
ALLOWED_USERS = [000000002, 000000003]

# Get project root (parent of src)
BASE_DIR = Path(__file__).resolve().parent.parent
# KB storage
PERSIST_DIR = BASE_DIR /"chroma_store"
Path(PERSIST_DIR).mkdir(exist_ok=True)
UPLOADS_DIR = BASE_DIR /"uploads"
Path(UPLOADS_DIR).mkdir(exist_ok=True)
INIT_KB_PATH = BASE_DIR /"knowledge_base_files"
#Path(INIT_KB_PATH).mkdir(exist_ok=True)
LOG_DIR=BASE_DIR /"logs"
Path(LOG_DIR).mkdir(exist_ok=True)
LOG_FILE = LOG_DIR/"bot.log"

# Max chars in GPT context
MAX_CONTEXT_CHARS = 3000
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
MODEL = "gpt-4o-mini"
EMBEDDINGS_DIR =BASE_DIR /"embeddings"