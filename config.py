import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Access control
ADMINS = [1255890632]
ALLOWED_USERS = [111111111, 222222222]

# KB storage
PERSIST_DIR = "./chroma_store"
Path(PERSIST_DIR).mkdir(exist_ok=True)
UPLOADS_DIR = "./uploads"
Path(UPLOADS_DIR).mkdir(exist_ok=True)
INIT_KB_PATH = "./knowledge_base_files"

# Max chars in GPT context
MAX_CONTEXT_CHARS = 3000
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")