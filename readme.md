# Telegram AI Bot with Knowledge Base and Redis History

This project is a Telegram chatbot powered by FastAPI. It supports both text and document (PDF) inputs, integrates with multiple knowledge bases (KBs), and uses Redis to store user chat history. Tasks are processed asynchronously via Celery.

---

## Features

- ✅ Telegram bot integration
- ✅ FastAPI webhook for receiving messages
- ✅ Text message processing with context retrieval from KBs
- ✅ PDF/document processing and queueing for knowledge base updates
- ✅ User access control (admins vs allowed users)
- ✅ Chat history stored in Redis
- ✅ Async task handling with Celery
- ✅ Logging for debugging and monitoring

---

## Requirements

- Python 3.10+
- Redis
- Celery
- Telegram Bot API token

Python dependencies (example):

```txt
fastapi
uvicorn
python-telegram-bot
redis
celery
```

## Setup

- Clone the repository: 
```
git clone <repo-url>
cd <repo-name>
```
- Install dependencies
```
pip install -r requirements.txt
```
- Configure environment

Edit src/config.py with:
``` 
TELEGRAM_TOKEN – your Telegram bot token
ADMINS – list of admin user IDs
ALLOWED_USERS – list of allowed user IDs
UPLOADS_DIR – directory for uploaded documents
```

run bot_runner.py
