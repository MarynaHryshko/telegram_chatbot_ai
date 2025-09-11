import requests
from celery import Celery
from openai import OpenAI
from kb_utils import extract_text_from_pdf, get_user_kb, global_kb, retrieve_from_kbs
from config import OPENAI_API_KEY, TELEGRAM_TOKEN
import logging
from logging_config import setup_logging

from celery.signals import worker_process_init
import sys
@worker_process_init.connect
def init_worker_logging(**kwargs):
    """Reconfigure logging inside each Celery worker process."""
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("[Logging] Celery worker logging initialized")

celery_app = Celery(
    "tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1"
)

openai_client = OpenAI(api_key=OPENAI_API_KEY)
setup_logging()
logger = logging.getLogger(__name__)

@celery_app.task(autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def process_user_message(user_id, message, context_text):
    logger.info(f"[Celery] Processing message for {user_id}: {message}")
    context = f"\nContext:\n{context_text}" if context_text else ""

    model = "gpt-3.5-turbo" if len(context_text) < 500 else "gpt-4o-mini"
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"User question: {message}{context}\nAnswer:"}]
        )
        answer = response.choices[0].message.content
        logger.info(f"[Celery] Got answer: {answer[:100]}...")

        # ✅ Push result to Telegram directly
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": user_id, "text": answer}
        )
        logger.info(f"[Celery] Telegram response: {r.status_code} {r.text}")
        return {"answer": answer, "model": model}
    except Exception as e:
        logger.error(f"[Celery] Failed processing message for {user_id}: {e}")
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": user_id, "text": "⚠️ Sorry, I couldn’t process your request."}
        )
    # return {
    #     "answer": response.choices[0].message.content,
    #     "model": model,
    #     "tokens_used": response.usage.total_tokens if hasattr(response, "usage") else None
    # }


@celery_app.task
def process_pdf(file_path, kb_type, user_id=None):
    kb = global_kb if kb_type=="global" else get_user_kb(user_id)
    #extract_text_from_pdf(file_path, kb)
    chunks_added = extract_text_from_pdf(file_path, kb)
    # ✅ Push confirmation to Telegram
    msg = (
        f"✅ PDF successfully added to your personal knowledge base ({chunks_added} chunks)."
        if kb_type == "user" else
        f"✅ PDF added to the global knowledge base ({chunks_added} chunks)."
    )

    if user_id:  # global PDFs might not have a user_id
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": user_id, "text": msg}
        )

    return {"kb_type": kb_type, "user_id": user_id, "status": "success", "chunks_added": chunks_added}
    #return {"kb_type": kb_type, "user_id": user_id, "status": "success"}

# # tasks.py
# from openai import OpenAI
# from kb_utils import retrieve_from_kbs
#
# import os
# from dotenv import load_dotenv

# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# openai_client = OpenAI(api_key=OPENAI_API_KEY)
#
#
# def handle_user_message_sync(user_id: int, user_message: str, context_text: str) -> str:
#     """
#     Process a user message immediately (without Celery).
#     Returns the bot reply.
#     """
#     try:
#         prompt = f"User question: {user_message}\nRelevant knowledge:\n{context_text}\nAnswer using knowledge if possible:"
#
#         response = openai_client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "user", "content": prompt}],
#         )
#         return response.choices[0].message.content
#
#     except Exception as e:
#         print(f"❌ Error generating GPT response: {e}")
#         return "⚠️ Sorry, something went wrong while generating the answer."
#
#
# def process_pdf(path: str, kb_type: str, user_id: int):
#     """
#     Dummy function to avoid Celery calls for PDF processing in local testing.
#     """
#     print(f"PDF processing skipped (path={path}, kb_type={kb_type}, user_id={user_id})")
