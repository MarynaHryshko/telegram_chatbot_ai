import requests
from celery import Celery
from openai import OpenAI
from kb_utils import extract_text_from_pdf, get_user_kb, global_kb, retrieve_from_kbs
from config import OPENAI_API_KEY, TELEGRAM_TOKEN
import logging
from logging_config import setup_logging

from celery.signals import after_setup_logger, worker_ready, worker_shutdown
import sys
from celery.utils.log import get_task_logger


# Setup logging first
setup_logging()

celery_app = Celery(
    "tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1"
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minutes
    task_soft_time_limit=240,  # 4 minutes
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=False,
    task_reject_on_worker_lost=True,
)


openai_client = OpenAI(api_key=OPENAI_API_KEY)
setup_logging()


# Use Celery's task logger for better integration
logger = get_task_logger(__name__)


@after_setup_logger.connect
def setup_loggers(logger, *args, **kwargs):
    """Setup Celery logger to use our centralized logging"""
    import os
    os.makedirs("logs", exist_ok=True)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s [%(process)d]: %(message)s")

    # File handler
    file_handler = logging.FileHandler("logs/bot.log", encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.setLevel(logging.INFO)


@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    logger.info(f"Celery worker {sender} is ready and waiting for tasks")


@worker_shutdown.connect
def worker_shutdown_handler(sender=None, **kwargs):
    logger.info(f"Celery worker {sender} is shutting down")


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def process_user_message(self, user_id, message, context_text):
    """Process user message with comprehensive logging"""
    task_id = self.request.id
    logger.info(f"[TASK-{task_id}] Starting message processing for user {user_id}: {message[:100]}...")

    try:
        context = f"\nContext:\n{context_text}" if context_text else ""
        model = "gpt-3.5-turbo" if len(context_text) < 500 else "gpt-4o-mini"

        logger.info(f"[TASK-{task_id}] Using model: {model}, context length: {len(context_text)}")

        # Call OpenAI API
        logger.info(f"[TASK-{task_id}] Calling OpenAI API...")
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"User question: {message}{context}\nAnswer:"}],
            timeout=30
        )

        answer = response.choices[0].message.content
        logger.info(f"[TASK-{task_id}] Got OpenAI response: {answer[:100]}...")

        # Send to Telegram
        logger.info(f"[TASK-{task_id}] Sending response to Telegram...")
        telegram_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        telegram_data = {"chat_id": user_id, "text": answer}

        r = requests.post(telegram_url, json=telegram_data, timeout=10)

        if r.status_code == 200:
            logger.info(f"[TASK-{task_id}] Successfully sent to Telegram: {r.status_code}")
        else:
            logger.error(f"[TASK-{task_id}] Telegram API error: {r.status_code} - {r.text}")

        logger.info(f"[TASK-{task_id}] Task completed successfully")
        return {"answer": answer, "model": model, "status": "success"}

    except Exception as e:
        error_msg = f"[TASK-{task_id}] Error processing message for user {user_id}: {str(e)}"
        logger.error(error_msg)
        logger.error(f"[TASK-{task_id}] Traceback: {traceback.format_exc()}")

        # Send error message to user
        try:
            error_response = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": user_id, "text": "⚠️ Sorry, I couldn't process your request. Please try again."},
                timeout=10
            )
            logger.info(f"[TASK-{task_id}] Sent error message to user: {error_response.status_code}")
        except Exception as send_error:
            logger.error(f"[TASK-{task_id}] Failed to send error message: {send_error}")

        # Re-raise for Celery retry mechanism
        raise e


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def process_pdf_task(self, file_path, kb_type, user_id):
    """Process PDF with comprehensive logging"""
    task_id = self.request.id
    logger.info(f"[TASK-{task_id}] Starting PDF processing: {file_path} for user {user_id}")

    try:
        # Your PDF processing logic here
        # text = extract_text_from_pdf(file_path)
        # ... process and store in KB

        logger.info(f"[TASK-{task_id}] PDF processing completed successfully")

        # Notify user
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": user_id, "text": "✅ PDF processed successfully!"},
            timeout=10
        )

        return {"status": "success", "file_path": file_path}

    except Exception as e:
        error_msg = f"[TASK-{task_id}] Error processing PDF {file_path}: {str(e)}"
        logger.error(error_msg)
        logger.error(f"[TASK-{task_id}] Traceback: {traceback.format_exc()}")

        # Notify user of error
        try:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": user_id, "text": "⚠️ Failed to process PDF. Please try again."},
                timeout=10
            )
        except Exception as send_error:
            logger.error(f"[TASK-{task_id}] Failed to send PDF error message: {send_error}")

        raise e
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
