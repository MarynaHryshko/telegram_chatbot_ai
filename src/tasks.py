# tasks.py
import sys
import logging
import requests
import traceback
from celery import Celery
from celery.signals import after_setup_logger, worker_ready, worker_shutdown
from celery.utils.log import get_task_logger
from openai import OpenAI
from src.config import OPENAI_API_KEY, TELEGRAM_TOKEN, LOG_FILE
from src.logging_config import setup_logging
from src.history_redis import add_message

# At the top (after imports and TELEGRAM_TOKEN import)
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# Setup logging first
setup_logging()

# --- Celery setup ---
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

# Use Celery's task logger for better integration
logger = get_task_logger(__name__)


@after_setup_logger.connect
def setup_loggers(logger, *args, **kwargs):
    """Setup Celery logger to use our centralized logging"""

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s [%(process)d]: %(message)s")

    # File handler
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
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
def process_user_message(self, payload):
    """Process user message with history + KB context + logging"""
    task_id = self.request.id
    user_id = payload["user_id"]
    question = payload["question"]
    history = payload.get("history", [])
    kb_context = payload.get("kb_context", "")
    processing_message_id = payload.get("processing_message_id")

    logger.info(f"[TASK-{task_id}] Starting message processing for user {user_id}: {question[:100]}...")

    try:
        # Choose model depending on KB context length
        model = "gpt-3.5-turbo" if len(kb_context) < 500 else "gpt-4o-mini"
        logger.info(f"[TASK-{task_id}] Using model: {model}, kb_context length: {len(kb_context)}")

        # Build OpenAI messages: system + history + latest question with KB context
        messages = [
            {"role": "system", "content": '''You are a helpful assistant. You have access to the full conversation history with this user. 
Always remember details the user has shared, including their name, preferences, and previous topics discussed. 
If a user has told you their name, always remember and use it appropriately.'''},
            *history,
            {"role": "user", "content": f"{question}\n\nContext:\n{kb_context}"}
        ]

        logger.info(f"[TASK-{task_id}] Sending message with history for user {user_id}: {messages}...")

        # Call OpenAI API
        logger.info(f"[TASK-{task_id}] Calling OpenAI API...")
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=30
        )

        answer = response.choices[0].message.content
        logger.info(f"[TASK-{task_id}] Got OpenAI response: {answer[:100]}...")

        # Delete the processing message if we have its ID
        if processing_message_id:
            try:
                delete_response = requests.post(
                    f"{TELEGRAM_API_URL}/deleteMessage",
                    json={"chat_id": user_id, "message_id": processing_message_id},
                    timeout=10
                )
                if delete_response.status_code == 200:
                    logger.info(f"[TASK-{task_id}] Deleted processing message {processing_message_id}")
                else:
                    logger.warning(
                        f"[TASK-{task_id}] Failed to delete processing message: {delete_response.status_code}"
                    )
            except Exception as delete_error:
                logger.warning(f"[TASK-{task_id}] Error deleting processing message: {delete_error}")

        # Send final answer to Telegram
        logger.info(f"[TASK-{task_id}] Sending response to Telegram...")
        telegram_url = f"{TELEGRAM_API_URL}/sendMessage"
        telegram_data = {"chat_id": user_id, "text": answer}

        r = requests.post(telegram_url, json=telegram_data, timeout=10)
        if r.status_code == 200:
            logger.info(f"[TASK-{task_id}] Successfully sent to Telegram: {r.status_code}")
        else:
            logger.error(f"[TASK-{task_id}] Telegram API error: {r.status_code} - {r.text}")

        # Save assistant reply into Redis history
        add_message(user_id, "assistant", answer)

        logger.info(f"[TASK-{task_id}] Task completed successfully")
        return {"answer": answer, "model": model, "status": "success"}

    except Exception as e:
        error_msg = f"[TASK-{task_id}] Error processing message for user {user_id}: {str(e)}"
        logger.error(error_msg)
        logger.error(f"[TASK-{task_id}] Traceback: {traceback.format_exc()}")

        # Delete processing message on error too
        if processing_message_id:
            try:
                requests.post(
                    f"{TELEGRAM_API_URL}/deleteMessage",
                    json={"chat_id": user_id, "message_id": processing_message_id},
                    timeout=10
                )
                logger.info(f"[TASK-{task_id}] Deleted processing message after error")
            except Exception as delete_error:
                logger.warning(f"[TASK-{task_id}] Error deleting processing message after error: {delete_error}")

        # Send error message to user
        try:
            error_response = requests.post(
                f"{TELEGRAM_API_URL}/sendMessage",
                json={"chat_id": user_id, "text": "⚠️ Sorry, I couldn't process your request. Please try again."},
                timeout=10
            )
            logger.info(f"[TASK-{task_id}] Sent error message to user: {error_response.status_code}")
        except Exception as send_error:
            logger.error(f"[TASK-{task_id}] Failed to send error message: {send_error}")

        raise e  # re-raise for Celery retry


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def notify_user(self, user_id, msg):
    requests.post(
        f"{TELEGRAM_API_URL}/sendMessage",
        json={"chat_id": user_id, "text": msg},
        timeout=10
    )
    add_message(user_id, "assistant", msg)
