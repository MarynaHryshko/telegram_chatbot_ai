# utils.py
import logging
import time
from kombu.exceptions import OperationalError
from tasks import process_user_message

from logging_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

def send_to_celery(chat_id: int, text: str, lang: str = "") -> bool:
    """Safely enqueue a Celery task with retries and Unicode handling."""
    # Clean up problematic characters
    safe_text = text.encode("utf-8", errors="ignore").decode("utf-8")

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            result = process_user_message.delay(chat_id, safe_text, lang)
            logger.info(
                f"[Webhook] Task {result.id} enqueued for chat {chat_id} (attempt {attempt})"
            )
            return True
        except OperationalError as e:
            logger.error(f"[Webhook] Broker error on attempt {attempt}: {e}")
            time.sleep(1)  # backoff before retry
        except Exception as e:
            logger.exception(f"[Webhook] Unexpected error enqueueing task: {e}")
            return False

    logger.error(f"[Webhook] Failed to enqueue message for chat {chat_id} after {max_retries} attempts")
    return False
