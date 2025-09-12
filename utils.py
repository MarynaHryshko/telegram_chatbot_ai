import logging
from celery.exceptions import Retry
from tasks import process_user_message, process_pdf_task

logger = logging.getLogger(__name__)


def send_to_celery(user_id, message_or_path, context_or_type):
    """Send task to Celery with error handling"""
    try:
        if isinstance(context_or_type, str) and context_or_type in ["global", "user"]:
            # PDF processing
            logger.info(f"Enqueueing PDF task for user {user_id}: {message_or_path}")
            task = process_pdf_task.delay(message_or_path, context_or_type, user_id)
        else:
            # Message processing
            logger.info(f"Enqueueing message task for user {user_id}: {message_or_path[:50]}...")
            task = process_user_message.delay(user_id, message_or_path, context_or_type)

        logger.info(f"Task enqueued successfully with ID: {task.id}")
        return True

    except Exception as e:
        logger.error(f"Failed to enqueue task for user {user_id}: {e}")
        return False
