# utils.py
import logging
from tasks import process_user_message, notify_user
from preprocess_pdf import preprocess_pdf
from embeddings import embed_json_smart
from pathlib import Path
from kb_utils import add_embeddings_to_kb

logger = logging.getLogger(__name__)


def send_to_celery(user_id, message_or_path, context_or_type, processing_message_id=None, kb_type='global'):
    """Send task to Celery with error handling"""
    try:
        if isinstance(context_or_type, str) and context_or_type in ["global", "user"]:
            # PDF processing
            logger.info(f"Enqueueing PDF task for user {user_id}: {message_or_path}")

            file_path = Path(message_or_path)
            # Step 1: preprocess PDF outside Celery (extract text + OCR if needed)
            logger.info(f"Preprocessing PDF: {file_path}")
            preprocessed_json = preprocess_pdf(file_path)  # returns path to JSON
            embeddings = embed_json_smart(preprocessed_json) # returns path to embeddings file
            add_embeddings_to_kb(embeddings, kb_type, user_id)
            # Call Celery task
            notify_user.delay(user_id, "✅ PDF processed")
            return True
        else:
            # Message processing
            logger.info(f"Enqueueing message task for user {user_id}: {message_or_path[:50]}...")
            task = process_user_message.delay(user_id, message_or_path, context_or_type, processing_message_id)

        logger.info(f"Task enqueued successfully with ID: {task.id}")
        return True

    except Exception as e:
        logger.error(f"Failed to enqueue task for user {user_id}: {e}")
        return False

