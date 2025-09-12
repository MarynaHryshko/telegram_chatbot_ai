from fastapi import FastAPI, Request
from config import TELEGRAM_TOKEN, ADMINS, ALLOWED_USERS
import logging
from telegram import Bot
from kb_utils import retrieve_from_kbs, get_user_kb
from utils import send_to_celery
from logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

TOKEN = TELEGRAM_TOKEN
bot = Bot(token=TELEGRAM_TOKEN)

app = FastAPI()


def is_allowed(user_id):
    return user_id in ADMINS or user_id in ALLOWED_USERS


def is_admin(user_id):
    return user_id in ADMINS


@app.post("/webhook")
async def webhook(request: Request):
    try:
        data = await request.json()
        logger.info(f"Received webhook data: {data}")

        if "message" not in data:
            logger.info("No message in webhook data, returning OK")
            return {"ok": True}

        msg = data["message"]
        user_id = msg["from"]["id"]

        logger.info(f"Processing message from user {user_id}")

        if not is_allowed(user_id):
            logger.warning(f"Access denied for user {user_id}")
            await bot.send_message(chat_id=user_id, text="🚫 Access denied.")
            return {"ok": True}

        if "text" in msg:
            text = msg["text"][:500]
            logger.info(f"Text message from user {user_id}: {text[:100]}...")

            try:
                context_text = retrieve_from_kbs(text, user_id)
                logger.info(f"Retrieved context length: {len(context_text) if context_text else 0}")

                # Send to Celery
                logger.info("Sending task to Celery...")
                ok = send_to_celery(user_id, text, context_text)

                if ok:
                    logger.info("Task successfully enqueued, sending acknowledgment")
                    await bot.send_message(chat_id=user_id, text="⏳ Processing your question...")
                else:
                    logger.error("Failed to enqueue task")
                    await bot.send_message(chat_id=user_id, text="⚠️ Could not process your request right now.")

            except Exception as e:
                logger.error(f"Error processing text message: {e}")
                await bot.send_message(chat_id=user_id, text="⚠️ An error occurred while processing your message.")

        elif "document" in msg:
            try:
                file_id = msg["document"]["file_id"]
                file_name = msg["document"]["file_name"]
                logger.info(f"Document received from user {user_id}: {file_name}")

                file = await bot.get_file(file_id)
                path = f"./uploads/{file_name}"
                await file.download_to_drive(path)

                kb_type = "global" if is_admin(user_id) else "user"
                ok = send_to_celery(user_id, path, kb_type)

                if ok:
                    logger.info("PDF task successfully enqueued")
                    await bot.send_message(chat_id=user_id, text="✅ PDF queued for processing")
                else:
                    logger.error("Failed to enqueue PDF task")
                    await bot.send_message(chat_id=user_id, text="⚠️ Could not queue PDF for processing")

            except Exception as e:
                logger.error(f"Error processing document: {e}")
                await bot.send_message(chat_id=user_id, text="⚠️ An error occurred while processing your document.")

        return {"ok": True}

    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return {"ok": False, "error": str(e)}