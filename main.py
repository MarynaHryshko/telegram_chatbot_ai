from fastapi import FastAPI, Request
from config import TELEGRAM_TOKEN, ADMINS, ALLOWED_USERS
import logging
from telegram import Bot
from kb_utils import retrieve_from_kbs, get_user_kb
from tasks import process_user_message, process_pdf
from utils import send_to_celery

# Configure logging
#logging.basicConfig(level=logging.INFO)

from logging_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)


TOKEN = TELEGRAM_TOKEN
bot = Bot(token=TELEGRAM_TOKEN)

app = FastAPI()

def is_allowed(user_id): return user_id in ADMINS or user_id in ALLOWED_USERS

def is_admin(user_id): return user_id in ADMINS

@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    if "message" not in data: return {"ok": True}

    print("RAW update:", data, flush=True)

    msg = data["message"]
    user_id = msg["from"]["id"]

    logger.info(f"Message from user {user_id}: {msg}")  # <-- log message details

    print(f"Message from user {user_id}: {msg}")
    if not is_allowed(user_id):
        await bot.send_message(chat_id=user_id, text="🚫 Access denied.")
        return {"ok": True}

    # if "message" in data:
    #     chat_id = data["message"]["chat"]["id"]
    #     text = data["message"].get("text", "")
    #     reply = f"You said: {text}"
    #     await bot.send_message(chat_id=user_id, text=reply)
    #     # requests.post(
    #     #     f"https://api.telegram.org/bot{TOKEN}/sendMessage",
    #     #     json={"chat_id": chat_id, "text": reply}
    #     # )

    if "text" in msg:
        text = msg["text"][:500]
        logger.info(f"Message from user text: {text}")
        context_text = retrieve_from_kbs(text, user_id)
        logger.info(f"Message from context_text: {context_text}")
        #task = process_user_message.delay(user_id, text, context_text)
        # Offload to Celery
        logger.info(f"Calling Celery task... ")
        ok = send_to_celery(user_id, text, context_text)

        if ok:
            logger.info("Task successfully enqueued.")
            await bot.send_message(chat_id=user_id, text="⏳ Processing your question...")
        else:
            logger.error("Failed to enqueue task, notifying user.")
            await bot.send_message(chat_id=user_id, text="⚠️ Could not process your request right now.")

        # Acknowledge immediately
        # logger.info(f"Sending processing message...")
        # await bot.send_message(chat_id=user_id, text="⏳ Processing your question...")


    if "document" in msg:
        file_id = msg["document"]["file_id"]
        file_name = msg["document"]["file_name"]
        file = await bot.get_file(file_id)
        path = f"./uploads/{file_name}"
        await file.download_to_drive(path)
        kb_type = "global" if is_admin(user_id) else "user"
        # process_pdf.delay(path, kb_type, user_id)
        # await bot.send_message(chat_id=user_id, text="✅ PDF queued for processing")
        ok = send_to_celery(user_id, path, kb_type)   # or keep separate task if you prefer

        if ok:
            await bot.send_message(chat_id=user_id, text="✅ PDF queued for processing")
        else:
            await bot.send_message(chat_id=user_id, text="⚠️ Could not queue PDF for processing")


    return {"ok": True}
