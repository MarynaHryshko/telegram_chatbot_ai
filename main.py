from fastapi import FastAPI, Request
import requests
from config import TELEGRAM_TOKEN, ADMINS, ALLOWED_USERS
import logging
import subprocess
import signal
import time
import requests
from pyngrok import ngrok


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOKEN = TELEGRAM_TOKEN

app = FastAPI()

@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    print("RAW update:", data, flush=True)

    if "message" in data:
        chat_id = data["message"]["chat"]["id"]
        text = data["message"].get("text", "")
        reply = f"You said: {text}"
        requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": reply}
        )

    return {"ok": True}
