from flask import Flask, request
import requests
from dotenv import load_dotenv
import os
from pyngrok import ngrok

load_dotenv()

TOKEN = os.getenv("TELEGRAM_TOKEN")
app = Flask(__name__)

@app.route("/webhook", methods=["POST"])
def webhook():
    print("Headers:", dict(request.headers), flush=True)
    print("Body:", request.data, flush=True)

    data = request.json
    print("RAW update:", data, flush=True)

    if "message" in data:
        chat_id = data["message"]["chat"]["id"]
        text = data["message"].get("text", "")
        reply = f"You said: {text}"
        requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": reply}
        )

    #return {"ok": True}
    # ✅ Explicit 200 OK response
    return {"ok": True}, 200


if __name__ == "__main__":
    # Start ngrok tunnel
    tunnel = ngrok.connect(5000)
    public_url = tunnel.public_url  # ✅ get clean https://... string
    print("Ngrok public URL:", public_url, flush=True)

    # Reset webhook
    requests.get(f"https://api.telegram.org/bot{TOKEN}/deleteWebhook")
    set_hook = requests.get(f"https://api.telegram.org/bot{TOKEN}/setWebhook?url={public_url}/webhook")
    print("SetWebhook response:", set_hook.json(), flush=True)

    # Confirm webhook info
    info = requests.get(f"https://api.telegram.org/bot{TOKEN}/getWebhookInfo").json()
    print("Webhook Info:", info, flush=True)

    # Run Flask
    #app.run(port=5000)
    app.run(host="0.0.0.0", port=5000)