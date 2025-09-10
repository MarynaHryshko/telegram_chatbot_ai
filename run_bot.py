import os
import subprocess
import requests
import time
import signal

# === CONFIG ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
BOT_APP = "main:app"   # FastAPI app inside main.py

# === START REDIS ===
print("🟥 Starting Redis...")
redis_proc = subprocess.Popen(["redis-server"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
time.sleep(2)

# === START CELERY WORKER ===
print("⚡ Starting Celery worker...")
celery_proc = subprocess.Popen(
    ["celery", "-A", "tasks", "worker", "--loglevel=info"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE
)
time.sleep(2)

# === START NGROK ===
print("🌐 Starting ngrok tunnel...")
ngrok_proc = subprocess.Popen(["ngrok", "http", "8000"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
time.sleep(3)

# === GET NGROK URL ===
resp = requests.get("http://localhost:4040/api/tunnels")
public_url = resp.json()["tunnels"][0]["public_url"]
webhook_url = f"{public_url}/webhook"
print(f"🌍 Ngrok public URL: {webhook_url}")

# === START BOT SERVER ===
print("🤖 Starting bot server...")
server_proc = subprocess.Popen(
    ["uvicorn", BOT_APP, "--host", "0.0.0.0", "--port", "8000"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE
)

# === WAIT UNTIL SERVER READY ===
for i in range(20):  # retry 20 times (~20 sec max)
    try:
        r = requests.get("http://localhost:8000/ping")
        if r.status_code == 200:
            print("✅ Bot server is ready!")
            break
    except Exception:
        pass
    time.sleep(1)
else:
    print("❌ Bot server did not start in time.")
    server_proc.kill()
    exit(1)

# === SET TELEGRAM WEBHOOK ===
print("📡 Setting Telegram webhook...")
resp = requests.post(
    f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setWebhook",
    data={"url": webhook_url}
)
print("✅ Webhook set:", resp.json())

try:
    server_proc.wait()
except KeyboardInterrupt:
    print("🛑 Shutting down...")
    for proc in [redis_proc, celery_proc, ngrok_proc, server_proc]:
        proc.send_signal(signal.SIGTERM)
