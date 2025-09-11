import subprocess
import signal
import time
import requests
from pyngrok import ngrok
from config import TELEGRAM_TOKEN

TOKEN = TELEGRAM_TOKEN

def main():
    processes = []

    try:
        # 1. Start Redis
        print("🟥 Starting Redis server...")
        redis_proc = subprocess.Popen(
            ["redis-server"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        processes.append(redis_proc)
        time.sleep(2)

        # 2. Start Celery worker (assuming you have tasks.py in project root)
        print("⚡ Starting Celery worker...")
        celery_proc = subprocess.Popen(
            ["celery", "-A", "tasks", "worker", "--loglevel=info"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        processes.append(celery_proc)
        time.sleep(2)

        # 3. Start ngrok
        print("🌍 Starting ngrok tunnel...")
        tunnel = ngrok.connect(8000)
        public_url = tunnel.public_url
        print(f"✅ Ngrok public URL: {public_url}", flush=True)

        # 4. Reset & set webhook
        print("🔗 Setting Telegram webhook...")
        requests.get(f"https://api.telegram.org/bot{TOKEN}/deleteWebhook")
        set_hook = requests.get(
            f"https://api.telegram.org/bot{TOKEN}/setWebhook?url={public_url}/webhook"
        )
        print("SetWebhook response:", set_hook.json(), flush=True)

        # 5. Start FastAPI server
        print("🚀 Starting FastAPI bot server...")
        server_proc = subprocess.Popen(
            ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        processes.append(server_proc)

        # 6. Keep running until Ctrl+C
        print("✅ Bot is running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Stopping all processes...")
        for proc in processes:
            proc.send_signal(signal.SIGTERM)
        print("✅ Clean exit.")

if __name__ == "__main__":
    main()
