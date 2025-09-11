import subprocess
import threading
import signal
import sys
import time
import socket
from pyngrok import ngrok
import requests
from config import TELEGRAM_TOKEN

processes = []

def stream_output(proc, name):
    for line in iter(proc.stdout.readline, b''):
        if line:
            print(f"[{name}] {line.decode().rstrip()}")
    proc.stdout.close()

def start_process(name, cmd):
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1
    )
    processes.append(proc)
    t = threading.Thread(target=stream_output, args=(proc, name), daemon=True)
    t.start()
    return proc

def find_free_port():
    """Returns a free port on localhost"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

try:
    # --- Start Redis ---
    print("🟥 Starting Redis...")
    start_process("REDIS", ["redis-server"])
    time.sleep(2)

    # --- Start Celery ---
    print("⚡ Starting Celery worker...")
    start_process("CELERY", ["celery", "-A", "tasks", "worker", "--loglevel=info"])
    time.sleep(2)

    # --- Find free port for FastAPI ---
    port = find_free_port()
    print(f"🚀 Starting FastAPI server on port {port}...")
    start_process("FASTAPI", ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", str(port)])
    time.sleep(2)

    # --- Start ngrok tunnel ---
    print("🌐 Starting ngrok tunnel...")
    tunnel = ngrok.connect(port)
    public_url = tunnel.public_url
    print("Ngrok public URL:", public_url)

    # --- Set Telegram webhook ---
    print("🔗 Setting Telegram webhook...")
    requests.get(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/deleteWebhook")
    r = requests.get(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setWebhook?url={public_url}/webhook")
    print("SetWebhook response:", r.json())

    print("✅ All services started. Send a message to your bot in Telegram!")

    # Keep runner alive
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("🛑 Stopping all services...")
    tunnel.kill()
    for proc in processes:
        proc.send_signal(signal.SIGTERM)
    print("✅ All services stopped.")
    sys.exit(0)
