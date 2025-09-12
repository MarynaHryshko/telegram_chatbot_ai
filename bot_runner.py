import subprocess
import threading
import signal
import sys
import time
import socket
from pyngrok import ngrok
import requests
from config import TELEGRAM_TOKEN
import logging
from logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

processes = []


def stream_output(proc, name):
    for line in iter(proc.stdout.readline, b''):
        if line:
            decoded_line = line.decode().rstrip()
            logger.info(f"[{name}] {decoded_line}")
    proc.stdout.close()


def start_process(name, cmd):
    logger.info(f"Starting {name} with command: {' '.join(cmd)}")
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


def check_process_health():
    """Check if all processes are still running"""
    for i, proc in enumerate(processes):
        if proc.poll() is not None:
            logger.error(f"Process {i} has died with return code {proc.returncode}")
            return False
    return True


try:
    # --- Start Redis ---
    logger.info("🟥 Starting Redis...")
    start_process("REDIS", ["redis-server"])
    time.sleep(3)

    # --- Start Celery with more verbose logging ---
    logger.info("⚡ Starting Celery worker...")
    celery_cmd = [
        "celery", "-A", "tasks", "worker",
        "--loglevel=info",
        "--concurrency=2",
        "--prefetch-multiplier=1"
    ]
    start_process("CELERY", celery_cmd)
    time.sleep(5)

    # --- Find free port for FastAPI ---
    port = find_free_port()
    logger.info(f"🚀 Starting FastAPI server on port {port}...")
    start_process("FASTAPI", ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", str(port)])
    time.sleep(3)

    # --- Start ngrok tunnel ---
    logger.info("🌐 Starting ngrok tunnel...")
    tunnel = ngrok.connect(port)
    public_url = tunnel.public_url
    logger.info(f"Ngrok public URL: {public_url}")

    # --- Set Telegram webhook ---
    logger.info("🔗 Setting Telegram webhook...")
    requests.get(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/deleteWebhook")
    r = requests.get(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setWebhook?url={public_url}/webhook")
    logger.info(f"SetWebhook response: {r.json()}")

    logger.info("✅ All services started. Send a message to your bot in Telegram!")

    # Keep runner alive and monitor processes
    while True:
        if not check_process_health():
            logger.error("One or more processes have died. Shutting down...")
            break
        time.sleep(10)

except KeyboardInterrupt:
    logger.info("🛑 Stopping all services...")

except Exception as e:
    logger.error(f"Fatal error: {e}")

finally:
    # Cleanup
    try:
        tunnel.kill()
    except:
        pass

    for proc in processes:
        try:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=5)
        except:
            proc.kill()

    logger.info("✅ All services stopped.")
    sys.exit(0)