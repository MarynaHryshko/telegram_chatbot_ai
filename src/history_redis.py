# history_redis.py
import redis
import json

# Connect to local Redis (default host/port, adjust if needed)
r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

def add_message(user_id: int, role: str, content: str, maxlen: int = 20):
    """
    Store a message for a user. Keeps only the last `maxlen` messages.
    """
    key = f"chat:{user_id}"
    entry = json.dumps({"role": role, "content": content})
    r.rpush(key, entry)
    r.ltrim(key, -maxlen, -1)  # keep last N

def get_history(user_id: int):
    """
    Retrieve conversation history for a user.
    """
    key = f"chat:{user_id}"
    data = r.lrange(key, 0, -1)
    return [json.loads(d) for d in data]

def clear_history(user_id: int):
    """Clear chat history for a user (useful if admin resets conversation)."""
    r.delete(f"chat:{user_id}")
