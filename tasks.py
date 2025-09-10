# import os, uuid
# from celery import Celery
# from openai import OpenAI
# from kb_utils import extract_text_from_pdf, get_user_kb, global_kb, retrieve_from_kbs
# from config import OPENAI_API_KEY
#
# celery_app = Celery(
#     "tasks",
#     broker="redis://localhost:6379/0",
#     backend="redis://localhost:6379/1"
# )
#
# openai_client = OpenAI(api_key=OPENAI_API_KEY)
#
# @celery_app.task
# def process_user_message(user_id, message, context_text):
#     model = "gpt-3.5-turbo" if len(context_text) < 500 else "gpt-4o-mini"
#     response = openai_client.chat.completions.create(
#         model=model,
#         messages=[{"role": "user", "content": f"User question: {message}\nContext:\n{context_text}\nAnswer:"}]
#     )
#     return response.choices[0].message.content
#
# @celery_app.task
# def process_pdf(file_path, kb_type, user_id=None):
#     kb = global_kb if kb_type=="global" else get_user_kb(user_id)
#     extract_text_from_pdf(file_path, kb)
# tasks.py
from openai import OpenAI
from kb_utils import retrieve_from_kbs

import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)


def handle_user_message_sync(user_id: int, user_message: str, context_text: str) -> str:
    """
    Process a user message immediately (without Celery).
    Returns the bot reply.
    """
    try:
        prompt = f"User question: {user_message}\nRelevant knowledge:\n{context_text}\nAnswer using knowledge if possible:"

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"❌ Error generating GPT response: {e}")
        return "⚠️ Sorry, something went wrong while generating the answer."


def process_pdf(path: str, kb_type: str, user_id: int):
    """
    Dummy function to avoid Celery calls for PDF processing in local testing.
    """
    print(f"PDF processing skipped (path={path}, kb_type={kb_type}, user_id={user_id})")
