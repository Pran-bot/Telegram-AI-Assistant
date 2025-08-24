from telethon import TelegramClient
import os
from dotenv import load_dotenv

load_dotenv()

API_ID = int(os.getenv("API_ID", 0))
API_HASH = os.getenv("API_HASH")
SESSION_FILE = os.getenv("SESSION_FILE", "anon")

if not API_ID or not API_HASH:
    raise ValueError("API_ID and API_HASH must be set in environment variables or .env file")

telegram_client = TelegramClient(SESSION_FILE, API_ID, API_HASH)
