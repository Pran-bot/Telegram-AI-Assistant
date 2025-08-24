import asyncio
from telethon import TelegramClient, errors
from telethon.sessions import StringSession
from my_telegram_client import telegram_client
import os
from dotenv import load_dotenv

load_dotenv()

class TelegramManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._client = None
        return cls._instance
    
    async def get_client(self):
        if not self._client:
            self._client = TelegramClient(
                StringSession(),  # Use in-memory session
                int(os.getenv("API_ID")),
                os.getenv("API_HASH")
            )
            await self._client.start()
        return self._client

async def send_message(contact, message):
    try:
        await telegram_client.connect()
        entity = await telegram_client.get_entity(contact)
        await telegram_client.send_message(entity, message)
        return True
    except errors.FloodWaitError as e:
        raise Exception(f"Wait {e.seconds} seconds before retrying")
    except (errors.UsernameNotOccupiedError, ValueError):
        raise Exception("Recipient not found")
    except Exception as e:
        raise Exception(f"Failed to send: {str(e)}")