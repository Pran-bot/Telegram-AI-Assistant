import asyncio
import json
from datetime import datetime
from telethon import TelegramClient, events
from my_telegram_client import telegram_client
from groq import Groq
import os
import sqlite3
from typing import Dict

class TelegramMessageHandler:
    def __init__(self, client: TelegramClient, manager, groq_api_key=None):
        self.client = client
        self.manager = manager
        # self.websocket_manager = websocket_manager  # ConnectionManager from FastAPI
        self.groq_client = Groq(api_key=groq_api_key) if groq_api_key else None
        self.pending_messages = {}

        # Initialize database
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect('telegram_messages.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                telegram_id INTEGER UNIQUE,
                sender_name TEXT,
                sender_id INTEGER,
                message TEXT,
                timestamp DATETIME,
                is_outgoing BOOLEAN DEFAULT FALSE,
                reply_to_id INTEGER DEFAULT NULL
            )
        ''')
        conn.commit()
        conn.close()

    def save_message(self, telegram_id: int, sender_name: str, sender_id: int, 
                    message: str, is_outgoing: bool = False, reply_to_id: int = None):
        conn = sqlite3.connect('telegram_messages.db')
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO messages 
                (telegram_id, sender_name, sender_id, message, timestamp, is_outgoing, reply_to_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (telegram_id, sender_name, sender_id, message, datetime.now(), is_outgoing, reply_to_id))
            conn.commit()
        except Exception as e:
            print(f"Database error: {e}")
        finally:
            conn.close()


    def get_conversation_context(self, sender_id: int, limit: int = 10):
        """Get recent conversation context for AI"""
        conn = sqlite3.connect('telegram_messages.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT sender_name, message, is_outgoing, timestamp
            FROM messages 
            WHERE sender_id = ?
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (sender_id, limit))
        messages = cursor.fetchall()
        conn.close()
        return list(reversed(messages))  # Reverse to get chronological order

    def get_recent_messages(self, limit: int = 50):
        conn = sqlite3.connect('telegram_messages.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT telegram_id, sender_name, sender_id, message, timestamp, is_outgoing, reply_to_id
            FROM messages 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        messages = cursor.fetchall()
        conn.close()
        return messages

    async def generate_ai_reply(self, message: str, sender_name: str, sender_id: int) -> str:
        """Generate AI reply using Groq with conversation context"""
        if not self.groq_client:
            return f"I received your message - '{message}', I will message you later."
        
        try:
            # Get conversation context
            context_messages = self.get_conversation_context(sender_id, 10)
            
            # Build conversation history for AI
            conversation_history = []
            conversation_history.append({
                "role": "system",
                "content": "You are user, a human user chatting on Telegram. "
                "You must never say you are an AI, chatbot, or assistant. "
                "Reply exactly as user would in first person (I, me, my). "
                "Keep replies natural, polite, and realistic depending on the context. "
                "Do not repeat the received message, only respond meaningfully. "
                "For professional messages (like job offers or interviews), reply formally. "
                "For casual/friend messages, reply informally and friendly. "
                "Stay fully in character as user at all times."
            })
            
            # Add conversation context
            for ctx_sender, ctx_message, ctx_is_outgoing, ctx_timestamp in context_messages:
                if ctx_is_outgoing:
                    conversation_history.append({
                        "role": "assistant",
                        "content": ctx_message
                    })
                else:
                    conversation_history.append({
                        "role": "user",
                        "content": f"{ctx_sender}: {ctx_message}"
                    })
            
            # Add current message
            conversation_history.append({
                "role": "user",
                "content": f"{sender_name}: {message}"
            })
            
            chat_completion = self.groq_client.chat.completions.create(
                messages=conversation_history,
                model="llama3-8b-8192",
                temperature=0.7,
                max_tokens=150
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Groq API error: {e}")
            return f"I received your message - '{message}', I will message you later."    

    async def start(self):
        self.client = telegram_client
        if not await self.client.is_user_authorized():
            print("Telegram client not authorized.")
            return

        @self.client.on(events.NewMessage(incoming=True))
        async def handler(event):
            sender = await event.get_sender()
            sender_name = sender.username or sender.first_name
            sender_id = sender.id
            message_text = event.message.message
            message_id = event.message.id

            # Save to DB
            self.save_message(message_id, sender_name, sender_id, message_text, is_outgoing=False)

            # Generate AI reply
            ai_reply = await self.generate_ai_reply(message_text, sender_name, sender_id)

            # Store pending messages
            self.pending_messages[message_id] = {
                'event': event,
                'sender_name': sender_name,
                'sender_id': sender_id,
                'message': message_text,
                'ai_reply': ai_reply
            }

            # broadcast to frontend
            if self.manager:
                await self.manager.broadcast({
                    "event": "new_message",
                    "data": {
                        "message_id": message_id,
                        "sender_name": sender_name,
                        "sender_id": sender_id,
                        "message": message_text,
                        "ai_reply": ai_reply,
                        "timestamp": datetime.now().isoformat(),
                        "is_outgoing": False
                    }
                })

            print(f"New message from {sender_name}: {message_text}")
            print(f"AI Reply: {ai_reply}")

        print("Telegram handler started. Listening for new messages...")
        asyncio.create_task(self.client.run_until_disconnected())

    async def stop(self):
        if self.client:
            await self.client.disconnect()
