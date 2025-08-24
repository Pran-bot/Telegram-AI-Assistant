import sqlite3
from typing import Dict, List, Optional, Union
from fastapi import Body, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv
from telethon import TelegramClient,events
from ai_helper import generate_ai_response
from telethon_helper import send_message
from channel_search import channel_searcher
from telegram_message_handler import TelegramMessageHandler
import joblib
from telethon.errors import SessionPasswordNeededError
from datetime import datetime
from my_telegram_client import telegram_client
import asyncio
import re
import os
import json
import bot

# Configuration
load_dotenv()
from groq import Groq

groq_client = Groq(api_key=os.getenv("groq_api_key"))
CHAT_HISTORY_FILE = "chat_history.json"
CONTACTS_FILE = "contacts.json"
SPAM_ALERTS_FILE = "spam_alerts.json"
DB_FILE = "spammers.db"  # Changed from JSON to SQLite

app = FastAPI()

PORT = os.getenv("PORT", 8000)

# Models
class Contact(BaseModel):
    name: str
    telegram_id: str
    is_favorite: bool = False
    categories: List[str] = []
    notes: str = ""

class ContactUpdate(BaseModel):
    name: Optional[str] = None
    telegram_id: Optional[str] = None
    is_favorite: Optional[bool] = None
    categories: Optional[List[str]] = None
    notes: Optional[str] = None

class SendRequest(BaseModel):
    message: str
    user_id: str = "default"
    history: Optional[List[Dict]] = None

class SpamAlert(BaseModel):
    id: Optional[int] = None
    user_id: str
    username: str
    message: str
    confidence: float
    timestamp: str
    processed_text: str


class Notification(BaseModel):
    event: str
    data: dict

class PhoneRequest(BaseModel):
    phone_number: str
    user_id: str

class VerifyCodeRequest(BaseModel):
    phone_number: str
    code: str
    phone_code_hash: str
    user_id: str

class Verify2FARequest(BaseModel):
    phone_number: str
    password: str
    user_id: str


def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS spammers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            username TEXT,
            message TEXT,
            confidence REAL,
            processed_text TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

class SpamDetector:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.load_models()

    def load_models(self):
        try:
            model_path = Path("models/spam_detector1.pkl")
            vectorizer_path = Path("models/vectorizer1.pkl")
            
            if not model_path.exists() or not vectorizer_path.exists():
                raise FileNotFoundError("Model files not found")
                
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            print("Successfully loaded spam detection models")
        except Exception as e:
            print(f"Error loading spam detection models: {e}")
            self.model = None
            self.vectorizer = None

    def preprocess_text(self, text: str) -> str:
        text = re.sub(r'[^\w\s]', '', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def detect_spam(self, message: str) -> Dict[str, Union[bool, float, str]]:
        if not self.model or not self.vectorizer:
            return {'is_spam': False, 'confidence': 0.0}
        
        processed = self.preprocess_text(message)
        vectorized = self.vectorizer.transform([processed])
        prediction = self.model.predict(vectorized)
        proba = self.model.predict_proba(vectorized)[0][1]
        
        return {
            'is_spam': bool(prediction[0]),
            'confidence': float(proba),
            'processed_text': processed
        }

spam_detector = SpamDetector()


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                pass

# Initialize the manager
manager = ConnectionManager()

telegram_handler = TelegramMessageHandler(
    client=telegram_client,
    manager=manager,
    groq_api_key=os.getenv("groq_api_key")
)



# Data Storage Functions
def load_contacts() -> Dict[str, dict]:
    try:
        if Path(CONTACTS_FILE).exists():
            with open(CONTACTS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"Error loading contacts: {e}")
        return {}

def save_contacts(contacts: Dict[str, dict]):
    try:
        with open(CONTACTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(contacts, f, indent=2)
    except Exception as e:
        print(f"Error saving contacts: {e}")

def load_chat_history() -> Dict[str, List[Dict]]:
    try:
        if Path(CHAT_HISTORY_FILE).exists():
            with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"Error loading chat history: {e}")
        return {}

def save_chat_history(history: Dict[str, List[Dict]]):
    try:
        with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Error saving chat history: {e}")

def log_spam_alert(alert: SpamAlert) -> SpamAlert:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO spammers 
        (user_id, username, message, confidence, processed_text) 
        VALUES (?, ?, ?, ?, ?)""",
        (
            alert.user_id,
            alert.username,
            alert.message,
            alert.confidence,
            alert.processed_text
        )
    )
    conn.commit()
    alert.id = cursor.lastrowid
    alert.timestamp = datetime.now().isoformat()
    conn.close()
    return alert

def save_spam_alerts(alerts: List[Dict]):
    try:
        with open(SPAM_ALERTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(alerts, f, indent=2)
    except Exception as e:
        print(f"Error saving spam alerts: {e}")

def get_spam_alerts() -> List[SpamAlert]:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM spammers ORDER BY timestamp DESC")
    alerts = [dict(alert) for alert in cursor.fetchall()]
    conn.close()
    return alerts

def block_user(user_id: str) -> bool:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM spammers WHERE user_id = ?", (user_id,))
    conn.commit()
    rows_affected = cursor.rowcount
    conn.close()
    return rows_affected > 0

def parse_send_command(text: str):
    patterns = [
        r"send message to (.+?):(.+)",
        r"send (.+?) (.+)",
        r"message (.+?) that (.+)",
        r"tell (.+?) (.+)",
        r"text (.+?) (.+)",
        r"write to (.+?) (.+)",
        r"inform (.+?) that (.+)",
        r"notify (.+?) about (.+)"
    ]
    
    for pattern in patterns:
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            return match.groups()
    
    return None

@app.websocket("/ws/telegram")
async def telegram_websocket(websocket: WebSocket):
    await manager.connect(websocket)
    
    try:
        # Send recent messages to newly connected client
        recent_messages = telegram_handler.get_recent_messages(50)
        await websocket.send_text(json.dumps({
            'type': 'message_history',
            'messages': [
                {
                    'message_id': msg[0],
                    'sender': msg[1],
                    'sender_id': msg[2],
                    'message': msg[3],
                    'timestamp': msg[4],
                    'is_outgoing': msg[5],
                    'reply_to_id': msg[6]
                }
                for msg in recent_messages
            ]
        }))
        
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Handle send reply action
            if message_data.get('action') == 'send_reply':
                message_id = int(message_data.get('message_id'))
                reply_text = message_data.get('reply')

                if message_id in telegram_handler.pending_messages:
                    pending = telegram_handler.pending_messages[message_id]

                    # Send reply via Telegram
                    try:
                        sent_message = await pending['event'].respond(reply_text)
                        
                        # Save outgoing message to database
                        telegram_handler.save_message(
                            sent_message.id,
                            "You",  # Assuming the bot is sending the reply
                            pending['sender_id'],
                            reply_text,
                            is_outgoing=True,
                            reply_to_id=message_id
                        )
                        
                        # Broadcast success to all clients
                        await manager.broadcast({
                            'type': 'reply_sent',
                            'message_id': message_id,
                            'sent_message_id': sent_message.id
                        })
                        
                        # Remove from pending messages
                        del telegram_handler.pending_messages[message_id]

                    except Exception as e:
                        await websocket.send_text(json.dumps({
                            'type': 'error',
                            'message': f'Failed to send reply: {str(e)}'
                        }))
            
            # Handle regenerate AI reply action
            elif message_data.get('action') == 'regenerate_ai':
                message_id = int(message_data.get('message_id'))
                if message_id in telegram_handler.pending_messages:
                    pending = telegram_handler.pending_messages[message_id]

                    # Generate new AI reply
                    new_ai_reply = await telegram_handler.generate_ai_reply(
                        pending['message'],
                        pending['sender_name'],
                        pending['sender_id']
                    )
                    
                    # Update the stored reply
                    telegram_handler.pending_messages[message_id]['ai_reply'] = new_ai_reply

                    # Send the updated reply to the client
                    await websocket.send_text(json.dumps({
                        'type': 'ai_reply_updated',
                        'message_id': message_id,
                        'ai_reply': new_ai_reply
                    }))
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        await websocket.close()

@app.post("/api/send-telegram-reply")
async def send_telegram_reply(request: Request):
    data = await request.json()
    message_id = data.get('message_id')
    reply = data.get('reply')
    
    if not message_id or not reply:
        raise HTTPException(status_code=400, detail="Missing message_id or reply")
    
    # This will be handled by the WebSocket connection
    return {"status": "success", "message": "Reply sent to WebSocket handler"}


@app.get("/api/messages")
async def get_messages(limit: int = 50):
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
    
    return {
        'messages': [
            {
                'message_id': msg[0],
                'sender': msg[1],
                'sender_id': msg[2],
                'message': msg[3],
                'timestamp': msg[4],
                'is_outgoing': msg[5],
                'reply_to_id': msg[6]
            }
            for msg in messages
        ]
    }

@app.post("/api/regenerate-telegram-reply")
async def regenerate_telegram_reply(request: Request):
    data = await request.json()
    message_id = data.get('message_id')
    
    if not message_id:
        raise HTTPException(status_code=400, detail="Missing message_id")
    
    # This will be handled by the WebSocket connection
    return {"status": "success", "message": "Regeneration request sent to WebSocket handler"}

@app.post("/api/search-channels")
async def search_telegram_channels(request: Request):
    try:
        data = await request.json()
        query = data.get("query", "").strip()
        
        if not query:
            raise HTTPException(
                status_code=400,
                detail="Please provide a search query"
            )
        
        return await channel_searcher.search_telegram_channels(query)
        
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid request format"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Server error: {str(e)}"
        )

# WebSocket Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return FileResponse("static/connect.html", media_type="text/html")

@app.get("/ai-chat", response_class=HTMLResponse)
async def serve_ai_chat():
    return FileResponse("static/telechat.html", media_type="text/html")

@app.post("/api/navigate")
async def navigate_to_page(request: Request):
    data = await request.json()
    page = data.get("page")
    
    if page == "ai-chat":
        return {"redirect": "/ai-chat"}
    elif page == "telechat":
        return {"redirect": "/"}
    else:
        raise HTTPException(status_code=400, detail="Invalid page requested")

# Contacts API
@app.post("/api/contacts")
async def add_contact(contact: Contact):
    contacts = load_contacts()
    if contact.name in contacts:
        raise HTTPException(status_code=400, detail="Contact already exists")
    
    contacts[contact.name] = contact.dict()
    save_contacts(contacts)
    return {"status": "success", "contact": contact.dict()}

@app.get("/api/contacts")
async def get_contacts():
    return load_contacts()

@app.get("/api/contacts/{contact_name}")
async def get_contact(contact_name: str):
    contacts = load_contacts()
    if contact_name not in contacts:
        raise HTTPException(status_code=404, detail="Contact not found")
    return contacts[contact_name]

@app.put("/api/contacts/{contact_name}")
async def update_contact(contact_name: str, update: ContactUpdate):
    contacts = load_contacts()
    if contact_name not in contacts:
        raise HTTPException(status_code=404, detail="Contact not found")
    
    for field, value in update.dict(exclude_unset=True).items():
        contacts[contact_name][field] = value
    
    if update.name:
        contacts[update.name] = contacts.pop(contact_name)
    
    save_contacts(contacts)
    return {"status": "success", "contact": contacts[update.name or contact_name]}

@app.delete("/api/contacts/{contact_name}")
async def delete_contact(contact_name: str):
    contacts = load_contacts()
    if contact_name not in contacts:
        raise HTTPException(status_code=404, detail="Contact not found")
    
    del contacts[contact_name]
    save_contacts(contacts)
    return {"status": "success", "message": "Contact deleted"}

# API Endpoints (updated)
@app.post("/api/notify")
async def handle_notification(notification: Notification):
    try:
        if notification.event == "new_spam":
            alert = SpamAlert(**notification.data)
            logged_alert = log_spam_alert(alert)
            await manager.broadcast({
                "event": "new_spam",
                "data": logged_alert.dict()
            })
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/spam-alerts")
async def get_spam_alerts_endpoint():
    return {"alerts": get_spam_alerts()}

@app.post("/api/block/{user_id}")
async def block_user_endpoint(user_id: str):
    if block_user(user_id):
        await manager.broadcast({
            "event": "user_blocked",
            "data": {"user_id": user_id}
        })
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="User not found")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Message Handling
@app.post("/api/send")
async def handle_send(request: Request):
    data = await request.json()
    message = data.get("message")
    user_id = data.get("user_id", "default")
    
    try:
        # Spam detection
        spam_result = spam_detector.detect_spam(message)
        if spam_result['is_spam'] and spam_result['confidence'] > 0.7:
            alert = SpamAlert(
                user_id=user_id,
                username=f"user_{user_id}",
                message=message,
                confidence=spam_result['confidence'],
                processed_text=spam_result['processed_text']
            )
            logged_alert = log_spam_alert(alert)
            await manager.broadcast({
                "event": "new_spam",
                "data": logged_alert.dict()
            })
            return {
                "type": "spam_detected",
                "content": "Your message was flagged as potential spam and not delivered."
            }
        
        if "find channel to download" in message.lower():
            query = message.lower().replace("find channel to download", "").strip()
            if query:
                return await search_telegram_channels(Request(
                    scope={"type": "http"},
                    receive=None,
                    send=None
                ))

        # Load existing history
        all_history = load_chat_history()
        user_history = all_history.get(user_id, [])
        
        # Check if it's a send command
        parsed = None
        if message and isinstance(message, str) and message.lower().startswith(('send ', 'message ', 'tell ', 'text ', 'write ', 'inform ', 'notify ')):
            parsed = parse_send_command(message)
        
        if parsed:
            contact, msg = parsed
            contacts = load_contacts()
            if contact not in contacts:
                return {
                    "type": "ai_response",
                    "content": f"Contact '{contact}' not found. Please add them first."
                }
            
            result = await send_message(contacts[contact]["telegram_id"], msg)
            
            if result:
                response = {
                    "type": "send_result",
                    "success": True,
                    "message": f"📨 Message sent to {contact}",
                    "content": msg
                }
            else:
                response = {
                    "type": "ai_response",
                    "content": f"Failed to send message to {contact}. Please try again."
                }
        else:
            response_content = await generate_ai_response(message, user_history)
            response = {
                "type": "ai_response",
                "content": response_content
            }
        
        # Update and save history
        user_history.extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": response["content"]}
        ])
        all_history[user_id] = user_history[-1000:]
        save_chat_history(all_history)
        
        return response
            
    except Exception as e:
         raise HTTPException(status_code=400, detail=str(e))

@app.on_event("startup")
async def startup_event():
    await telegram_client.connect()
    if await telegram_client.is_user_authorized():
        print("✅ Telegram client authorized")
        # Start the handler in the background
        asyncio.create_task(telegram_handler.start())
    else:
        print("⚠️ Telegram client not authorized. Login via frontend.")

@app.get("/api/auth/check")
async def auth_check():
    if not telegram_client.is_connected():
        await telegram_client.connect()
    if await telegram_client.is_user_authorized():
            print("Already logged in, no need to request code again.")
            return {"is_authenticated": True}
    return {"is_authenticated": False}

@app.post("/api/auth/request-code")
async def request_code(data: PhoneRequest):
    try:
        await telegram_client.connect()
        result = await telegram_client.send_code_request(data.phone_number)
        return {"phone_code_hash": result.phone_code_hash}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/auth/logout")
async def logout():
    try:
        if telegram_client.is_connected():
            await telegram_client.log_out()
            await telegram_client.disconnect()

        # Remove the session file
        session_file = os.getenv("SESSION_FILE")
        if session_file and os.path.exists(session_file):
            os.remove(session_file)
        
        return {"status": "success", "message": "Logged out successfully and deleted all data."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Logout failed: {str(e)}")

@app.post("/api/auth/verify-code")
async def verify_code(data: dict = Body(...)):
    phone = data.get("phone_number")
    code = data.get("code")
    phone_code_hash = data.get("phone_code_hash")

    client = telegram_client
    await client.connect()

    try:
        await client.sign_in(phone=phone, code=code, phone_code_hash=phone_code_hash)
        return JSONResponse({"message": "Logged in successfully"}, status_code=200)
    except Exception as e:
        return JSONResponse({"message": str(e)}, status_code=400)
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=PORT)