from telethon import events
import requests, re, joblib
from datetime import datetime
from my_telegram_client import telegram_client
import httpx

# Load spam model
spam_model = joblib.load("models/spam_detector1.pkl")
vectorizer = joblib.load("models/vectorizer1.pkl")

def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    return re.sub(r'\s+', ' ', text).strip()

@telegram_client.on(events.NewMessage(incoming=True))
async def handle_message(event):
    if not event.is_private:
        return

    try:
        text = preprocess(event.message.message)
        user_id = event.sender_id
        username = (await event.get_sender()).username or "NoUsername"

        # Predict spam
        X_test = vectorizer.transform([text])
        if spam_model.predict(X_test)[0] == 1:
            proba = spam_model.predict_proba(X_test)[0][1]

            # Async HTTP POST
            async with httpx.AsyncClient() as client:
                await client.post(
                    "http://localhost:8000/api/notify",
                    json={
                        "event": "new_spam",
                        "data": {
                            "user_id": str(user_id),
                            "username": username,
                            "message": event.message.message,
                            "confidence": float(proba),
                            "processed_text": text,
                            "timestamp": datetime.now().isoformat()
                        }
                    }
                )

            # Delete spam message
            await telegram_client.delete_messages(event.chat_id, event.id)

    except Exception as e:
        print(f"⚠️ Spam handler error: {e}")
