from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("groq_api_key"))

async def generate_ai_response(prompt, history=None):
    try:
        messages = [{"role": "system", "content": (
        "You are user, a human user chatting on Telegram. "
        "You must never say you are an AI, chatbot, or assistant. "
        "Reply exactly as user would in first person (I, me, my). "
        "Keep replies natural, polite, and realistic depending on the context. "
        "Do not repeat the received message, only respond meaningfully. "
        "For casual/friend messages, reply informally and friendly. "
        "Stay fully in character as user at all times."
    )}]
        
        if history:
            messages.extend(history)
            
        messages.append({"role": "user", "content": prompt})
        
        completion = client.chat.completions.create(
            messages=messages,
            model="llama3-70b-8192"
        )
        return completion.choices[0].message.content
    except Exception as e:
        raise Exception(f"AI error: {str(e)}")