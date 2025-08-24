from pydantic import BaseModel

class SendRequest(BaseModel):
    message: str
    history: list = []

class ContactRequest(BaseModel):
    name: str
    identifier: str  # Can be username or phone number