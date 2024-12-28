
from packages.chat.helpers import get_chat_memory




def getMessages(data):
  chat_memory = get_chat_memory(data["user"]+":"+data["chat_id"])
  chat_history = chat_memory.messages
  return chat_history

def formatMessages(chat_history):
  chats = []
  for msg in chat_history:
    role = "user" if msg.type == "human" else "ai"
    text = msg.content
    chats.append({"text": text, "role": role})
  return chats

def chat_messages(data):
  chat_history = getMessages(data)
  formatted_chats = formatMessages(chat_history)
  return formatted_chats



