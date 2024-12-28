from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from packages.chat import chatbot_chain
from packages.get_chat_messages import chat_messages

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
def chat_endpoint(input_data: dict):
    config = {"configurable": {"thread_id": input_data.get("chat_id")}}
    response = chatbot_chain.invoke(input_data, config)
    return response

@app.post("/chat-messages")
def get_chat_messages(input_data: dict):
    response = chat_messages(input_data)
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)