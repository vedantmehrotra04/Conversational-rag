from fastapi import FastAPI
from pydantic import BaseModel
from app.chains.conversational_rag import conversational_rag_chain

app = FastAPI()

class ChatRequest(BaseModel):
    question: str
    session_id: str

@app.post("/chat")
def chat(request: ChatRequest):
    result = conversational_rag_chain.invoke(
       { "input": request.question
       },
       config={
           "configurable": {
               "session_id": request.session_id
           }
       }
    )

    return {
        "answer": result["answer"]
    }