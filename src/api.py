from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from src.db import get_session, Chat, Message
from src.llm_client import foundry_chat
from src.rag import retrieve
import uuid

app = FastAPI()

class ChatRequest(BaseModel):
    title: str

class MessageRequest(BaseModel):
    chat_id: str
    content: str

class QuestionRequest(BaseModel):
    question: str
    chat_id: str | None = None

# Create a new chat
@app.post("/chat/")
def create_chat(chat: ChatRequest, session: Session = Depends(get_session)):
    new_chat = Chat(
        chat_id=str(uuid.uuid4()),
        title=chat.title
    )
    session.add(new_chat)
    session.commit()
    return {"chat_id": new_chat.chat_id, "title": new_chat.title}

# Get all chats
@app.get("/chats/")
def get_chats(session: Session = Depends(get_session)):
    chats = session.query(Chat).all()
    return [{"chat_id": c.chat_id, "title": c.title} for c in chats]

# Get messages for a chat
@app.get("/messages/{chat_id}")
def get_messages(chat_id: str, session: Session = Depends(get_session)):
    messages = session.query(Message).filter(Message.chat_id == chat_id).all()
    return [{"role": m.role, "content": m.content} for m in messages]

@app.delete("/chat/{chat_id}")
def delete_chat(chat_id: str, session: Session = Depends(get_session)):
    chat = session.query(Chat).filter(Chat.chat_id == chat_id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    session.delete(chat)
    session.commit()
    return {"detail": f"Chat {chat_id} deleted successfully"}

@app.post("/ask/")
def ask_question(request: QuestionRequest, session: Session = Depends(get_session)):
    # Retrieve relevant context from FAISS with citation information
    context = retrieve(request.question, k=3)  # Adjust `k` as needed
    
    # Build the prompt with context and citation information
    prompt = f"""Answer the question using the context below. Your answer should be precise, include relevant **quantitative data** (e.g., percentages, ratios, specific metrics), and cite the source for each fact. Provide direct references to numbers wherever possible.

Context (with citations):
{context}

Question:
{request.question}

Citation format: [doc_id: chunk_id]

Be sure to answer using specific figures or data points mentioned in the context. If the question involves a comparison, ensure that you include all necessary details to demonstrate that comparison clearly.
Be sure to provide **all relevant data** in the format presented in the context. If numerical values or percentages are given, make sure to **state them explicitly** in your answer.
"""
    
    # Call LLM
    messages = [{"role": "user", "content": prompt}]
    answer = foundry_chat(messages, max_tokens=500)
    
    # Save to database if chat_id provided
    if request.chat_id:
        # Save user message
        user_msg = Message(
            id=str(uuid.uuid4()),
            chat_id=request.chat_id,
            role="user",
            content=request.question
        )
        session.add(user_msg)
        
        # Save assistant message
        assistant_msg = Message(
            id=str(uuid.uuid4()),
            chat_id=request.chat_id,
            role="assistant",
            content=answer
        )
        session.add(assistant_msg)
        session.commit()
    
    return {"answer": answer, "context_used": context}  # Remove truncation
