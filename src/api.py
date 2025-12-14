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
    context = retrieve(request.question, k=3)  # Retrieve context from past interactions
    
    # Fetch last few messages in the current chat to include memory
    if request.chat_id:
        previous_messages = session.query(Message).filter(Message.chat_id == request.chat_id).order_by(Message.created_at.desc()).limit(5).all()
        memory = "\n".join([msg.content for msg in previous_messages])
        context = f"{memory}\n\n{context}"  # Append previous chat memory to the context

    # Build the prompt with context and citation information
    prompt = f"""Aşağıdakı kontekstdən istifadə edərək sualı cavablandırın, həmişə Azərbaycan dilində. Kontekst Azərbaycan dilindədir. Cavabınız dəqiq olmalı, müvafiq **kəmiyyət məlumatlarını** (məsələn, faizlər, nisbətlər, konkret ölçülər) daxil etməli və hər bir faktın mənbəyini qeyd etməlidir. Mümkün olduğu halda, rəqəmlərə birbaşa istinad edin.

Kontekst (istinadlarla):
{context}

Sual:
{request.question}

İstinad formatı: [doc_id: chunk_id]

Kontekstdə qeyd olunan konkret rəqəmləri və ya məlumat nöqtələrini istifadə edərək cavab verməyə əmin olun. Əgər sual müqayisə ilə bağlıdırsa, bu müqayisəni aydın şəkildə göstərmək üçün bütün lazımlı detalları daxil etdiyinizdən əmin olun.
Hazırkı suala cavab verməklə yanaşı, əvvəlki cavablara və ya qarşılıqlı əlaqələrə aid olan məlumatlara da istifadə edin, söhbətin davamlılığını və ardıcıllığını qoruyun.
"""
    
    # Call LLM (Language Model) for the response
    messages = [{"role": "user", "content": prompt}]
    answer = foundry_chat(messages, max_tokens=500)
    
    # Save the user and assistant messages in the database if chat_id is provided
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
    
    return {"answer": answer, "context_used": context}  # Return the context used for the response
