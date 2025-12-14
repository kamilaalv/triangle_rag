from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from src.db import get_session, Chat, Message
from src.llm_client import foundry_chat
from src.rag import retrieve_with_metadata
import uuid

app = FastAPI()

class ChatRequest(BaseModel):
    title: str

class MessageRequest(BaseModel):
    chat_id: str
    content: str

class ChatMessage(BaseModel):
    role: str
    content: str

class LLMRequest(BaseModel):
    messages: list[dict[str, str]]

class SourceReference(BaseModel):
    pdf_name: str
    page_number: int
    content: str

class LLMResponse(BaseModel):
    sources: list[SourceReference]
    answer: str

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

@app.post("/llm/", response_model=LLMResponse)
def llm_endpoint(request: LLMRequest):
    """
    LLM endpoint that receives chat history and produces an answer with source references.
    Handles follow-up questions by including previous context in RAG query.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages array cannot be empty")
    
    # Get the last user message for RAG retrieval
    last_user_message = None
    previous_user_message = None
    previous_assistant_message = None
    
    # We need to track pairs: find the assistant message RIGHT BEFORE the last user message
    for i in range(len(request.messages) - 1, -1, -1):
        msg = request.messages[i]
        
        if msg.role == "user" and last_user_message is None:
            # This is the most recent user message
            last_user_message = msg.content
        elif msg.role == "assistant" and last_user_message is not None and previous_assistant_message is None:
            # This is the assistant message right before the last user message
            previous_assistant_message = msg.content
        elif msg.role == "user" and previous_assistant_message is not None and previous_user_message is None:
            # This is the user message before the last exchange (for additional context)
            previous_user_message = msg.content
            break  # We have enough context
    
    if not last_user_message:
        raise HTTPException(status_code=400, detail="No user message found in chat history")
    
    # Build context-aware query for RAG
    # For follow-up questions, combine the previous exchange with the current question
    rag_query_parts = []
    
    if previous_user_message:
        rag_query_parts.append(previous_user_message)
    
    if previous_assistant_message:
        rag_query_parts.append(previous_assistant_message)
    
    # Always include the current question
    rag_query_parts.append(last_user_message)
    
    # Combine with spacing
    rag_query = " ".join(rag_query_parts)
    
    print(f"RAG Query: {rag_query}")  # Debug output
    
    # Retrieve relevant context with metadata (pdf_name, page_number, content)
    sources = retrieve_with_metadata(rag_query, k=5)
    
    # Build context string for the LLM prompt
    context_parts = []
    for idx, source in enumerate(sources):
        context_parts.append(
            f"[Source {idx+1}: {source['pdf_name']}, Page {source['page_number']}]\n{source['content']}"
        )
    context = "\n\n".join(context_parts)
    
    # Build the system prompt with context
    system_prompt = f"""Aşağıdakı kontekstdən istifadə edərək sualı cavablandırın, həmişə Azərbaycan dilində. Kontekst Azərbaycan dilindədir. Cavabınız dəqiq olmalı, müvafiq **kəmiyyət məlumatlarını** (məsələn, faizlər, nisbətlər, konkret ölçülər) daxil etməli və hər bir faktın mənbəyini qeyd etməlidir. Mümkün olduğu halda, rəqəmlərə birbaşa istinad edin.

Kontekst (istinadlarla):
{context}

İstinad formatı: [Source N: pdf_name, Page X]

Kontekstdə qeyd olunan konkret rəqəmləri və ya məlumat nöktələrini istifadə edərək cavab verməyə əmin olun. Əgər sual müqayisə ilə bağlıdırsa, bu müqayisəni aydın şəkildə göstərmək üçün bütün lazımlı detalları daxil etdiyinizdən əmin olun.
Hazırkı suala cavab verməklə yanaşı, əvvəlki cavablara və ya qarşılıqlı əlaqələrə aid olan məlumatlara da istifadə edin, söhbətin davamlılığını və ardıcıllığını qoruyun.
"""
    
    # Build messages array for LLM (system + chat history)
    llm_messages = [{"role": "system", "content": system_prompt}]
    
    # Add the full chat history to maintain conversation continuity
    for msg in request.messages:
        llm_messages.append({"role": msg.role, "content": msg.content})
    
    # Call LLM
    answer = foundry_chat(llm_messages, max_tokens=500)
    
    # Prepare source references for response
    source_refs = [
        SourceReference(
            pdf_name=source["pdf_name"],
            page_number=source["page_number"],
            content=source["content"]
        )
        for source in sources
    ]
    print(f"Sources: {source_refs}")
    print(f"Answer: {answer}")
    return LLMResponse(sources=source_refs, answer=answer)
    """Example endpoint showing pretty-formatted JSON"""
    return JSONResponse(
        content={
            "sources": [
                {
                    "pdf_name": "report.pdf",
                    "page_number": 3,
                    "content": "Extracted text snippet..."
                }
            ],
            "answer": "Here is the response based on the retrieved content..."
        },
        media_type="application/json; charset=utf-8",
        headers={"X-Pretty-JSON": "true"}
    )
    """
    LLM endpoint that receives chat history and produces an answer with source references.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages array cannot be empty")
    
    # Get the last user message for RAG retrieval
    last_user_message = None
    user_messages = []
    previous_assistant_message = None  # Variable to store the previous assistant's response
    
    for msg in request.messages:
        if msg.role == "user":
            user_messages.append(msg.content)
            last_user_message = msg.content
        elif msg.role == "assistant":
            previous_assistant_message = msg.content  # Capture the last assistant's response
    
    if not last_user_message:
        raise HTTPException(status_code=400, detail="No user message found in chat history")
    
    # Build context-aware query for RAG by combining recent conversation
    rag_query = last_user_message
    
    # If there was a previous assistant response, add it to the query
    if previous_assistant_message:
        rag_query = f"{previous_assistant_message} {rag_query}"
    
    print(f"RAG Query: {rag_query}")  # Debug output
    
    # Retrieve relevant context with metadata (pdf_name, page_number, content)
    sources = retrieve_with_metadata(rag_query, k=5)  # Increased k to 5 for better coverage
    
    # Build context string for the LLM prompt
    context_parts = []
    for idx, source in enumerate(sources):
        context_parts.append(
            f"[Source {idx+1}: {source['pdf_name']}, Page {source['page_number']}]\n{source['content']}"
        )
    context = "\n\n".join(context_parts)
    
    # Build the system prompt with context
    system_prompt = f"""Aşağıdakı kontekstdən istifadə edərək sualı cavablandırın, həmişə Azərbaycan dilində. Kontekst Azərbaycan dilindədir. Cavabınız dəqiq olmalı, müvafiq **kəmiyyət məlumatlarını** (məsələn, faizlər, nisbətlər, konkret ölçülər) daxil etməli və hər bir faktın mənbəyini qeyd etməlidir. Mümkün olduğu halda, rəqəmlərə birbaşa istinad edin.

Kontekst (istinadlarla):
{context}

İstinad formatı: [Source N: pdf_name, Page X]

Kontekstdə qeyd olunan konkret rəqəmləri və ya məlumat nöqtələrini istifadə edərək cavab verməyə əmin olun. Əgər sual müqayisə ilə bağlıdırsa, bu müqayisəni aydın şəkildə göstərmək üçün bütün lazımlı detalları daxil etdiyinizdən əmin olun.
Hazırkı suala cavab verməklə yanaşı, əvvəlki cavablara və ya qarşılıqlı əlaqələrə aid olan məlumatlara da istifadə edin, söhbətin davamlılığını və ardıcıllığını qoruyun.
"""
    
    # Build messages array for LLM (system + chat history)
    llm_messages = [{"role": "system", "content": system_prompt}]
    
    # Add the chat history
    for msg in request.messages:
        llm_messages.append({"role": msg.role, "content": msg.content})
    
    # Call LLM
    answer = foundry_chat(llm_messages, max_tokens=500)
    
    # Prepare source references for response
    source_refs = [
        SourceReference(
            pdf_name=source["pdf_name"],
            page_number=source["page_number"],
            content=source["content"]
        )
        for source in sources
    ]
    
    return LLMResponse(sources=source_refs, answer=answer)

# Keep the old /ask/ endpoint for backward compatibility (optional)
class QuestionRequest(BaseModel):
    question: str
    chat_id: str | None = None

@app.post("/ask/")
def ask_question(request: QuestionRequest, session: Session = Depends(get_session)):
    """
    Legacy endpoint - converts to new format internally
    """
    # Build message history from database if chat_id exists
    messages = []
    if request.chat_id:
        # Get the last 10 messages (5 exchanges) in chronological order
        previous_messages = session.query(Message).filter(
            Message.chat_id == request.chat_id
        ).order_by(Message.created_at.desc()).limit(10).all()
        
        # Reverse to get chronological order (oldest to newest)
        previous_messages.reverse()
        
        for msg in previous_messages:
            messages.append({"role": msg.role, "content": msg.content})
    
    # Add current question
    messages.append({"role": "user", "content": request.question})
    
    # Call new LLM endpoint
    llm_request = LLMRequest(messages=[ChatMessage(**msg) for msg in messages])
    response = llm_endpoint(llm_request)
    
    # Save messages to database if chat_id provided
    if request.chat_id:
        user_msg = Message(
            id=str(uuid.uuid4()),
            chat_id=request.chat_id,
            role="user",
            content=request.question
        )
        session.add(user_msg)
        
        assistant_msg = Message(
            id=str(uuid.uuid4()),
            chat_id=request.chat_id,
            role="assistant",
            content=response.answer
        )
        session.add(assistant_msg)
        session.commit()
    
    # Return in legacy format
    return {
        "answer": response.answer,
        "sources": [s.dict() for s in response.sources]
    }