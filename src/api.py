from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form
from pydantic import BaseModel, RootModel
from sqlalchemy.orm import Session
from src.db import get_session, Chat, Message
from src.llm_client import foundry_chat
from src.rag import retrieve_with_metadata
import uuid
import os
import tempfile
import re
import base64
import mimetypes
from pathlib import Path
from typing import List, Dict
import io

from fastapi.responses import JSONResponse
import uvicorn

import fitz  # PyMuPDF
from PIL import Image, ImageEnhance
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


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
   RootModel: List[Dict[str, str]]

class Message(BaseModel):
    role: str
    content: str

class RequestBody(BaseModel):
    messages: List[Message]

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
    return [{"role": m["role"], "content": m["content"]} for m in messages]

@app.delete("/chat/{chat_id}")
def delete_chat(chat_id: str, session: Session = Depends(get_session)):
    chat = session.query(Chat).filter(Chat.chat_id == chat_id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    session.delete(chat)
    session.commit()
    return {"detail": f"Chat {chat_id} deleted successfully"}

@app.post("/llm/")
def llm_endpoint(request: List[Dict[str, str]]):
    """
    LLM endpoint that receives chat history and produces an answer with source references.
    Handles follow-up questions by including previous context in RAG query.
    """
    if not request:
        raise HTTPException(status_code=400, detail="Messages array cannot be empty")
    
    # Get the last user message for RAG retrieval
    last_user_message = None
    previous_user_message = None
    previous_assistant_message = None
    
    # We need to track pairs: find the assistant message RIGHT BEFORE the last user message
    for i in range(len(request) - 1, -1, -1):
        msg = request[i]
        
        if msg["role"] == "user" and last_user_message is None:
            # This is the most recent user message
            last_user_message = msg["content"]
        elif msg["role"] == "assistant" and last_user_message is not None and previous_assistant_message is None:
            # This is the assistant message right before the last user message
            previous_assistant_message = msg["content"]
        elif msg["role"] == "user" and previous_assistant_message is not None and previous_user_message is None:
            # This is the user message before the last exchange (for additional context)
            previous_user_message = msg["content"]
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

    print(request)  # Debug output
    
    # Add the full chat history to maintain conversation continuity
    for msg in request:
        llm_messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Call LLM
    answer = foundry_chat(llm_messages, max_tokens=500)

    print(answer)
    # Prepare source references for response
    source_refs = [
        # SourceReference(
            # pdf_name=source["pdf_name"],
            # page_number=source["page_number"],
            # content=source["content"]
        # )
        {"pdf_name": source["pdf_name"], "page_number": source["page_number"], "content": source["content"]}
        for source in sources
    ]
    print(f"Sources: {source_refs}")
    print(f"Answer: {answer}")
    return {"sources": source_refs, "answer": answer}
  

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
            messages.append({"role": msg["role"], "content": msg["content"]})
    
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

def image_to_data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def save_enhanced_page_image(
    page: fitz.Page,
    out_path: Path,
    zoom: float = 1.7,
    contrast: float = 1.2,
    sharpness: float = 1.1,
    jpeg_quality: int = 88,
    max_side: int = 2200,
):
    """Render a PDF page -> PIL -> enhance -> save as JPG."""
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)

    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="JPEG", quality=jpeg_quality, optimize=True)


def make_client():
    load_dotenv()

    base_url = os.getenv("BASE_URL", "").strip()
    api_key = os.getenv("API_KEY", "").strip()
    model = os.getenv("MODEL", "").strip()

    if not base_url or not api_key or not model:
        raise SystemExit("Missing BASE_URL / API_KEY / MODEL in environment (.env).")

    if not base_url.endswith("/openai/v1/"):
        base_url = base_url.rstrip("/") + "/openai/v1/"

    client = OpenAI(base_url=base_url, api_key=api_key)
    return client, model


def detect_script(text_sample: str) -> str:
    """Detect if text is primarily Cyrillic or Latin."""
    cyrillic_count = len(re.findall(r'[А-Яа-яЁёӘәІіҮүҒғҚқҢңҺһ]', text_sample))
    latin_count = len(re.findall(r'[A-Za-zƏəİıÖöÜüĞğŞşÇç]', text_sample))
    
    if cyrillic_count > latin_count:
        return "cyrillic"
    else:
        return "latin"


LATIN_PROMPT = (
    "Extract ALL text from this page in Markdown format.\n"
    "Rules:\n"
    "- Use proper Markdown formatting (headers with #, lists, bold, italic, etc.)\n"
    "- For images, use: ![](brief description of the image)\n"
    "- Preserve tables using Markdown table syntax\n"
    "- Keep chemical formulas as close as possible\n"
    "- Preserve all formatting, line breaks, and structure\n"
    "- Return ONLY the Markdown text, no explanations\n"
)

CYRILLIC_PROMPT = (
    "Extract ALL Cyrillic text from this page in Markdown format.\n"
    "This is Azerbaijani written in Cyrillic script (some letters differ from Russian).\n"
    "Rules:\n"
    "- Extract exactly what you see in Cyrillic, including special letters like Ә\n"
    "- Use proper Markdown formatting (headers with #, lists, bold, italic, etc.)\n"
    "- For images, use: ![](brief description of the image)\n"
    "- Preserve tables using Markdown table syntax\n"
    "- Keep chemical formulas as close as possible\n"
    "- Preserve all formatting, line breaks, and structure\n"
    "- Return ONLY the Markdown text in Cyrillic, no explanations\n"
)


def is_internal_server_error(exc: Exception) -> bool:
    """Check if exception is a 500 internal server error."""
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    resp = getattr(exc, "response", None)
    if status is None and resp is not None:
        status = getattr(resp, "status_code", None)
    
    msg = str(exc).lower()
    
    if status == 500:
        return True
    if "internal server error" in msg:
        return True
    if "server error" in msg and ("500" in msg or "status: 500" in msg):
        return True
    
    return False


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type(Exception),
)
def extract_text_from_image(
    client: OpenAI, 
    model: str, 
    img_path: Path, 
    prompt: str, 
    max_tokens: int
):
    """Extract text from image using LLM."""
    img_data_url = image_to_data_url(str(img_path))

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": img_data_url}},
                    ],
                }
            ],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    except Exception as e:
        if is_internal_server_error(e):
            raise
        raise

app.post("/ocr/")
async def ocr_pdf(
    file: UploadFile = File(...),
    zoom: float = Form(1.7),
    contrast: float = Form(1.2),
    sharpness: float = Form(1.1),
    jpeg_quality: int = Form(88),
    max_side: int = Form(2200),
    max_tokens: int = Form(1500),
):
    """
    Extract text from PDF and return as Markdown for each page.
    Automatically detects Cyrillic vs Latin script.
    
    Returns:
        List[Dict]: [{"page_number": 1, "MD_text": "..."}, ...]
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        pdf_path = temp_path / file.filename
        img_dir = temp_path / "images"
        img_dir.mkdir()
        
        # Save uploaded file
        with open(pdf_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Get client
        try:
            client, model = make_client()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize client: {str(e)}")
        
        # Process PDF
        try:
            doc = fitz.open(str(pdf_path))
            results = []
            
            # First, sample first page to detect script
            if doc.page_count > 0:
                first_page = doc[0]
                sample_text = first_page.get_text()
                script_type = detect_script(sample_text)
                prompt = CYRILLIC_PROMPT if script_type == "cyrillic" else LATIN_PROMPT
            else:
                prompt = LATIN_PROMPT
            
            # Process each page
            for i in range(doc.page_count):
                page = doc[i]
                img_path = img_dir / f"page_{i+1:04d}.jpg"
                
                # Save enhanced image
                save_enhanced_page_image(
                    page=page,
                    out_path=img_path,
                    zoom=zoom,
                    contrast=contrast,
                    sharpness=sharpness,
                    jpeg_quality=jpeg_quality,
                    max_side=max_side,
                )
                
                # Extract text
                try:
                    md_text = extract_text_from_image(
                        client=client,
                        model=model,
                        img_path=img_path,
                        prompt=prompt,
                        max_tokens=max_tokens,
                    )
                except Exception as e:
                    md_text = f"[ERROR extracting page {i+1}: {str(e)}]"
                
                results.append({
                    "page_number": i + 1,
                    "MD_text": md_text
                })
            
            doc.close()
            
            return JSONResponse(content=results)
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)