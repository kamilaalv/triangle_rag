
---

# FastAPI Chat Application with LLM Integration

## Overview

This FastAPI application provides functionality for managing chat conversations and interacting with a Large Language Model (LLM). It includes the ability to create and manage chat sessions, store messages in a database, and process chat history to generate context-aware responses using LLMs. The LLM answers questions based on retrieved data and provides source references for the answers.

## Features

* **Chat Management**: Create, list, and delete chat sessions.
* **Message Storage**: Store user and assistant messages in a database.
* **Context-Aware Responses**: LLM provides answers based on the chat history and metadata, with context pulled from relevant sources.
* **Source Referencing**: Each response is paired with source references indicating the origin of the data used in the answer.
* **Legacy Endpoint**: Compatibility with an older endpoint for asking questions, converting them to the new format internally.

## Dependencies

1. Python 3.6+
2. Install required dependencies by running:

   ```bash
   pip install -r requirements.txt
   ```
3. Database setup for SQLAlchemy (PostgreSQL, SQLite, etc.)
4. OpenAI API credentials for LLM integration.

### Required Environment Variables

* `BASE_URL`: The base URL for the OpenAI API.
* `API_KEY`: Your OpenAI API key.
* `MODEL`: The model to be used for text generation (e.g., GPT-3, GPT-4).

## API Endpoints

### 1. `POST /chat/`

Create a new chat session.

#### Request Body:

```json
{
  "title": "Chat Title"
}
```

#### Response:

```json
{
  "chat_id": "generated-chat-id",
  "title": "Chat Title"
}
```

### 2. `GET /chats/`

Retrieve all chat sessions.

#### Response:

```json
[
  {
    "chat_id": "chat-id",
    "title": "Chat Title"
  },
  {
    "chat_id": "chat-id-2",
    "title": "Another Chat Title"
  }
]
```

### 3. `GET /messages/{chat_id}`

Retrieve messages for a specific chat session.

#### Request:

```bash
GET /messages/{chat_id}
```

#### Response:

```json
[
  {
    "role": "user",
    "content": "Hello, assistant!"
  },
  {
    "role": "assistant",
    "content": "Hello, how can I assist you today?"
  }
]
```

### 4. `DELETE /chat/{chat_id}`

Delete a chat session.

#### Request:

```bash
DELETE /chat/{chat_id}
```

#### Response:

```json
{
  "detail": "Chat {chat_id} deleted successfully"
}
```

### 5. `POST /llm/`

Send chat history to the LLM for context-aware response generation. The LLM generates an answer based on the context from the chat and metadata retrieved via the RAG (Retrieval-Augmented Generation) method.

#### Request Body:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is the capital of France?"
    },
    {
      "role": "assistant",
      "content": "The capital of France is Paris."
    }
  ]
}
```

#### Response:

```json
{
  "sources": [
    {
      "pdf_name": "source.pdf",
      "page_number": 1,
      "content": "Content of the source"
    }
  ],
  "answer": "The capital of France is Paris."
}
```

### 6. `POST /ask/` (Legacy Endpoint)

This legacy endpoint is for backward compatibility. It accepts a question and chat ID, retrieves chat history from the database, and generates an LLM response.

#### Request Body:

```json
{
  "question": "What is the capital of France?",
  "chat_id": "chat-id"
}
```

#### Response:

```json
{
  "answer": "The capital of France is Paris.",
  "sources": [
    {
      "pdf_name": "source.pdf",
      "page_number": 1,
      "content": "Content of the source"
    }
  ]
}
```

## LLM Response Generation

* The LLM is used to generate answers based on chat history. The context is dynamically built by extracting the last user message and the assistant's previous response. This context is combined into a RAG query that is used to fetch relevant information.
* The response is then enhanced with references to the sources from which the context was retrieved, such as PDF names and page numbers.

## Running the Server

To run the server, use the following command:

```bash
uvicorn src.api:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

---

