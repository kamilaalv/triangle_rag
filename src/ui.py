import streamlit as st
import requests
import time

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Azeri Chatbot", layout="wide")
st.title("Azeri Chatbot")

# Initialize session state
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for chat management
with st.sidebar:
    st.header("Chats")
    
    # Button to create new chat
    new_chat_title = st.text_input("New chat title:")
    if st.button("Create New Chat"):
        if new_chat_title:
            try:
                response = requests.post(
                    f"{API_URL}/chat/", 
                    json={"title": new_chat_title},
                    timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.current_chat_id = data["chat_id"]
                    st.session_state.messages = []
                    st.success(f"Created: {new_chat_title}")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(f"Error creating chat: {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Cannot connect to API: {e}")
        else:
            st.warning("Please enter a chat title")
    
    # Load and display existing chats
    try:
        chats_response = requests.get(f"{API_URL}/chats/", timeout=10)
        if chats_response.status_code == 200:
            chats = chats_response.json()
            
            if not chats:
                st.info("No chats yet. Create one above!")
            
            for chat in chats:
                if st.button(chat["title"], key=chat["chat_id"]):
                    st.session_state.current_chat_id = chat["chat_id"]
                    # Load messages for this chat
                    try:
                        msgs_response = requests.get(
                            f"{API_URL}/messages/{chat['chat_id']}",
                            timeout=10
                        )
                        if msgs_response.status_code == 200:
                            st.session_state.messages = msgs_response.json()
                        else:
                            st.error(f"Error loading messages: {msgs_response.text}")
                            st.session_state.messages = []
                    except requests.exceptions.RequestException as e:
                        st.error(f"Error loading messages: {e}")
                        st.session_state.messages = []
                    st.rerun()
        else:
            st.error(f"Error loading chats: {chats_response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Cannot connect to API at {API_URL}")
        st.info("Make sure the FastAPI server is running: `uvicorn src.api:app --reload`")

# Display current chat info
if st.session_state.current_chat_id:
    st.caption(f"Chat ID: {st.session_state.current_chat_id}")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if st.session_state.current_chat_id is None:
    st.info("👈 Create or select a chat from the sidebar to start")

if prompt := st.chat_input("Ask a question...", disabled=st.session_state.current_chat_id is None):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get response from API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask/",
                    json={
                        "question": prompt,
                        "chat_id": st.session_state.current_chat_id
                    },
                    timeout=30
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    # Show context in expander
                    if "context_used" in data:
                        with st.expander("📄 Context used"):
                            st.text(data["context_used"])
                else:
                    error_msg = f"Error {response.status_code}: {response.text}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {error_msg}"})
            except requests.exceptions.RequestException as e:
                error_msg = f"Connection error: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})