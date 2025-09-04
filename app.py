import streamlit as st
from groq import Groq
import os
from typing import List, Dict

st.set_page_config(page_title="Groq Chatbot", page_icon="🤖")

# --- Initialize Groq Client ---
def initialize_groq_client():
    """Retrieve the Groq API key from secrets or environment and initialize the client."""
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
    if not api_key:
        st.error("Groq API key not found. Please set GROQ_API_KEY in Streamlit secrets or environment variables.")
        st.stop()
    return Groq(api_key=api_key)

# --- Stream Groq Response ---
def generate_streamed_response(client: Groq, messages: List[Dict[str, str]], model: str):
    """Stream assistant responses chunk by chunk from Groq API."""
    response_stream = client.chat.completions.create(
        messages=messages,
        model=model,
        stream=True
    )
    full_response = ""
    try:
        for chunk in response_stream:
            try:
                choice = chunk.choices[0]
                content_part = None

                if hasattr(choice, "delta"):
                    delta = choice.delta
                    if isinstance(delta, dict):
                        content_part = delta.get("content") or (delta.get("message", {}) or {}).get("content")
                    else:
                        content_part = getattr(delta, "content", None) or getattr(getattr(delta, "message", {}), "content", None)

                if content_part is None:
                    content_part = getattr(choice, "text", None)

            except Exception:
                content_part = None

            if content_part:
                full_response += content_part
                yield full_response

    except Exception:
        yield full_response
        raise

# --- UI Configuration ---
st.title("🤖 Groq-Bot: AI Chat Assistant")
st.write("An open-source chatbot built with Streamlit and Groq's Python SDK, featuring streaming responses and session management.")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Settings")
    selected_model = st.selectbox(
        "Choose a model:",
        options=[
            "llama-3.3-70b-versatile",
            "mistral-saba-24b",
            "gemma-7b"
        ],
        index=0
    )
    temperature = st.slider("Response Creativity (Temperature)", 0.0, 1.0, 0.2, step=0.05)
    clear_chat = st.button("Clear Conversation")

# --- Initialize Session State ---
if "chat_history" not in st.session_state or clear_chat:
    st.session_state.chat_history = [{"role": "system", "content": "You are a helpful assistant."}]
    st.session_state.display_log = []

# --- Display Chat History ---
for message in st.session_state.display_log:
    st.chat_message(message["role"]).write(message["content"])

# --- Chat Input ---
user_message = st.chat_input("Ask your question here...")
if user_message:
    st.session_state.display_log.append({"role": "user", "content": user_message})
    st.session_state.chat_history.append({"role": "user", "content": user_message})

    client = initialize_groq_client()

    response_container = st.chat_message("assistant")
    with response_container:
        placeholder = st.empty()
        placeholder.markdown("_...generating response..._")

    try:
        accumulated_text = ""
        for accumulated_text in generate_streamed_response(client, st.session_state.chat_history, selected_model):
            with response_container:
                placeholder.markdown(accumulated_text)

        final_response = accumulated_text or ""
        st.session_state.display_log.append({"role": "assistant", "content": final_response})
        st.session_state.chat_history.append({"role": "assistant", "content": final_response})

    except Exception as e:
        error_message = f"An error occurred: {e}"
        st.error(error_message)
        st.session_state.display_log.append({"role": "assistant", "content": error_message})
        st.session_state.chat_history.append({"role": "assistant", "content": error_message})

