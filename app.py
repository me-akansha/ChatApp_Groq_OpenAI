
import streamlit as st
from groq import Groq
import os
from typing import List, Dict

st.set_page_config(page_title="Groq Streamlit Chat", page_icon="🤖")

# ----- Helpers -----
def get_client():
    """Initialize Groq client using secrets or environment variable."""
    api_key = None
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("Groq API key not found. Set GROQ_API_KEY in Streamlit secrets or environment.")
        st.stop()
    return Groq(api_key=api_key)

def stream_response(client: Groq, messages: List[Dict[str, str]], model: str):
    """Yield incremental assistant text chunks from Groq streaming API."""
    stream_iter = client.chat.completions.create(
        messages=messages,
        model=model,
        stream=True
    )
    assistant_text = ""
    try:
        for delta in stream_iter:
            try:
                choice = delta.choices[0]
                part = None
                if hasattr(choice, "delta"):
                    d = choice.delta
                    if isinstance(d, dict):
                        part = d.get("content") or (d.get("message", {}) or {}).get("content")
                    else:
                        part = getattr(d, "content", None) or getattr(getattr(d, "message", {}), "content", None)
                if part is None:
                    part = getattr(choice, "text", None)
            except Exception:
                part = None

            if part:
                assistant_text += part
                yield assistant_text
    except Exception:
        yield assistant_text
        raise

# ----- UI -----
st.title("🤖 Groq-Bot : An AI ChatBot")
st.write("A minimal open-source Streamlit chat using Groq's Python SDK with streaming responses and session_state.")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    model = st.selectbox(
        "Model",
        options=[
            "llama-3.3-70b-versatile",
            "mistral-saba-24b",
            "gemma-7b",
        ],
        index=0  # default to llama-3.3-70b-versatile
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    clear = st.button("Clear chat")

if "messages" not in st.session_state or clear:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]
    st.session_state.history = []

# Show chat history
for msg in st.session_state.history:
    st.chat_message(msg["role"]).write(msg["content"])

# Input
user_input = st.chat_input("Type a message and press Enter")
if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "user", "content": user_input})

    client = get_client()

    assistant_msg = st.chat_message("assistant")
    with assistant_msg:
        placeholder = st.empty()
        placeholder.markdown("_...thinking..._")

    try:
        partial = ""
        for partial in stream_response(client, st.session_state.messages, model):
            with assistant_msg:
                placeholder.markdown(partial)
        final_text = partial or ""
        st.session_state.history.append({"role": "assistant", "content": final_text})
        st.session_state.messages.append({"role": "assistant", "content": final_text})
    except Exception as e:
        st.error(f"Error while streaming response: {e}")
        final_text = partial or f"(error: {e})"
        st.session_state.history.append({"role": "assistant", "content": final_text})
        st.session_state.messages.append({"role": "assistant", "content": final_text})
