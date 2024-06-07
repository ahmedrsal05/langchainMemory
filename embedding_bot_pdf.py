import os
import tempfile
import streamlit as st
from rag_PDF import ChatPDF

from streamlit_chat import message
 

#Adds a title for the web page
st.set_page_config(page_title="PDF Document Chatbot")

def display_messages():
    """
    Displays chat messages in the streamlit app
    """
    st.subheader("Chat to your PDF")
    for i, (msg, is_user) in enumerate(st.session_state.messages):
        message(msg, is_user=is_user, key=str(i))
    st.session_state.thinking_spinner = st.empty()

def process_input():
    """
    Processes user input and updates the chat messages in the Streamlit app.
    """


    if st.session_state.user_input and len(st.session_state.user_input.strip()) > 0:
        user_text = st.session_state.user_input.strip()
        with st.session_state.thinking_spinner, st.spinner(f"Thinking"):
            agent_text = st.session_state.assistant.ask(user_text)
        st.session_state.messages.append((user_text, True))
        st.session_state.messages.append((agent_text, False))

def read_and_save_file():
    """
    Reads and saves the uploaded file, performs ingestion, and clears the assistant     state.
    """

    st.session_state.assistant.clear()
    st.session_state.messages = []
    st.session_state.user_input = ""
    for file in st.session_state.file_uploader:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
            tf.write(file.getbuffer())
            file_path = tf.name
        with st.session_state.ingestion_spinner, st.spinner(f"Ingesting '{file.name}'"):
            st.session_state.assistant.ingest(file_path)
        os.remove(file_path)

def page():
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.session_state.assistant = ChatPDF()
    st.header("PDF BOT 🤖")
    st.file_uploader("Upload PDF", type=["pdf"], key="file_uploader", on_change=read_and_save_file, accept_multiple_files=True)
    st.session_state.ingestion_spinner = st.empty()
    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)

if __name__ == "__main__":
    page()
