import ollama
import streamlit as st


# Center-aligning function
def center_text(text):
    return f"<div style='text-align:center'>{text}</div>"

# App Title, Subheader
st.markdown(center_text("<h1>  Ahmed's Streamlit Chatbot</h1>"), unsafe_allow_html=True)

# Display the image in the middle column
st.image('/Users/garbo/Downloads/Dalle_AI_JPEG_PIC.jpg', width=500)

# Introduction
st.markdown(center_text("<p>I'm here to assist you in accomplishing your daily tasks!</p>"), unsafe_allow_html=True)

#Initialise message history
if "messages" not in st.session_state:
    st.session_state["messages"] = []
 

#init models
if "model" not in st.session_state:
    st.session_state["model"] = [""]

models = [model["name"] for model in ollama.list()["models"]]
st.session_state["model"] = st.selectbox("Get started by choosing one of the following models",models)

#Message Generator once at a time
def model_res_generator():
    stream = ollama.chat(
        model=st.session_state["model"],
        messages=st.session_state["messages"],
        stream=True,
    )
    for chunk in stream:
        yield chunk["message"]["content"]

#Display chat messages from history on app rerun
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])    

#Declaring the Chat Input
if prompt := st.chat_input("Hi there, how can I help?"):

    #add latest message to history in format {role, content}
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message = st.write_stream(model_res_generator)
        st.session_state["messages"].append({"role": "assistant", "content": message})

   


