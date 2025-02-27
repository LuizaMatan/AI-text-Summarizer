import streamlit as st
from transformers import pipeline
import torch

##loading the AI model --> BART model
@st.cache_resource
def load_summarizer():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return pipeline("summarization", model="facebook/bart-large-cnn", device=-1)  # Force CPU

summarizer = load_summarizer()

##web interface
st.title("📝 AI Conversation Summarizer")
st.write("Paste a long conversation or text, and AI will summarize it for you!")

##get user input
user_input = st.text_area("Enter conversation here:")

##summarizing the text
if st.button("Summarize"):
    if user_input:
        summary = summarizer(user_input, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]
        st.subheader("Summary:")
        st.success(summary)
    else:
        st.warning("Please enter some text to summarize.")
