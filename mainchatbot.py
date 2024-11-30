import streamlit as st
from transformers import pipeline
from PyPDF2 import PdfReader
from keybert import KeyBERT
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io

model_name = "deepset/roberta-base-squad2"


# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Summarize text
def summarize_text(text, max_length=200, min_length=50):
    st.info("Loading summarization model...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text[:1024], max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]["summary_text"]


# Multilingual translation
def translate_text(text, target_language="es"):
    st.info("Loading translation model...")
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
    translation = translator(text[:512])  # Translate first 512 characters
    return translation[0]["translation_text"]

# Question answering
def answer_question(context, question):
    st.info("Loading question-answering model...")
    qa_pipeline = pipeline('question-answering', model=model_name, tokenizer=model_name)
    answer = qa_pipeline(question=question, context=context)
    return answer["answer"]

# Search in text
def search_in_text(text, search_term):
    occurrences = text.lower().count(search_term.lower())
    return occurrences

# Download content as a file
def download_file(content, filename):
    file = io.BytesIO(content.encode())
    return st.download_button(
        label="Download Result", data=file, file_name=filename, mime="text/plain"
    )

# Streamlit UI
st.title("AI-Powered PDF Analysis Tool")
st.write("Upload a PDF and explore various AI-powered features!")

uploaded_pdf = st.file_uploader("Upload your PDF file", type="pdf")
if uploaded_pdf is not None:
    pdf_text = extract_text_from_pdf(uploaded_pdf)
    
    # Summarization
    if st.button("Summarize PDF"):
        summary = summarize_text(pdf_text)
        st.subheader("Summary:")
        st.success(summary)
        download_file(summary, "summary.txt")
    
    
    # Search Functionality
    search_term = st.text_input("Search in PDF (Case-Insensitive):")
    if search_term:
        occurrences = search_in_text(pdf_text, search_term)
        st.write(f"'{search_term}' found *{occurrences}* times in the document.")
    
    # Multilingual Translation
    target_language = st.selectbox("Select Target Language", options=["es", "fr", "de", "hi", "zh"], index=0)
    if st.button(f"Translate to {target_language.upper()}"):
        translated_text = translate_text(pdf_text, target_language)
        st.subheader(f"Translated Text ({target_language.upper()}):")
        st.write(translated_text)
        download_file(translated_text, f"translated_{target_language}.txt")
    
    # Question Answering
    st.subheader("Ask Questions About the PDF")
    user_question = st.text_input("Type your question:")
    if user_question:
        answer = answer_question(pdf_text, user_question)
        st.write("Answer:")
        st.success(answer)