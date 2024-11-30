import streamlit as st
from transformers import pipeline
import PyPDF2
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/roberta-base-squad2"

model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize the Hugging Face question-answering pipeline
# qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
qa_pipeline = pipeline('question-answering', model=model_name, tokenizer=model_name)

def extract_text_from_pdf(pdf_file):
    """
    Extracts text from a PDF file using PyPDF2.
    """
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def ask_question(context, question):
    """
    Uses Hugging Face's QA model to answer questions based on the context.
    """
    result = qa_pipeline({"context": context, "question": question})
    return result.get("answer", "No answer found.")

# Streamlit UI
st.title("PDF Question-Answering App")
st.write("Upload a PDF, and ask any questions based on its content.")

# Upload PDF
uploaded_pdf = st.file_uploader("Upload your PDF file", type="pdf")
if uploaded_pdf:
    st.write("PDF uploaded successfully. Extracting text...")
    pdf_text = extract_text_from_pdf(uploaded_pdf)

    st.write("Text extracted from the PDF:")
    st.text_area("Extracted Text", pdf_text, height=300)

    # Ask questions
    question = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        if question.strip():
            answer = ask_question(pdf_text, question)
            st.success(f"Answer: {answer}")
        else:
            st.warning("Please enter a question.")