import os
from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# Initialize Flask app
app = Flask(__name__)

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Preprocess text into chunks for better search
def preprocess_text(text, chunk_size=512):
    lines = text.split("\n")
    chunks = []
    current_chunk = ""
    for line in lines:
        if len(current_chunk) + len(line) <= chunk_size:
            current_chunk += line + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = line
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Load PDF and process content
PDF_PATH = r"c:\chatbot\input.pdf"  # Replace with your uploaded PDF path
pdf_text = extract_text_from_pdf(PDF_PATH)
pdf_chunks = preprocess_text(pdf_text)

# Load transformer model for semantic search
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModelForSequenceClassification.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
semantic_search_pipeline = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

# Search function
def search_query_in_pdf(query):
    results = []
    for chunk in pdf_chunks:
        scores = semantic_search_pipeline(query, [chunk], multi_class=True)
        if scores['scores'][0] > 0.6:  # Confidence threshold
            results.append(chunk)
    return results

# Fallback response
FALLBACK_RESPONSE = "Sorry, I didn’t understand your question. Do you want to connect with a live agent?"

# Routes
@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    query = data.get("query", "")
    if not query:
        return jsonify({"response": "Please provide a valid query."}), 400
    
    # Search for answers in PDF
    results = search_query_in_pdf(query)
    if results:
        return jsonify({"response": results[:3]})  # Return top 3 results
    else:
        return jsonify({"response": FALLBACK_RESPONSE})

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
