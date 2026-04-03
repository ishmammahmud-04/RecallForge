# server.py
import os
import json
import chromadb
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from pypdf import PdfReader

app = Flask(__name__)
CORS(app)

# 🔑 PASTE YOUR GEMINI API KEY HERE
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# Setup ChromaDB in memory (Perfect for ephemeral cloud servers)
chroma_client = chromadb.Client()
try:
    chroma_client.delete_collection("study_notes")
except:
    pass
collection = chroma_client.create_collection(name="study_notes")

@app.route('/api/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('files')
    pasted_text = request.form.get('pasted_text', '')
    
    full_text = pasted_text + "\n"
    
    for file in files:
        if file.filename.endswith('.pdf'):
            reader = PdfReader(file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    full_text += extracted + "\n"
        elif file.filename.endswith(('.txt', '.md')):
            full_text += file.read().decode('utf-8') + "\n"

    chunk_size = 1000
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
    
    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) > 10:
            # Use Gemini's text embedding model for the RAG database
            embedding = genai.embed_content(
                model="models/text-embedding-004",
                content=chunk,
                task_type="retrieval_document"
            )['embedding']
            
            collection.add(
                ids=[f"chunk_{i}"],
                embeddings=[embedding],
                documents=[chunk]
            )

    return jsonify({"message": f"Successfully processed {len(chunks)} chunks!"})

@app.route('/api/generate', methods=['POST'])
def generate_questions():
    data = request.json
    qty = data.get('qty', 5)
    q_types = data.get('types', 'short answer')
    
    results = collection.get(limit=5)
    context_text = "\n".join(results['documents']) if results['documents'] else "No context found."

    prompt = f"""
    You are an expert study coach and guide and mentor. 
    Using ONLY the study material below, generate exactly {qty} questions of these types: {q_types}.
    
    Format the output strictly as a JSON array like this:
    [
      {{"type": "short", "question": "What is...?", "answer": "The answer is..."}}
    ]
    Do not output any markdown formatting, code blocks, or backticks. Just the raw JSON array.
    
    Study Material:
    {context_text}
    """

    try:
        response = model.generate_content(prompt)
        clean_json = response.text.replace('```json', '').replace('```', '').strip()
        return jsonify({"questions": clean_json})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/evaluate', methods=['POST'])
def evaluate_answer():
    data = request.json
    question = data['question']
    correct_answer = data['correct_answer']
    user_answer = data['user_answer']

    prompt = f"""
    Grade this student's answer.
    Question: {question}
    Correct Answer: {correct_answer}
    Student Answer: {user_answer}
    
    Respond ONLY with a JSON object: {{"grade": "correct" or "partial" or "wrong", "feedback": "one short sentence explaining why"}}
    Do not use markdown.
    """
    
    response = model.generate_content(prompt)
    clean_json = response.text.replace('```json', '').replace('```', '').strip()
    return jsonify({"evaluation": clean_json})
    
@app.route('/api/health', methods=['GET'])
def health():
return jsonify({"status": "ok"})

if __name__ == '__main__':
    # Cloud servers assign their own ports, so we have to bind to 0.0.0.0
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
