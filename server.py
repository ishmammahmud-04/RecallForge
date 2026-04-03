import os
import io
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from pypdf import PdfReader

app = Flask(__name__)
# Allows your Netlify frontend to communicate with this Railway backend
CORS(app)

# Pulls the API Key from your Railway Variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Use 'gemini-1.5-flash' for the best balance of speed and reliability
model = genai.GenerativeModel('gemini-1.5-flash')

# Simple in-memory storage for study sessions
study_text = ""

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

@app.route('/api/upload', methods=['POST'])
def upload_files():
    global study_text
    files = request.files.getlist('files')
    pasted_text = request.form.get('pasted_text', '')

    full_text = pasted_text + "\n"

    for file in files:
        try:
            if file.filename.endswith('.pdf'):
                pdf_bytes = file.read()
                reader = PdfReader(io.BytesIO(pdf_bytes))
                # Extract text from all pages
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        full_text += extracted + "\n"
            elif file.filename.endswith(('.txt', '.md')):
                full_text += file.read().decode('utf-8') + "\n"
        except Exception as e:
            print(f"Error reading {file.filename}: {e}")
            continue

    # Limit context size to stay within free tier limits and maintain speed
    study_text = full_text.strip()[:35000]
    
    if len(study_text) < 10:
        return jsonify({"error": "No readable text found"}), 400

    return jsonify({"message": "Successfully processed study material!"})

@app.route('/api/generate', methods=['POST'])
def generate_questions():
    global study_text
    if not study_text:
        return jsonify({"error": "Please upload study material first."}), 400

    data = request.json
    qty = data.get('qty', 5)
    q_types = data.get('types', 'short answer')

    # CRITICAL: Double braces {{ }} prevent the 'Invalid format specifier' ValueError
    prompt = f"""You are a helpful study assistant. 
Using ONLY the following text, generate exactly {qty} recall questions of types: {q_types}.

Rules:
- Output ONLY a raw JSON array. No conversational text or markdown.
- Format: [{{ "type": "short", "question": "...", "answer": "..." }}]

Study Material:
{study_text}"""

    try:
        response = model.generate_content(prompt)
        # Strip potential markdown backticks that the AI might include
        clean_json = response.text.replace('```json', '').replace('```', '').strip()
        return jsonify({"questions": clean_json})
    except Exception as e:
        print(f"Generation error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/evaluate', methods=['POST'])
def evaluate_answer():
    data = request.json
    
    # Again, use double braces {{ }} to escape JSON characters in the f-string
    prompt = f"""Evaluate this student's answer based on the correct answer provided.
Question: {data['question']}
Correct Answer: {data['correct_answer']}
Student Answer: {data['user_answer']}

Respond ONLY with this JSON format:
{{ "grade": "correct" or "partial" or "wrong", "feedback": "A short, helpful sentence." }}"""

    try:
        response = model.generate_content(prompt)
        clean_json = response.text.replace('```json', '').replace('```', '').strip()
        return jsonify({"evaluation": clean_json})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Railway provides the PORT environment variable automatically
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
