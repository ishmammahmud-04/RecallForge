import os
import io
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from pypdf import PdfReader

app = Flask(__name__)
# Crucial: This allows your Netlify frontend to talk to your Railway backend
CORS(app, resources={r"/*": {"origins": "*"}}) 

# Make sure this variable is set in Railway 'Variables' tab!
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# UPDATED: Using a valid high-speed model
model = genai.GenerativeModel('gemini-1.5-flash')

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
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        full_text += extracted + "\n"
            elif file.filename.endswith(('.txt', '.md')):
                full_text += file.read().decode('utf-8') + "\n"
        except Exception as e:
            print(f"Error reading {file.filename}: {e}")
            continue

    study_text = full_text.strip()[:30000] # Kept slightly shorter for stability
    return jsonify({"message": "Successfully stored context!"})

@app.route('/api/generate', methods=['POST'])
def generate_questions():
    global study_text
    if not study_text:
        return jsonify({"error": "No study material found."}), 400

    data = request.json
    qty = data.get('qty', 5)
    q_types = data.get('types', 'short answer')

    prompt = f"""Generate exactly {qty} recall questions of types: {q_types}.
    Base them ONLY on this text: {study_text}
    
    Output ONLY a raw JSON array. No markdown, no triple backticks.
    Format: [{"type": "...", "question": "...", "answer": "..."}]"""

    try:
        response = model.generate_content(prompt)
        # Cleaner JSON handling
        raw_text = response.text.replace('```json', '').replace('```', '').strip()
        return jsonify({"questions": raw_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/evaluate', methods=['POST'])
def evaluate_answer():
    data = request.json
    prompt = f"""Grade this. Question: {data['question']} | Correct: {data['correct_answer']} | Student: {data['user_answer']}
    Respond ONLY with JSON: {{"grade": "correct/partial/wrong", "feedback": "..."}}"""

    try:
        response = model.generate_content(prompt)
        raw_text = response.text.replace('```json', '').replace('```', '').strip()
        return jsonify({"evaluation": raw_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
