import os
import io
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from pypdf import PdfReader

app = Flask(__name__)
CORS(app)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# Simple in-memory store — no ChromaDB needed
study_text = ""


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


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
                total_pages = len(reader.pages)
                print(f"PDF has {total_pages} pages")

                if total_pages <= 80:
                    # Small enough — read everything
                    pages_to_read = list(range(total_pages))
                else:
                    # Large PDF — sample 80 pages spread across the whole doc
                    import math
                    step = total_pages / 80
                    pages_to_read = [int(i * step) for i in range(80)]

                for i in pages_to_read:
                    try:
                        extracted = reader.pages[i].extract_text()
                        if extracted:
                            full_text += extracted + "\n"
                    except Exception:
                        continue

                print(f"Read {len(pages_to_read)} pages out of {total_pages}")

            elif file.filename.endswith(('.txt', '.md')):
                full_text += file.read().decode('utf-8') + "\n"

        except Exception as e:
            print(f"Error reading {file.filename}: {e}")
            continue

    study_text = full_text.strip()[:40000]

    char_count = len(study_text)
    print(f"Stored {char_count} characters")

    if char_count < 20:
        return jsonify({"error": "No readable text found in uploaded files."}), 400

    return jsonify({"message": f"Successfully stored {char_count} characters!"})


@app.route('/api/generate', methods=['POST'])
def generate_questions():
    global study_text

    if not study_text or len(study_text.strip()) < 20:
        return jsonify({"error": "No study material found. Please upload files first."}), 400

    data = request.json
    qty = data.get('qty', 5)
    q_types = data.get('types', 'short answer')

    prompt = f"""You are an expert study and active recall coach,mentor and guide.
Using ONLY the study material below, generate exactly {qty} questions of these types: {q_types}.

Rules:
- Base every question strictly on the provided material
- For fill-in-the-blank, use ___ where the key term goes
- Keep questions specific and meaningful
- Output ONLY a raw JSON array, no markdown, no backticks, no explanation

JSON format:
[
  {{"type": "short", "question": "What is...?", "answer": "The answer is..."}}
]

Study Material:
{study_text}"""

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

    prompt = f"""Grade this student's answer.
Question: {question}
Correct Answer: {correct_answer}
Student Answer: {user_answer}

Respond ONLY with this JSON object, no markdown:
{{"grade": "correct" or "partial" or "wrong", "feedback": "one short sentence explaining why"}}"""

    try:
        response = model.generate_content(prompt)
        clean_json = response.text.replace('```json', '').replace('```', '').strip()
        return jsonify({"evaluation": clean_json})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
