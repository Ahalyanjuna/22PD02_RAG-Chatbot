# Main Flask application for Document RAG Chatbot

import os
from flask import Flask, request, jsonify, render_template
import json
from document_processor import add_to_knowledge_base
from vector_db_manager import VectorDatabase
from query_engine import query_knowledge_base

# Initialize Flask app
app = Flask(__name__)

# Directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
KB_FOLDER = os.path.join(BASE_DIR, 'knowledge_base')
VECTOR_DB_PATH = os.path.join(BASE_DIR, 'vector_db')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(KB_FOLDER, exist_ok=True)
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

# Initialize vector database
vector_db = VectorDatabase()
try:
    vector_db.load(VECTOR_DB_PATH)
    print("Loaded existing vector database")
except:
    print("Creating new vector database")

# API Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def api_query():
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    # Process query
    result = query_knowledge_base(query, vector_db)
    
    return jsonify(result)

@app.route('/api/upload', methods=['POST'])
def api_upload():
    # Check if files were uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Save file temporarily
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    # Process file and add to knowledge base
    result = add_to_knowledge_base(file_path, vector_db, VECTOR_DB_PATH)
    
    # Clean up
    os.remove(file_path)
    
    return jsonify(result)

@app.route('/api/prebuilt-status', methods=['GET'])
def api_prebuilt_status():
    # Return status of prebuilt knowledge base
    status = {
        "total_chunks": len(vector_db.texts),
        "document_count": len(set([meta.get('doc_id') for meta in vector_db.metadata]))
    }
    return jsonify(status)

# Initialize prebuilt knowledge base
def build_prebuilt_knowledge_base():
    """Process all documents in the knowledge base folder."""
    for filename in os.listdir(KB_FOLDER):
        file_path = os.path.join(KB_FOLDER, filename)
        if os.path.isfile(file_path):
            print(f"Processing {filename} for prebuilt knowledge base...")
            result = add_to_knowledge_base(file_path, vector_db, VECTOR_DB_PATH)
            print(result)

if not vector_db.loaded or len(vector_db.texts) == 0:
    build_prebuilt_knowledge_base()

# Start the app
if __name__ == '__main__':
    app.run(debug=True, port=5000)