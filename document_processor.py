# Document Processing for Document RAG Chatbot

import os
import uuid
import PyPDF2
import docx
import csv
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from vector_db_manager import get_embeddings

# Download required NLTK resources
nltk.download('punkt', quiet=True)

def chunk_text(text, chunk_size=200, overlap=50):
    """Split text into overlapping chunks."""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_tokens = len(sentence.split())
        
        if current_size + sentence_tokens > chunk_size and current_chunk:
            # Add current chunk to chunks
            chunks.append(' '.join(current_chunk))
            
            # Keep overlap sentences for next chunk
            overlap_size = 0
            overlap_chunk = []
            for s in reversed(current_chunk):
                s_tokens = len(s.split())
                if overlap_size + s_tokens <= overlap:
                    overlap_chunk.insert(0, s)
                    overlap_size += s_tokens
                else:
                    break
            
            current_chunk = overlap_chunk
            current_size = overlap_size
        
        current_chunk.append(sentence)
        current_size += sentence_tokens
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def process_pdf(file_path):
    """Extract text from PDF."""
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    
    return text

def process_docx(file_path):
    """Extract text from DOCX."""
    doc = docx.Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    
    return text

def process_csv(file_path):
    """Extract text from CSV by converting to descriptive text."""
    try:
        df = pd.read_csv(file_path)
        # Convert dataframe to readable text format
        text = f"CSV file with {len(df)} rows and {len(df.columns)} columns.\n"
        text += f"Columns: {', '.join(df.columns)}.\n\n"
        
        # Sample data summary
        text += "Sample data and statistics:\n"
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                text += f"Column {col}: min={df[col].min()}, max={df[col].max()}, "
                text += f"mean={df[col].mean():.2f}, median={df[col].median()}.\n"
            else:
                unique_vals = df[col].nunique()
                text += f"Column {col}: {unique_vals} unique values.\n"
                if unique_vals < 10:  # Only show all values if there are few
                    text += f"Values: {', '.join(df[col].unique().astype(str))}.\n"
                else:
                    text += f"Sample values: {', '.join(df[col].sample(5).astype(str))}.\n"
        
        # Add actual data rows as text
        text += "\nSample rows:\n"
        for i, row in df.head(5).iterrows():
            text += f"Row {i}: {' | '.join([f'{col}={val}' for col, val in row.items()])}.\n"
        
        return text
    except Exception as e:
        return f"Error processing CSV: {str(e)}"

def process_document(file_path):
    """Process any supported document type."""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.pdf':
        return process_pdf(file_path)
    elif file_ext == '.docx':
        return process_docx(file_path)
    elif file_ext == '.csv':
        return process_csv(file_path)
    else:
        return f"Unsupported file type: {file_ext}"

def add_to_knowledge_base(file_path, vector_db, vector_db_path, document_id=None):
    """Process document and add to knowledge base."""
    if document_id is None:
        document_id = str(uuid.uuid4())
    
    # Extract text
    full_text = process_document(file_path)
    if full_text.startswith("Error") or full_text.startswith("Unsupported"):
        return {"status": "error", "message": full_text}
    
    # Chunk the text
    chunks = chunk_text(full_text)
    
    # Create metadata for each chunk
    metadata_list = [{"source": os.path.basename(file_path), "doc_id": document_id, "chunk_id": i} for i in range(len(chunks))]
    
    # Get embeddings for all chunks
    embeddings = get_embeddings(chunks)
    
    # Add to vector database
    vector_db.add(chunks, embeddings, metadata_list)
    
    # Save the updated database
    vector_db.save(vector_db_path)
    
    return {"status": "success", "message": f"Added {len(chunks)} chunks to knowledge base", "doc_id": document_id}