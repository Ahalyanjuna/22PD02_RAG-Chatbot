import os
import numpy as np
import torch
import faiss
import json
from sentence_transformers import SentenceTransformer

# Text embedding model - SentenceTransformers
# Use FP16 if a compatible GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_model = SentenceTransformer('intfloat/e5-large-v2', device=device)
class VectorDatabase:
    def __init__(self, dim=None):
        dim = embedding_model.get_sentence_embedding_dimension()
        self.dimension = dim
        #self.index = faiss.IndexFlatL2(dim)
        self.index = faiss.IndexFlatIP(dim)
        self.texts = []
        self.metadata = []
        self.loaded = False

    
    def add(self, texts, embeddings, metadata_list=None):
        if metadata_list is None:
            metadata_list = [{} for _ in texts]
        
        # Convert to numpy array if not already
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.numpy()
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        self.texts.extend(texts)
        self.metadata.extend(metadata_list)
    
    def search(self, query_embedding, k=5):
        # Ensure query embedding is properly formatted
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.numpy()
            
        # Reshape if needed
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize query vector
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.texts):  # Safety check
                results.append({
                    'text': self.texts[idx],
                    'score': float(distances[0][i]),
                    'metadata': self.metadata[idx]
                })
        
        return results
    
    def save(self, path):
        # Save the index
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        
        # Save texts and metadata
        with open(os.path.join(path, "data.json"), 'w') as f:
            json.dump({'texts': self.texts, 'metadata': self.metadata}, f)
    
    def load(self, path):
        # Load the index
        self.index = faiss.read_index(os.path.join(path, "faiss.index"))
        
        # Load texts and metadata
        with open(os.path.join(path, "data.json"), 'r') as f:
            data = json.load(f)
            self.texts = data['texts']
            self.metadata = data['metadata']
        
        self.loaded = True
        return True

# Function to get embeddings for a list of texts
def get_embeddings(texts):
    """Generate embeddings for a list of texts using the sentence transformer model."""
    passages = [f"passage: {text}" for text in texts]
    return embedding_model.encode(passages, normalize_embeddings=True)

# Function to get embedding for a query
def get_query_embedding(query):
    """Generate embedding for a query string."""
    formatted_query = f"query: {query}"
    return embedding_model.encode(formatted_query, normalize_embeddings=True) 