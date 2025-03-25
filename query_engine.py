# Query Engine for Document RAG Chatbot

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering,
    pipeline
)
from vector_db_manager import get_query_embedding

device = "cuda" if torch.cuda.is_available() else "cpu"

# LLM for text generation - FLAN-T5-base (smaller free version)
tokenizer_llm = AutoTokenizer.from_pretrained("google/flan-t5-large")
model_llm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to(device)

# QA model for direct answers
qa_tokenizer = AutoTokenizer.from_pretrained("deepset/deberta-v3-large-squad2")
qa_model = AutoModelForQuestionAnswering.from_pretrained("deepset/deberta-v3-large-squad2").to(device)
qa_pipeline = pipeline('question-answering', model=qa_model, tokenizer=qa_tokenizer,device=0 if device == 'cuda' else -1)

# Answer Generation Function
def generate_answer(query, context_texts, max_length=5000):
    """Generate an answer using the LLM, encouraging list outputs when appropriate."""
    prompt = f"""
    Based on the following information, please answer the question.

    Question: {query}

    Information:
    {" ".join(context_texts)}

    If the answer is a list, provide all relevant items clearly and completely.

    Answer:
    """

    inputs = tokenizer_llm(prompt, return_tensors="pt", max_length=5000, truncation=True).to(device)

    outputs = model_llm.generate(
        inputs.input_ids,
        max_length=max_length,
        num_beams=7,            # Increase diversity
        temperature=0.4,        # Lower for factual accuracy
        repetition_penalty=1.2,# Reduce repetition
         do_sample=True, 
        early_stopping=True
    )

    return tokenizer_llm.decode(outputs[0], skip_special_tokens=True)

# Query Knowledge Base Function
def query_knowledge_base(query, vector_db, top_k=5):
    """Query the knowledge base and return an accurate, multi-item answer."""
    query_embedding = get_query_embedding(query)
    results = vector_db.search(query_embedding, k=top_k)

    if not results:
        return {
            "answer": "I don't have enough information to answer that question.",
            "confidence": None,
            "sources": [],
            "contexts": []
        }

    context_texts = [r['text'] for r in results]
    sources = [r['metadata'].get('source', 'unknown') for r in results]

    # Check if query needs a list (based on keywords)
    list_keywords = ["list", "items", "options", "examples", "steps"]
    needs_list = any(word in query.lower() for word in list_keywords)

    # If a list is expected, prefer the LLM
    if needs_list:
        answer = generate_answer(query, context_texts)
        confidence = None
    else:
        # Use QA model with the most relevant chunk
        context = context_texts[0]
        try:
            qa_result = qa_pipeline(question=query, context=context)
            answer = qa_result['answer']
            confidence = qa_result['score']
        except Exception as e:
            print(f"QA Error: {e}")
            answer = generate_answer(query, context_texts)
            confidence = None

    return {
        "answer": answer,
        "confidence": confidence,
        "sources": list(set(sources)),
        "contexts": context_texts
    }
