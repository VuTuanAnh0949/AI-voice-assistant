# Feature: Voice-based Q&A using Whisper + ChromaDB + Falcon LLM

import whisper
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import pipeline

def voice_query_to_answer(audio_path, documents):
    # Transcribe audio input to text
    whisper_model = whisper.load_model("base")
    question = whisper_model.transcribe(audio_path)["text"]

    # Embed documents into a Chroma vector store
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    db = chromadb.Client()
    collection = db.create_collection("rag")
    for i, doc in enumerate(documents):
        collection.add(documents=[doc], embeddings=[embedder.encode(doc).tolist()], ids=[f"doc_{i}"])

    # Search for relevant context
    query_vec = embedder.encode(question).tolist()
    results = collection.query(query_embeddings=[query_vec], n_results=2)
    context = "\n".join(results["documents"][0])

    # Generate an answer using a language model
    llm = pipeline("text-generation", model="tiiuae/falcon-7b-instruct")
    prompt = f"Context:\n{context}\n\nQ: {question}\nA:"
    return llm(prompt, max_new_tokens=100)[0]["generated_text"]
