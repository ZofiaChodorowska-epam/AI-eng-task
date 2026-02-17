import os
import json
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Persistence directory
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "../data/chroma_db")
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/static/parking_info.json")

def get_vectorstore():
    # In a real scenario, we'd use a real embedding model.
    # For this stage, we might need an API key or use a local one.
    # If OPENAI_API_KEY is not set, this will fail. Use Fake if needed for strict offline dev,
    # but the requirement implies a "working chatbot". I'll default to OpenAI but warn if missing.
    try:
        if "OPENAI_API_KEY" not in os.environ:
             # Fallback or error. For this stage, let's assume user provides it or we use a localized generic embedding.
             # Actually, for "intelligent" chatbot, we usually need a real model.
             # I'll use a dummy/open source one if I can, but langchain-chroma defaults often require one.
             # Let's try to use 'all-MiniLM-L6-v2' via HuggingFace or just OpenAI if env is there.
             # For simplicity in this environment without forcing user to give keys, I'll use a local HF model if possible,
             # but `sentence-transformers` might be heavy.
             # Let's stick to OpenAIEmbeddings and assume the user puts the key in .env or environment.
             pass
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    except Exception as e:
        print(f"Warning: Could not init OpenAIEmbeddings: {e}")
        # Return none or raise, but let's try to proceed.
        embeddings = None
        
    if embeddings is None:
         # Fallback for testing without keys
        from langchain_community.embeddings import FakeEmbeddings
        embeddings = FakeEmbeddings(size=1536)

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    return vectorstore

def ingest_data():
    if not os.path.exists(DATA_PATH):
        print(f"Data file not found at {DATA_PATH}")
        return

    with open(DATA_PATH, 'r') as f:
        data = json.load(f)

    documents = []
    for item in data:
        doc = Document(
            page_content=f"{item['topic']}: {item['content']}",
            metadata={"topic": item['topic']}
        )
        documents.append(doc)

    vectorstore = get_vectorstore()
    vectorstore.add_documents(documents)
    print(f"Ingested {len(documents)} documents into ChromaDB.")

if __name__ == "__main__":
    ingest_data()
