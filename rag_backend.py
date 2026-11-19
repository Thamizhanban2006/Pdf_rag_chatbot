import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / ".env")
api_key = os.getenv("GOOGLE_API_KEY")


def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_llm():
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY in PDF_RAG_CHATBOT/.env")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0,
    )


def split_text(text, chunk_size=300, overlap=50):
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")

    chunks = []
    start = 0
    while start < len(text):
        part = text[start:start + chunk_size].strip()
        if part:
            chunks.append(part)
        start += chunk_size - overlap
    return chunks


def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if not text.strip():
        raise ValueError("No selectable text found. Scanned PDFs require OCR.")
    return text


def save_uploaded_pdf(uploaded_file):
    safe_name = Path(uploaded_file.name).name
    destination = DATA_DIR / safe_name
    destination.write_bytes(uploaded_file.getvalue())
    return destination


def build_collection(file_name, uploaded_file, embedder):
    save_uploaded_pdf(uploaded_file)
    raw_text = extract_pdf_text(uploaded_file)
    chunks = split_text(raw_text)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection_name = "pdf_chatbot"

    try:
        existing = client.get_collection(collection_name)
        client.delete_collection(collection_name)
        if existing:
            pass
    except Exception:
        pass

    collection = client.get_or_create_collection(collection_name)
    embeddings = embedder.encode(chunks).tolist()
    collection.upsert(
        ids=[f"{file_name}-chunk-{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"file_name": file_name, "chunk": i} for i in range(len(chunks))],
    )
    return collection, chunks


def answer_question(collection, embedder, llm, question, k=3):
    found = collection.query(
        query_embeddings=embedder.encode([question]).tolist(),
        n_results=min(k, collection.count()),
    )
    evidence = found["documents"][0] if found["documents"] else []
    context = "\n\n".join(evidence)
    prompt = f"""Answer using ONLY the context below.
If the answer is absent, say you cannot find it in the document.
Do not invent document details.

CONTEXT:
{context}

QUESTION: {question}"""
    response = llm.invoke(prompt)
    return response.content, evidence
