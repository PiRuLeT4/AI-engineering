from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
from vector_db import save_emb, get_results, inspect_db

load_dotenv()  

client = OpenAI()

DOCS_DIR = Path(__file__).parent / "docs"

#.. Document chunking
def chunk_text(text, chunk_size=80, overlap=20):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start += chunk_size - overlap
    
    return chunks

#.. Embeddings
def embed(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

#.. Indexing, ingest docs into the vector db
def ingest_docs():
    ids, chunks, embeddings, metadatas = [], [], [], []

    for file_path in sorted(DOCS_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")
        file_chunks = chunk_text(text)

        for i, chunk in enumerate(file_chunks):
            ids.append(f"{file_path.stem}_chunk_{i}")
            chunks.append(chunk)
            embeddings.append(embed(chunk))
            metadatas.append({"source": file_path.name})

    save_emb(ids, chunks, embeddings, metadatas)


def generate(prompt):
    response = client.chat.completions.create(
        model="gpt-6-luna",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

def ask(question, top_k=3):
    q_emb = embed(question)
    results = get_results(q_emb, n_results=top_k)

    retrieved_docs = results["documents"][0]
    retrieved_meta = results["metadatas"][0]

    context = "\n\n".join(
        f"[Source: {meta['source']}]\n{doc}"
        for doc, meta in zip(retrieved_docs, retrieved_meta)
    )

    prompt = (
        "Eres el asistente del Taller Roble & Viruta"
        "Responde SOLO usando el contexto proporcionado y la cita de la fuente"
        "Si la respuesta no esta en el contexto, di que no tienes informacion.\n\n"
        f"Contexto: {context}\n\n"
        f"Pregunta: {question}"
    )

    return generate(prompt)

if __name__=="__main__":
    ingest_docs()
    inspect_db()

    preguntas = [
        "Puedo meter la tabla de cortar en el lavavajillas o usar aceite de oliva?",
        "Si pido unos posavasos con mis iniciales grabadas, puedo devolverlos?",
        "Haceis envios a Espania?"
    ]

    for q in preguntas:
        print(f"\nQ: {q}")
        print(f"A: {ask(q)}")