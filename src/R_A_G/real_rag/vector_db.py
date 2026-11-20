from pathlib import Path
import chromadb

DB_PATH = Path(__file__).parent / "chroma_db"

client = chromadb.PersistentClient(path=str(DB_PATH))
collection = client.get_or_create_collection(name="my_docs")

def save_emb(ids, chunks, embeddings, metadatas):
    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

def get_results(query_embedding, n_results=3):
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )


def inspect_db():
    print(f"\n---ChromaDB ({collection.count()} chunks guardados)---")
    data = collection.get(include=["documents", "embeddings", "metadatas"])
    for doc_id, doc, emb, meta in zip(
        data["ids"], data["documents"], data["embeddings"], data["metadatas"]
    ):
        print(f"{doc_id} | source={meta["source"]} | dims={len(emb)} | {doc[:55]}...")