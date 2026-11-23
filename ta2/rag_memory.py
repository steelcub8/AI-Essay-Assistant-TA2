
import chromadb
import requests

DB_PATH = "./chroma_db"
COLLECTION_NAME = "essay_memory"

client = chromadb.PersistentClient(path=DB_PATH)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


def get_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "nomic-embed-text",
            "input": text
        },
        timeout=120
    )

    response.raise_for_status()
    return response.json()["embeddings"][0]


def save_memory(text, memory_id):
    embedding = get_embedding(text)

    collection.upsert(
        ids=[memory_id],
        documents=[text],
        embeddings=[embedding]
    )

    print("Memory saved:", text)


def search_memory(query, limit=3):
    if collection.count() == 0:
        return []

    embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(limit, collection.count())
    )

    return results["documents"][0]


if __name__ == "__main__":
    save_memory(
        "The user prefers simple language and short paragraphs.",
        "writing_style_1"
    )

    save_memory(
        "The user prefers a clear introduction and logical flow.",
        "writing_style_2"
    )

    print("\nRetrieved memories:")

    for memory in search_memory("How should an essay be written?"):
        print("-", memory)
