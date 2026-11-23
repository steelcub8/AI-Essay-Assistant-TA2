import chromadb
import requests

print("Testing Ollama embeddings...")

response = requests.post(
    "http://localhost:11434/api/embed",
    json={
        "model": "nomic-embed-text",
        "input": "A good essay has a clear introduction."
    },
    timeout=120
)
response.raise_for_status()
embedding = response.json()["embeddings"][0]

print("Embedding works! Dimensions:", len(embedding))

client = chromadb.Client()
collection = client.create_collection("essay_style_test")

collection.add(
    ids=["style1"],
    documents=["Use simple language and short paragraphs."],
    embeddings=[embedding]
)

results = collection.query(
    query_embeddings=[embedding],
    n_results=1
)

print("Retrieved preference:", results["documents"][0][0])
print("RAG test passed!")
