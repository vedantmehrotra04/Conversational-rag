from app.loaders.pdf_loader import load_documents

from app.embeddings.embedding_model import (
    embedding_model
)

from langchain_qdrant import QdrantVectorStore


docs = load_documents("data/sample.pdf")


vector_store = QdrantVectorStore.from_documents(
    documents=docs,
    embedding=embedding_model,
    path="./qdrant_data",
    collection_name="research_docs"
)

print("Documents indexed successfully")