from langchain_classic.retrievers import (
    EnsembleRetriever
)

from langchain_community.retrievers import BM25Retriever

from langchain_qdrant import QdrantVectorStore

from app.embeddings.embedding_model import (
    embedding_model
)
from app.loaders.pdf_loader import load_documents

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    path="./qdrant_data",
    collection_name="research_docs"
)

# Dense Retriever
dense_retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)

# Sparse Retriever

docs = load_documents("data/sample.pdf")

bm25_retriever = BM25Retriever.from_documents(docs)

bm25_retriever.k = 4

hybrid_retriever = EnsembleRetriever(
    retrievers=[
        dense_retriever,
        bm25_retriever
    ],
    weight=[0.7,0.3]
)

