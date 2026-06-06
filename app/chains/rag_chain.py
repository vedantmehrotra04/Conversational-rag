from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import (
    StrOutputParser
)

from langchain_google_genai import (
    ChatGoogleGenerativeAI
)

from app.retrievers.hybrid_retriever import (
    hybrid_retriever
)


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)


prompt = PromptTemplate.from_template(
    """
You are a helpful AI assistant.

Answer the question ONLY using the context below.

Context:
{context}

Question:
{question}

Answer:
"""
)


def format_docs(docs):

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


rag_chain = (
    {
        "context": hybrid_retriever | format_docs,
        "question": lambda x: x
    }
    | prompt
    | llm
    | StrOutputParser()
)