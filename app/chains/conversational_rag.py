from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_core.output_parsers import (
    StrOutputParser
)

from langchain_google_genai import (
    ChatGoogleGenerativeAI
)

from langchain_core.runnables.history import (
    RunnableWithMessageHistory
)

from langchain_classic.chains.history_aware_retriever import (
    create_history_aware_retriever
)

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)

from langchain_classic.chains.retrieval import (
    create_retrieval_chain
)

from app.retrievers.hybrid_retriever import (
    hybrid_retriever
)

from app.memory.memory_store import (
    get_session_history
)


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)


# ---------------------------------------------------
# Query Rewriting Prompt
# ---------------------------------------------------

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Given the chat history and latest user question,
rewrite the question into a standalone question.

Do NOT answer the question.
Only rewrite it if needed.
"""
    ),

    MessagesPlaceholder("chat_history"),

    (
        "human",
        "{input}"
    )
])


# ---------------------------------------------------
# History-Aware Retriever
# ---------------------------------------------------

history_aware_retriever = (
    create_history_aware_retriever(
        llm,
        hybrid_retriever,
        contextualize_q_prompt
    )
)


# ---------------------------------------------------
# QA Prompt
# ---------------------------------------------------

qa_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a production-grade AI assistant.

Answer ONLY using the provided context.

If the answer is not in the context,
say you don't know.

Context:
{context}
"""
    ),

    MessagesPlaceholder("chat_history"),

    (
        "human",
        "{input}"
    )
])


# ---------------------------------------------------
# Document Chain
# ---------------------------------------------------

question_answer_chain = (
    create_stuff_documents_chain(
        llm,
        qa_prompt
    )
)


# ---------------------------------------------------
# Retrieval Chain
# ---------------------------------------------------

rag_chain = create_retrieval_chain(
    history_aware_retriever,
    question_answer_chain
)


# ---------------------------------------------------
# Conversational Wrapper
# ---------------------------------------------------

conversational_rag_chain = (
    RunnableWithMessageHistory(
        rag_chain,

        get_session_history,

        input_messages_key="input",

        history_messages_key="chat_history",

        output_messages_key="answer"
    )
)