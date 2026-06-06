from app.chains.conversational_rag import (
    conversational_rag_chain
)


config = {
    "configurable": {
        "session_id": "user_1"
    }
}


while True:

    query = input("\nYou: ")

    if query.lower() == "exit":
        break

    result = conversational_rag_chain.invoke(
        {
            "input": query
        },
        config=config
    )

    print("\nAI:", result["answer"])