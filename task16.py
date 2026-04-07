# ─────────────────────────────────────────────────────────────
# TASK 16 — Conversational RAG with Chat History
# ─────────────────────────────────────────────────────────────
"""
TASK 16: Conversational RAG
------------------------------
Build a RAG pipeline that is aware of conversation history.

Requirements:
  - Use create_history_aware_retriever to rephrase follow-up
    questions into standalone queries.
  - Use create_retrieval_chain + create_stuff_documents_chain
    to answer with context.
  - Run a 2-turn conversation:
      Turn 1: "What is LangChain?"
      Turn 2: "What version introduced LCEL?"  ← follow-up
  - Return both answers as a list: [answer1, answer2]

HINT:
  from langchain.chains import create_history_aware_retriever
  from langchain.chains import create_retrieval_chain
  from langchain.chains.combine_documents import create_stuff_documents_chain
  from langchain_core.messages import HumanMessage, AIMessage

  contextualize_prompt — asks the LLM to rephrase the question
                         given history.
  qa_prompt           — answers based on context + history.
"""


def conversational_rag(documents: list) -> list:
    """Returns [answer_turn1, answer_turn2] for a 2-turn RAG conversation."""
    # ── YOUR CODE BELOW ──────────────────────────────────────
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_community.vectorstores import PGVector
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
    from langchain_classic.chains.retrieval import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_core.documents import Document
    import os

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docs = [Document(page_content=text) for text in documents]
    vectorstore = PGVector.from_documents(
        docs,
        embedding=embeddings,
        connection_string=os.getenv("PG_CONNECTION_STRING"),
    )
    retriever = vectorstore.as_retriever()

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rephrase the question to be standalone."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using the context:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        qa_chain
    )

    chat_history = []

    res1 = rag_chain.invoke({
        "input": "What is LangChain?",
        "chat_history": chat_history
    })
    answer1 = res1["answer"]

    chat_history.extend([
        HumanMessage(content="What is LangChain?"),
        AIMessage(content=answer1)
    ])

    res2 = rag_chain.invoke({
        "input": "What version introduced LCEL?",
        "chat_history": chat_history
    })
    answer2 = res2["answer"]

    return [answer1, answer2]

    

    # ── END OF YOUR CODE ─────────────────────────────────────

print("\n[Task 16] Conversational RAG")
conv_answers = conversational_rag(RAG_DOCUMENTS)
print("  Turn 1:", conv_answers[0][:80] if conv_answers else "")
print("  Turn 2:", conv_answers[1][:80] if len(conv_answers) > 1 else "")