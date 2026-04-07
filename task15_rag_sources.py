# ─────────────────────────────────────────────────────────────
# TASK 15 — RAG with Source Attribution
# ─────────────────────────────────────────────────────────────
"""
TASK 15: RAG with Source Attribution
---------------------------------------
Extend the RAG pipeline to also return the source documents
used to generate the answer.  Return a dict:
  {
    "answer" : str,
    "sources": [{"content": str, "score": float}, ...]
  }

HINT:
  Use RunnableParallel to run retrieval and generation
  in parallel, or retrieve docs first and pass them to both
  the formatter and the chain:

  from langchain_core.runnables import RunnableParallel, RunnablePassthrough

  retrieval_chain = RunnableParallel(
      {"context": retriever, "question": RunnablePassthrough()}
  )
  # Then use the context in both the answer chain and as sources.
"""
from dotenv import load_dotenv

load_dotenv()
RAG_DOCUMENTS = [
    "LangChain v0.2 introduced LangChain Expression Language (LCEL) for composing chains.",
    "pgvector is a PostgreSQL extension supporting L2, inner product, and cosine distance.",
    "LangSmith provides tracing for every LLM call including token counts and latency.",
    "RAG stands for Retrieval-Augmented Generation and improves factual accuracy of LLMs.",
    "OpenAI's text-embedding-3-small produces 1536-dimensional embedding vectors.",
    "LangChain agents use a ReAct loop: Thought → Action → Observation → Answer.",
]

def rag_with_sources(documents: list, question: str) -> dict:
    """Returns the answer AND the source documents used."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

    import os
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import PGVector
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    conn_str = os.getenv("PG_CONNECTION_STRING")
    if not conn_str:
        raise ValueError("PG_CONNECTION_STRING is not set")

    docs = [Document(page_content=d) for d in documents]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    store = PGVector(
        connection_string=conn_str,
        embedding_function=embeddings,
        collection_name="lc_documents_rag_sources",
    )

    store.add_documents(docs)

    retrieved = store.similarity_search_with_score(question, k=3)

    retrieved_docs = [doc for doc, score in retrieved]

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    context = format_docs(retrieved_docs)

    prompt = ChatPromptTemplate.from_template(
        "Answer using only this context:\n{context}\n\nQuestion: {question}"
    )

    llm = ChatOpenAI(model="gpt-4o-mini")

    chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke({"context": context, "question": question})

    sources = []
    for doc, score in retrieved:
        sources.append({
            "content": doc.page_content,
            "score": float(score)
        })

    return {
        "answer": answer,
        "sources": sources
    }

    # ── END OF YOUR CODE ─────────────────────────────────────

print("\n[Task 15] RAG with Source Attribution")
rag_src = rag_with_sources(
    RAG_DOCUMENTS, "What distance metrics does pgvector support?"
)
print("  Answer  :", rag_src.get("answer", ""))
print("  Sources :")
for s in rag_src.get("sources", []):
    print(f"    [{s.get('score', 0):.4f}] {s.get('content', '')[:60]}")