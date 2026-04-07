# ─────────────────────────────────────────────────────────────
# TASK 14 — Basic RAG Pipeline
# ─────────────────────────────────────────────────────────────
"""
TASK 14: Basic RAG Pipeline
------------------------------
Build an end-to-end RAG chain that:
  1. Loads documents from a list of strings.
  2. Stores them in a PGVector vectorstore.
  3. Creates a retriever (top-3 results).
  4. Passes retrieved context + question to ChatOpenAI.
  5. Returns the final answer string.

Use the LCEL pattern:
  chain = (
      {"context": retriever | format_docs, "question": RunnablePassthrough()}
      | prompt
      | llm
      | StrOutputParser()
  )

HINT:
  def format_docs(docs):
      return "\n\n".join(doc.page_content for doc in docs)

  prompt = ChatPromptTemplate.from_template(
      "Answer using only this context:\n{context}\n\nQuestion: {question}"
  )
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


def basic_rag_pipeline(documents: list, question: str) -> str:
    """Indexes documents and answers the question using RAG."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

    import os
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import PGVector
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    conn_str = os.getenv("PG_CONNECTION_STRING")
    if not conn_str:
        raise ValueError("PG_CONNECTION_STRING is not set")

    docs = [Document(page_content=d) for d in documents]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    store = PGVector(
        connection_string=conn_str,
        embedding_function=embeddings,
        collection_name="lc_documents_rag",
    )
    store.add_documents(docs)
    retriever = store.as_retriever(search_kwargs={"k": 3})
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_template(
        "Answer using only this context:\n{context}\n\nQuestion: {question}"
    )

    llm = ChatOpenAI(model="gpt-4o-mini")

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    answer = chain.invoke(question)
    return answer

    # ── END OF YOUR CODE ─────────────────────────────────────

print("[Task 14] Basic RAG Pipeline")
rag_ans = basic_rag_pipeline(RAG_DOCUMENTS, "What is LCEL?")
print(" ", rag_ans)    