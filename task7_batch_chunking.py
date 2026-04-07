"""
TASK 7 — Batch Embedding with Chunking
"""
from dotenv import load_dotenv

load_dotenv()
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
# ─────────────────────────────────────────────────────────────
# TASK 7 — Batch Embedding with Chunking
# ─────────────────────────────────────────────────────────────
"""
TASK 7: Batch Embedding with Chunking
----------------------------------------
Given a long text document, split it into overlapping chunks
using RecursiveCharacterTextSplitter, then embed all chunks
in a single batch call.  Return:
  {
    "num_chunks"   : int,
    "chunk_size"   : int,   # configured chunk size
    "overlap"      : int,   # configured overlap
    "embedding_dim": int,
    "chunks"       : list[str]
  }

Use chunk_size=200, chunk_overlap=40.

HINT:
  from langchain.text_splitter import RecursiveCharacterTextSplitter
  splitter = RecursiveCharacterTextSplitter(
      chunk_size=200, chunk_overlap=40
  )
  chunks = splitter.split_text(long_text)
  vectors = embeddings.embed_documents(chunks)
"""

SAMPLE_DOCUMENT = """
LangChain is a framework for developing applications powered by language models.
It provides tools for prompt management, chains, agents, and memory.
LangChain integrates with many LLM providers including OpenAI, Anthropic, and Cohere.
The framework also supports vector stores, document loaders, and output parsers.
RAG (Retrieval-Augmented Generation) is a technique that enhances LLM responses
by fetching relevant documents from a knowledge base at query time.
pgvector is a PostgreSQL extension that enables efficient storage and similarity
search of high-dimensional vector embeddings directly inside a relational database.
LangSmith is an observability platform for LangChain applications that provides
tracing, evaluation, and debugging of LLM pipelines.
"""


def batch_embed_with_chunks(text: str, chunk_size: int, overlap: int) -> dict:
    """Splits text into chunks, embeds them, and returns metadata."""
    # ── YOUR CODE BELOW ──────────────────────────────────────
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )

    chunks = splitter.split_text(text)

    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectors = embeddings.embed_documents(chunks)

    return {
        "num_chunks": len(chunks),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "embedding_dim": len(vectors[0]) if vectors else 0,
        "chunks": chunks,
    }

    # ── END OF YOUR CODE ─────────────────────────────────────


def batch_embed_with_chunks(text, chunk_size=200, overlap=40):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    chunks = splitter.split_text(text)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectors = embeddings.embed_documents(chunks)
    return {
        "num_chunks": len(chunks),
        "embedding_dim": len(vectors[0]) if vectors else 0,
    }


print("\n[Task 7] Batch Embedding with Chunking")
chunk_info = batch_embed_with_chunks(SAMPLE_DOCUMENT, 200, 40)
print(f"  Chunks     : {chunk_info.get('num_chunks')}")
print(f"  Embed dims : {chunk_info.get('embedding_dim')}")
