# ─────────────────────────────────────────────────────────────
# TASK 5 — Generate and Inspect Embeddings
# ─────────────────────────────────────────────────────────────
"""
TASK 5: Generate and Inspect Embeddings
-----------------------------------------
Use OpenAIEmbeddings (text-embedding-3-small) to embed a list
of sentences. Return a dict with:
  {
    "num_sentences" : int,
    "embedding_dim" : int,
    "first_5_values": list[float],   # first 5 values of sentence[0]
    "vectors"       : list[list[float]]
  }

sentences = [
  "LangChain simplifies LLM application development.",
  "pgvector adds vector search to PostgreSQL.",
  "RAG grounds language models with external knowledge.",
]

HINT:
  from langchain_openai import OpenAIEmbeddings
  embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
  vectors = embeddings.embed_documents(sentences)
  A single vector is a plain Python list of floats.
"""
from dotenv import load_dotenv

load_dotenv()

def generate_embeddings(sentences: list) -> dict:
    """Embeds a list of sentences and returns metadata + vectors."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectors = embeddings.embed_documents(sentences)
    result = {
        "num_sentences": len(sentences),
        "embedding_dim": len(vectors[0]) if vectors else 0,
        "first_5_values": vectors[0][:5] if vectors else [],
        "vectors": vectors,
    }
    return result

    # ── END OF YOUR CODE ─────────────────────────────────────