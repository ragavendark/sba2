# ─────────────────────────────────────────────────────────────
# TASK 8 — Compare Two Embedding Models
# ─────────────────────────────────────────────────────────────
"""
TASK 8: Compare Two Embedding Models
--------------------------------------
Embed the same sentence using two different OpenAI models:
  Model A: text-embedding-3-small   (1536 dims)
  Model B: text-embedding-3-large   (3072 dims)

For the sentence:  "Vector databases power semantic search."

Return a dict:
  {
    "sentence"   : str,
    "model_a"    : {"model": str, "dims": int, "first_3": list[float]},
    "model_b"    : {"model": str, "dims": int, "first_3": list[float]},
    "dim_ratio"  : float   # model_b_dims / model_a_dims
  }

HINT:
  OpenAIEmbeddings(model="text-embedding-3-small")
  OpenAIEmbeddings(model="text-embedding-3-large")
  embeddings.embed_query(sentence) → single vector (list of floats)
"""

from dotenv import load_dotenv

load_dotenv()

def compare_embedding_models(sentence: str) -> dict:
    """Embeds a sentence with two models and compares their dimensions."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

    from langchain_openai import OpenAIEmbeddings

    model_a = OpenAIEmbeddings(model="text-embedding-3-small")
    model_b = OpenAIEmbeddings(model="text-embedding-3-large")

    vec_a = model_a.embed_query(sentence)
    vec_b = model_b.embed_query(sentence)

    result = {
        "sentence": sentence,
        "model_a": {
            "model": "text-embedding-3-small",
            "dims": len(vec_a),
            "first_3": vec_a[:3],
        },
        "model_b": {
            "model": "text-embedding-3-large",
            "dims": len(vec_b),
            "first_3": vec_b[:3],
        },
        "dim_ratio": len(vec_b) / len(vec_a),
    }

    return result


print("\n[Task 8] Compare Embedding Models")
model_cmp = compare_embedding_models("Vector databases power semantic search.")
print(f"  Model A dims : {model_cmp.get('model_a', {}).get('dims')}")
print(f"  Model B dims : {model_cmp.get('model_b', {}).get('dims')}")
print(f"  Dim ratio    : {model_cmp.get('dim_ratio')}")