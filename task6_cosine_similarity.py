
# ─────────────────────────────────────────────────────────────
# TASK 6 — Cosine Similarity (from scratch, then with numpy)
# ─────────────────────────────────────────────────────────────
"""
TASK 6: Cosine Similarity
---------------------------
Part A: Implement cosine_similarity_manual(v1, v2) WITHOUT
        using numpy.  Use only Python loops / math.
Part B: Implement cosine_similarity_numpy(v1, v2) using numpy.

Both should return a float between -1 and 1.

Then embed these two pairs and print which pair is more similar:
  Pair 1: "dog" vs "puppy"
  Pair 2: "dog" vs "automobile"

Formula:
  cosine_similarity = (v1 · v2) / (||v1|| × ||v2||)

HINT:
  dot product: sum(a*b for a, b in zip(v1, v2))
  magnitude  : sum(x**2 for x in v) ** 0.5
  numpy equiv: np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
"""

import math

import numpy as np
from dotenv import load_dotenv

load_dotenv()
from tasks.task_5_get_embeddings import generate_embeddings


def cosine_similarity_manual(v1: list, v2: list) -> float:
    """Computes cosine similarity using pure Python."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

    dot_product = sum(a * b for a, b in zip(v1, v2))
    mag_v1 = math.sqrt(sum(x**2 for x in v1))
    mag_v2 = math.sqrt(sum(x**2 for x in v2))
    if mag_v1 == 0 or mag_v2 == 0:
        return 0.0

    return dot_product / (mag_v1 * mag_v2)

    # ── END OF YOUR CODE ─────────────────────────────────────


def cosine_similarity_numpy(v1: list, v2: list) -> float:
    """Computes cosine similarity using numpy."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

    v1 = np.array(v1)
    v2 = np.array(v2)

    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0

    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    # ── END OF YOUR CODE ─────────────────────────────────────


def compare_word_pairs() -> dict:
    """
    Embeds dog/puppy and dog/automobile, returns:
    {
      "dog_vs_puppy"      : float,
      "dog_vs_automobile" : float,
      "more_similar_pair" : str
    }
    """
    # ── YOUR CODE BELOW ──────────────────────────────────────

    dog = generate_embeddings("dog")["vectors"][0]
    puppy = generate_embeddings("puppy")["vectors"][0]
    automobile = generate_embeddings("automobile")["vectors"][0]

    sim_dog_puppy = cosine_similarity_numpy(dog, puppy)
    sim_dog_auto = cosine_similarity_numpy(dog, automobile)

    more_similar = (
        "dog_vs_puppy" if sim_dog_puppy > sim_dog_auto else "dog_vs_automobile"
    )

    return {
        "dog_vs_puppy": sim_dog_puppy,
        "dog_vs_automobile": sim_dog_auto,
        "more_similar_pair": more_similar,
    }
    # ── END OF YOUR CODE ─────────────────────────────────────


print("\n[Task 6] Cosine Similarity")
word_pairs = compare_word_pairs()
print(f"  dog vs puppy      : {word_pairs.get('dog_vs_puppy', ''):.4f}")
print(f"  dog vs automobile : {word_pairs.get('dog_vs_automobile', ''):.4f}")
print(f"  More similar      : {word_pairs.get('more_similar_pair')}")