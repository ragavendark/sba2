# ─────────────────────────────────────────────────────────────
# TASK 20 — Run an Evaluation with LangSmith
# ─────────────────────────────────────────────────────────────
"""
TASK 20: LangSmith Evaluation (evaluate)
------------------------------------------
Run an automated evaluation of your RAG pipeline using the
dataset created in Task 19.

Steps:
  1. Define a target function that takes a dict {"question": str}
     and returns {"answer": str} using the basic RAG pipeline.
  2. Define a custom evaluator that checks if the expected
     answer appears (case-insensitive) in the generated answer.
  3. Run the evaluation using langsmith.evaluate().
  4. Return the evaluation results summary dict:
     {"dataset": str, "num_examples": int, "pass_rate": float}

HINT:
  from langsmith.evaluation import evaluate, LangChainStringEvaluator

  def target(inputs: dict) -> dict:
      return {"answer": basic_rag_pipeline(RAG_DOCUMENTS, inputs["question"])}

  results = evaluate(
      target,
      data="rag-eval-dataset",
      evaluators=[...],
      experiment_prefix="rag-challenge-eval",
  )
"""


def run_langsmith_evaluation() -> dict:
    """Evaluates the RAG pipeline on the LangSmith dataset."""
    # ── YOUR CODE BELOW ──────────────────────────────────────

def run_langsmith_evaluation() -> dict:
    """Evaluates the RAG pipeline on the LangSmith dataset."""

    from langsmith.evaluation import evaluate
    from langsmith import Client

    client = Client()

    def target(inputs: dict) -> dict:
        answer = basic_rag_pipeline(RAG_DOCUMENTS, inputs["question"])
        return {"answer": answer}

    def contains_answer(run, example):
        predicted = run.outputs["answer"].lower()
        expected = example.outputs["answer"].lower()
        return {"score": float(expected in predicted)}

    results = evaluate(
        target,
        data="rag-eval-dataset",
        evaluators=[contains_answer],
        experiment_prefix="rag-challenge-eval",
    )

    summary = results._summary_results

    return {
        "dataset": "rag-eval-dataset",
        "num_examples": None,
        "pass_rate": None
    }


    # ── END OF YOUR CODE ─────────────────────────────────────

print("\n[Task 20] Run LangSmith Evaluation")
eval_summary = run_langsmith_evaluation()
print(f"  Dataset     : {eval_summary.get('dataset')}")
print(f"  # Examples  : {eval_summary.get('num_examples')}")
print(f"  Pass rate   : {eval_summary.get('pass_rate')}")
