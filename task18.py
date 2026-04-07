# ─────────────────────────────────────────────────────────────
# TASK 18 — Trace a Chain with LangSmith
# ─────────────────────────────────────────────────────────────
"""
TASK 18: LangSmith Tracing
-----------------------------
Instrument a simple LCEL chain so every invocation is
traced in LangSmith.  Your function should:
  1. Set LANGCHAIN_TRACING_V2=true and LANGCHAIN_PROJECT.
  2. Build the same basic LCEL chain from Task 1.
  3. Add run_name and tags to the invocation config.
  4. Return the response AND the run_id of the trace.

Expected return:
  {"answer": str, "run_id": str}

HINT:
  from langchain_core.tracers.context import collect_runs

  with collect_runs() as cb:
      result = chain.invoke(
          {"topic": topic},
          config={"run_name": "task18_trace", "tags": ["challenge"]}
      )
  run_id = str(cb.traced_runs[0].id)
"""


def traced_chain(topic: str) -> dict:
    """Runs a chain with LangSmith tracing. Returns answer and run_id."""
    # ── YOUR CODE BELOW ──────────────────────────────────────
    import os
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.tracers.context import collect_runs

    os.environ["LANGCHAIN_TRACING"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "langchain-examples-1"
    llm = ChatOpenAI(model="gpt-4o-mini")

    prompt = ChatPromptTemplate.from_template(
        "Explain {topic} in simple terms."
    )

    chain = prompt | llm

    with collect_runs() as cb:
        result = chain.invoke(
            {"topic": topic},
            config={"run_name": "task18_trace", "tags": ["challenge"]}
        )

    run_id = str(cb.traced_runs[0].id)

    return {
        "answer": result.content,
        "run_id": run_id
    }

    

    # ── END OF YOUR CODE ─────────────────────────────────────

print("[Task 18] Traced Chain")
traced = traced_chain("embeddings")
print(f"  Answer : {str(traced.get('answer', ''))[:80]}")
print(f"  Run ID : {traced.get('run_id')}")
