# ─────────────────────────────────────────────────────────────
# TASK 17 — RAG Agent (Tool-based Retrieval)
# ─────────────────────────────────────────────────────────────
"""
TASK 17: RAG Agent with Retriever as Tool
-------------------------------------------
Convert the vector store retriever into a LangChain Tool,
then wrap it in a ReAct agent.  This lets the agent DECIDE
when to retrieve rather than always retrieving.

Steps:
  1. Build a PGVector store from RAG_DOCUMENTS.
  2. Wrap the retriever in a Tool named "knowledge_base".
  3. Create a ReAct agent with that tool.
  4. Ask: "What distance metrics does pgvector support?"
  5. Return the final answer string.

HINT:
  from langchain.tools.retriever import create_retriever_tool
  retriever_tool = create_retriever_tool(
      retriever,
      name="knowledge_base",
      description="Search the knowledge base for technical info."
  )
  Then pass [retriever_tool] to create_react_agent.
"""


def rag_agent(question: str) -> str:
    """Uses a ReAct agent with a retriever tool to answer the question."""
    # ── YOUR CODE BELOW ──────────────────────────────────────
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_community.vectorstores import PGVector
    from langchain_classic.tools.retriever import create_retriever_tool
    from langchain_classic.agents import AgentExecutor, create_react_agent
    from langchain_core.documents import Document
    from langchain_core.prompts import PromptTemplate
    import os

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docs = [Document(page_content=text) for text in RAG_DOCUMENTS]
    vectorstore = PGVector.from_documents(
        docs,
        embedding=embeddings,
        connection_string=os.getenv("PG_CONNECTION_STRING"),
    )

    retriever = vectorstore.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        name="knowledge_base",
        description="Search the knowledge base for technical info."
    )
    

    prompt1 = PromptTemplate.from_template(
        """You are a helpful assistant that can use tools to answer questions.

           You have access to the following tools:
           {tools}

            Use the following format:

            Question: {input}
            Thought: you should think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ...(this Thought/Action/Observation can repeat)
            Thought: I now know the final answer
            Final Answer: the final answer to the user

            {agent_scratchpad}
        """
    )

    agent = create_react_agent(
        llm,
        tools=[retriever_tool],
        prompt=prompt1
    )

    executor = AgentExecutor(agent=agent, tools=[retriever_tool], verbose=False)

    result = executor.invoke({"input": question})

    return result["output"]

    

    # ── END OF YOUR CODE ─────────────────────────────────────

print("\n[Task 17] RAG Agent")
agent_ans = rag_agent("What distance metrics does pgvector support?")
print(" ", agent_ans)
