"""
graph.py
This is where everything gets wired together using LangGraph.

Flow:
  User question -> Supervisor (decides needs_sql / needs_rag)
                 -> SQL Agent (if needed) and/or RAG Agent (if needed), in parallel
                 -> Combine node (merges results into one final answer)

Run this file directly to test the full pipeline end-to-end.
"""

import asyncio
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from agents.supervisor import route_question
from agents.sql_agents import ask_sql_agent
from agents.rag_agents import ask_rag_agent
from anthropic import Anthropic
import os
from dotenv import load_dotenv

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# ---- 1. Shared state ----
# This is the object that gets passed between every node in the graph.
# Each node reads from it and writes new fields into it.
class AgentState(TypedDict):
    question: str
    needs_sql: bool
    needs_rag: bool
    sql_result: Optional[str]
    sql_query: Optional[str]
    rag_result: Optional[str]
    final_answer: Optional[str]


# ---- 2. Node functions ----
# Each node is just a normal Python function: takes the state in, returns
# updated fields out. LangGraph handles calling them in the right order.

def supervisor_node(state: AgentState) -> dict:
    """Decides which agent(s) are needed for this question."""
    decision = route_question(state["question"])
    print(f"[Supervisor] needs_sql={decision.needs_sql}, needs_rag={decision.needs_rag}")
    print(f"[Supervisor] reasoning: {decision.reasoning}")
    return {"needs_sql": decision.needs_sql, "needs_rag": decision.needs_rag}


def sql_node(state: AgentState) -> dict:
    """Runs the SQL Agent (async under the hood, wrapped here for LangGraph)."""
    result = asyncio.run(ask_sql_agent(state["question"]))
    print(f"[SQL Agent] generated SQL: {result['sql_query']}")
    return {"sql_result": result["result"], "sql_query": result["sql_query"]}


def rag_node(state: AgentState) -> dict:
    """Runs the RAG Agent."""
    result = ask_rag_agent(state["question"])
    print(f"[RAG Agent] retrieved {len(result['retrieved_docs'])} chunks")
    return {"rag_result": result["answer"]}


COMBINE_PROMPT = """You are a helpful personal finance assistant.
Write ONE clear, natural answer to the user's question using the information below.
If only one source is present, just use that one. Do not mention "SQL" or "database" 
or "knowledge base" explicitly — just answer naturally as a knowledgeable assistant.

User question: {question}

{sql_section}
{rag_section}

Final answer:"""


def combine_node(state: AgentState) -> dict:
    """Merges SQL and/or RAG results into one final natural-language answer."""
    sql_section = f"Data from records: {state['sql_result']}" if state.get("sql_result") else ""
    rag_section = f"Relevant knowledge: {state['rag_result']}" if state.get("rag_result") else ""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=400,
        messages=[
            {
                "role": "user",
                "content": COMBINE_PROMPT.format(
                    question=state["question"],
                    sql_section=sql_section,
                    rag_section=rag_section,
                ),
            }
        ],
    )
    final_answer = response.content[0].text.strip()
    print(f"[Combine] final answer generated")
    return {"final_answer": final_answer}


# ---- 3. Routing logic ----
# This function tells LangGraph which node(s) to go to after the supervisor,
# based on the decision it made. Returning a LIST means both branches run.
def route_after_supervisor(state: AgentState):
    next_nodes = []
    if state["needs_sql"]:
        next_nodes.append("sql_agent")
    if state["needs_rag"]:
        next_nodes.append("rag_agent")
    return next_nodes if next_nodes else ["combine"]  # fallback safety


# ---- 4. Build the graph ----
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("sql_agent", sql_node)
    graph.add_node("rag_agent", rag_node)
    graph.add_node("combine", combine_node)

    graph.set_entry_point("supervisor")

    # Conditional edges: supervisor can fan out to sql_agent, rag_agent, or both
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        ["sql_agent", "rag_agent", "combine"],
    )

    # Both branches feed into combine
    graph.add_edge("sql_agent", "combine")
    graph.add_edge("rag_agent", "combine")
    graph.add_edge("combine", END)

    return graph.compile()


# ---- Standalone test ----
if __name__ == "__main__":
    app = build_graph()

    test_questions = [
        "How much did I spend on dining in total?",
        "What's a good rule of thumb for emergency savings?",
        "How much am I spending on subscriptions, and is that a healthy amount?",
    ]

    for q in test_questions:
        print("\n" + "=" * 60)
        print(f"QUESTION: {q}")
        print("=" * 60)
        result = app.invoke({"question": q})
        print(f"\nFINAL ANSWER:\n{result['final_answer']}")