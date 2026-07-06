"""
supervisor.py
This is the SUPERVISOR AGENT — the router.
 
It reads the user's question and decides, using the LLM, whether it needs:
  - SQL Agent (real transaction data)
  - RAG Agent (general finance knowledge)
  - BOTH (a question that needs real numbers AND context/advice)
 
This uses Claude's structured output (via a strict prompt + JSON parsing)
to return a clean, reliable decision that code can act on.
"""
 
import os
import json
from dotenv import load_dotenv
from anthropic import Anthropic
from pydantic import BaseModel
 
load_dotenv()
 
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
 
 
# ---- Structured output schema ----
# Using Pydantic means we get automatic validation: if the LLM's response
# doesn't match this shape, we'll know immediately instead of silently
# proceeding with bad data.
class RouteDecision(BaseModel):
    needs_sql: bool
    needs_rag: bool
    reasoning: str  # short explanation, useful for logging/debugging
 
 
ROUTER_PROMPT = """You are a routing agent for a personal finance assistant.
Decide which capability is needed to answer the user's question.
 
- needs_sql = true if the question requires looking up specific numbers, 
  amounts, dates, or counts from actual transaction data.
  Examples: "how much did I spend on X", "how many transactions in June"
 
- needs_rag = true if the question requires general finance knowledge, 
  advice, definitions, or rules of thumb.
  Examples: "what's a good savings rate", "how do I budget better"
 
- Both can be true if the question needs real data AND general advice/context.
  Example: "How much am I spending on dining, and is that healthy?"
 
User question: "{question}"
 
Respond with ONLY a valid JSON object, no other text, no markdown formatting, 
in exactly this format:
{{"needs_sql": true or false, "needs_rag": true or false, "reasoning": "brief explanation"}}
"""
 
 
def route_question(question: str) -> RouteDecision:
    """Calls Claude to classify the question, returns a validated RouteDecision."""
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=200,
        messages=[
            {"role": "user", "content": ROUTER_PROMPT.format(question=question)}
        ],
    )
 
    raw_text = response.content[0].text.strip()
    # Safety cleanup in case the model adds markdown fences anyway
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()
 
    # Parse the JSON text into a Python dict, then validate it against
    # our Pydantic schema. If the LLM returned something malformed,
    # this will raise a clear error instead of failing silently later.
    data = json.loads(raw_text)
    decision = RouteDecision(**data)
    return decision
 
 
# ---- Standalone test ----
if __name__ == "__main__":
    test_questions = [
        "How much did I spend on dining last month?",
        "What's a good rule of thumb for emergency savings?",
        "How much am I spending on subscriptions, and is that healthy?",
    ]
 
    for q in test_questions:
        decision = route_question(q)
        print(f"Question: {q}")
        print(f"  -> needs_sql={decision.needs_sql}, needs_rag={decision.needs_rag}")
        print(f"  -> reasoning: {decision.reasoning}\n")