"""
app.py
FastAPI wrapper around the FinAgent LangGraph pipeline.

Exposes one endpoint:
  POST /ask   -> takes a question, runs the full multi-agent pipeline, returns the answer

Run with:
  uvicorn app:app --reload
"""

from fastapi import FastAPI
from pydantic import BaseModel
from graph import build_graph

app = FastAPI(title="FinAgent API")

# Build the LangGraph pipeline once, at startup, so we don't rebuild it
# on every single request (rebuilding is unnecessary work).
pipeline = build_graph()


# ---- Request/response schemas ----
class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    answer: str
    used_sql: bool
    used_rag: bool
    sql_query: str | None = None


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    Takes a user question, runs it through the full multi-agent pipeline
    (Supervisor -> SQL Agent / RAG Agent -> Combine), and returns the answer.
    """
    result = pipeline.invoke({"question": request.question})

    return AskResponse(
        question=request.question,
        answer=result["final_answer"],
        used_sql=result.get("needs_sql", False),
        used_rag=result.get("needs_rag", False),
        sql_query=result.get("sql_query"),
    )


@app.get("/")
def health_check():
    """Simple endpoint to confirm the API is running."""
    return {"status": "FinAgent API is running"}