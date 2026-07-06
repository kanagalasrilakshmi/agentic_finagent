# FinAgent — Multi-Agent Fintech Research Assistant

A multi-agent system that answers personal finance questions by intelligently routing between **structured data lookup (SQL)** and **general knowledge retrieval (RAG)** — orchestrated with LangGraph and connected to its database via **MCP (Model Context Protocol)**.

Built as a portfolio project to demonstrate practical, hands-on agentic AI system design: multi-agent orchestration, tool calling, retrieval-augmented generation, and structured, guarded execution — applied to a realistic fintech use case.

> All data used is synthetic/fake, generated locally. No real financial data, company data, or proprietary information is used anywhere in this project.

---

## The problem this solves

A user's finance question can need two very different kinds of answers:

- **"How much did I spend on dining last month?"** → needs an actual number, looked up from real transaction records.
- **"What's a good rule of thumb for emergency savings?"** → needs general financial knowledge/advice, not a database lookup.
- **"How much am I spending on subscriptions, and is that healthy?"** → needs *both* — a real number AND context to judge it.

Most simple chatbots handle only one of these well. FinAgent uses an LLM-based **Supervisor Agent** to read the question and decide, in real time, which capability (or both) is needed — then routes accordingly and combines the results into one natural answer.

---

## Architecture

```
                          User question
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Supervisor Agent   │
                    │  (LLM classifies:    │
                    │  needs_sql? needs_rag?)
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                  │
              ▼ (if needs_sql)                   ▼ (if needs_rag)
     ┌─────────────────┐                ┌──────────────────┐
     │    SQL Agent     │                │    RAG Agent      │
     │  NL → SQL (LLM)  │                │  LangChain +      │
     └────────┬─────────┘                │  Chroma retrieval │
              │                          └─────────┬─────────┘
              ▼                                    │
     ┌─────────────────┐                           │
     │   MCP CLIENT     │                           │
     │  (calls tool)    │                           │
     └────────┬─────────┘                           │
              │  MCP protocol                       │
              ▼                                     │
     ┌─────────────────┐                            │
     │   MCP SERVER     │                            │
     │  run_query tool  │                            │
     │  + SQL guardrail │                            │
     └────────┬─────────┘                            │
              │                                      │
              ▼                                      │
     ┌─────────────────┐                             │
     │  SQLite Database │                             │
     │   (finance.db)   │                             │
     └─────────────────┘                             │
              │                                      │
              └──────────────┬───────────────────────┘
                              ▼
                    ┌──────────────────┐
                    │   Combine Node    │
                    │  (LLM merges into │
                    │  one final answer)│
                    └─────────┬─────────┘
                              ▼
                        Final Answer
```

The whole flow above is wired together using **LangGraph** — a `StateGraph` where each box is a node, and a shared state object carries the question and intermediate results between them. The Supervisor's routing decision determines whether the graph runs the SQL branch, the RAG branch, or **both in parallel**.

---

## How each piece actually works

### 1. Supervisor Agent (`agents/supervisor.py`)
Reads the user's question and asks Claude to classify it: does this need `sql`, `rag`, or both? The LLM's response is parsed as JSON and validated against a **Pydantic schema** (`RouteDecision`), so malformed output fails loudly instead of silently corrupting downstream logic.

### 2. SQL Agent (`agents/sql_agent.py`)
Takes the question, gives Claude the database schema (table names, columns, relationships), and asks it to generate a single, safe, read-only SQL query. It does **not** run this query directly — instead it acts as an **MCP client**:
- Starts the MCP server as a subprocess
- Opens a standard MCP session with it
- Calls the server's `run_query` tool, passing the generated SQL
- Gets the result back over the MCP protocol

### 3. MCP Server (`mcp_server/sql_server.py`)
A small, independent server exposing exactly one tool: `run_query(sql)`. It has no idea what an "agent" or "LLM" is — its only job is to receive SQL text, check it against a **guardrail** (must start with `SELECT`, must not contain `DROP`/`DELETE`/`UPDATE`/`INSERT`/`ALTER`/a second statement), execute it against `finance.db`, and return the result as text.

This client/server separation is the actual point of MCP: the server doesn't know or care who's calling it, and the client doesn't need to know how the server works internally — they only communicate through the shared MCP protocol. This makes the tool swappable/reusable across any future agent, not just this one.

### 4. RAG Agent (`agents/rag_agent.py`)
Answers conceptual questions using Retrieval-Augmented Generation:
- A small set of finance knowledge snippets are chunked (`RecursiveCharacterTextSplitter`) and embedded using a **LangChain-integrated HuggingFace embedding model** (`sentence-transformers/all-MiniLM-L6-v2`, runs locally, no API cost)
- Embeddings are stored in **Chroma**, a local vector database
- When a question comes in, it's embedded the same way, and Chroma finds the most semantically similar chunks (pure vector-distance math, no LLM involved in this step)
- Those chunks + the question are handed to Claude, which writes a grounded natural-language answer

### 5. Combine Node (`graph.py`)
If both SQL and RAG results are present, a final LLM call merges them into one coherent, natural answer — rather than just concatenating two separate outputs.

### 6. LangGraph orchestration (`graph.py`)
Defines the shared `AgentState`, registers each function above as a graph node, and uses **conditional edges** to route to one or both branches based on the Supervisor's decision — enabling true parallel execution when a question needs both capabilities.

### 7. FastAPI layer (`app.py`)
Wraps the entire LangGraph pipeline in a single `POST /ask` endpoint, so it's callable over HTTP instead of only from a Python script. This is what the frontend (and, in principle, any other application) talks to.

### 8. Streamlit frontend (`streamlit_app.py`)
A simple UI — question box, "Ask" button — that calls the FastAPI endpoint and displays the answer, which agent(s) were used, and (for transparency) the generated SQL query in an expandable section.

---

## Tech stack

| Piece | Technology |
|---|---|
| Multi-agent orchestration | LangGraph |
| Tool execution / standardized tool interface | MCP (Model Context Protocol) |
| Vector database | Chroma |
| Embedding model | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (via LangChain) |
| Text chunking | LangChain `RecursiveCharacterTextSplitter` |
| LLM | Claude (Anthropic API) |
| Structured output / validation | Pydantic |
| Data store | SQLite |
| API layer | FastAPI |
| Frontend | Streamlit |
| Synthetic data generation | Faker |

---

## Project structure

```
finagent/
├── data/
│   ├── generate_data.py      # Creates synthetic transactions/merchants/accounts
│   └── finance.db            # Generated SQLite database (not committed)
├── mcp_server/
│   └── sql_server.py         # MCP server exposing run_query tool + guardrail
├── agents/
│   ├── __init__.py
│   ├── supervisor.py         # Routing agent (needs_sql / needs_rag decision)
│   ├── sql_agent.py          # NL → SQL, calls MCP server as a client
│   └── rag_agent.py          # RAG: LangChain + Chroma + HuggingFace embeddings
├── graph.py                  # LangGraph wiring — the full multi-agent pipeline
├── app.py                    # FastAPI wrapper exposing POST /ask
├── streamlit_app.py          # Streamlit frontend
├── requirements.txt
├── .env                      # ANTHROPIC_API_KEY (not committed)
└── README.md
```

---

## Setup & running it

```bash
# Clone and enter the project
git clone https://github.com/<your-username>/finagent.git
cd finagent

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Add your Anthropic API key
echo "ANTHROPIC_API_KEY=your-key-here" > .env

# Generate the synthetic database
cd data && python3 generate_data.py && cd ..

# Terminal 1 — start the backend
uvicorn app:app --reload

# Terminal 2 — start the frontend
streamlit run streamlit_app.py
```

Then open `http://localhost:8501` in your browser.

You can also test the API directly at `http://localhost:8000/docs` (FastAPI's interactive docs).

---

## Example questions to try

- "How much did I spend on dining in total?" → routes to **SQL Agent**
- "What's a good rule of thumb for emergency savings?" → routes to **RAG Agent**
- "How much am I spending on subscriptions, and is that a healthy amount?" → routes to **both**, combined into one answer

---

## Design decisions & tradeoffs

- **Guardrails over trust**: the MCP server independently validates that every SQL query is a single, read-only `SELECT` — it doesn't rely on the LLM "behaving," since prompts can occasionally be bypassed or misinterpreted.
- **Structured output (Pydantic) everywhere an LLM's response feeds into code**: prevents silent failures from malformed LLM output — errors surface immediately and clearly instead of corrupting downstream state.
- **Local embedding model, not an API-based one**: keeps the project runnable with just an Anthropic API key, no extra billing setup, while still using a real, named, LangChain-integrated embedding model rather than an unnamed default.
- **Routing via prompted LLM classification, not a fine-tuned model**: the practical, standard approach for a project at this scale. At production scale, this could be distilled into a smaller, fine-tuned or DPO-trained routing model to reduce cost and latency — noted here as a deliberate scope decision, not an oversight.

## Future enhancements

- Reflection/retry loop: have the Combine node judge whether an answer is sufficient and loop back to retry with a refined query if not (LangGraph supports cycles, not just linear flow — this project currently uses a linear flow for simplicity)
- A second MCP server (e.g., exposing RAG retrieval as an MCP tool too), so both capabilities are accessed through the same standardized protocol
- Native LLM tool-calling (letting the model choose from a tool list and decide arguments itself, rather than the current human-designed routing sequence)
- LangSmith or structured logging for full observability into every agent decision
- Semantic caching to reduce repeated LLM calls for similar questions
