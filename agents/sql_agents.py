"""
sql_agent.py
This is the SQL Agent — the "brain" that:
  1. Takes a plain English question
  2. Uses Claude (the LLM) to convert it into SQL, using known schema
  3. Acts as an MCP CLIENT — connects to sql_server.py and calls run_query
  4. Returns the final result
 
Run this file directly to test it standalone with a sample question.
"""
 
import asyncio
import os
import sys
from dotenv import load_dotenv
from anthropic import Anthropic
 
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
 
load_dotenv()
 
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
 
# Build an ABSOLUTE path to sql_server.py, based on THIS file's location,
# not on whatever folder the command happens to be run from.
# This is what makes it work whether you run:
#   python3 sql_agent.py        (from inside agents/)
#   python3 graph.py            (from the project root)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SQL_SERVER_PATH = os.path.join(_THIS_DIR, "..", "mcp_server", "sql_server.py")
 
# ---- 1. Schema knowledge, given to the LLM so it can write correct SQL ----
SCHEMA_DESCRIPTION = """
You have access to this SQLite database schema:
 
accounts(account_id, account_holder, account_type, opened_date)
merchants(merchant_id, merchant_name, category)
transactions(transaction_id, account_id, merchant_id, amount, transaction_date, category)
 
Relationships:
- transactions.account_id links to accounts.account_id
- transactions.merchant_id links to merchants.merchant_id
 
Notes:
- transaction_date is stored as TEXT in 'YYYY-MM-DD' format.
- amount is a REAL number (dollars).
- category exists on both merchants and transactions (same values, e.g. 'dining', 'groceries').
"""
 
SQL_GENERATION_PROMPT = """You are a SQL expert. Convert the user's question into a single, 
valid, read-only SQLite SELECT query.
 
{schema}
 
Rules:
- Only output the SQL query itself, nothing else. No explanation, no markdown formatting, no backticks.
- Only use SELECT statements. Never use INSERT, UPDATE, DELETE, DROP, or ALTER.
- Only write ONE statement (no semicolons separating multiple statements).
 
User question: {question}
 
SQL query:"""
 
 
def generate_sql(question: str) -> str:
    """Calls Claude to convert the natural language question into SQL."""
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": SQL_GENERATION_PROMPT.format(
                    schema=SCHEMA_DESCRIPTION, question=question
                ),
            }
        ],
    )
    sql = response.content[0].text.strip()
    # Safety: strip accidental markdown code fences if the model adds them
    sql = sql.replace("```sql", "").replace("```", "").strip()
    return sql
 
 
async def run_sql_via_mcp(sql: str) -> str:
    """
    Acts as an MCP CLIENT.
    Starts sql_server.py as a subprocess and calls its run_query tool
    with the generated SQL, using the standard MCP protocol.
    """
    server_params = StdioServerParameters(
        command=sys.executable,  # uses the same Python interpreter running this script
        args=[SQL_SERVER_PATH],  # absolute path, works from any working directory
    )
 
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("run_query", {"sql": sql})
            # result.content is a list of content blocks; get the text out
            return result.content[0].text
 
 
async def ask_sql_agent(question: str) -> dict:
    """
    Full SQL Agent pipeline: question -> SQL -> MCP call -> result.
    Returns a dict with both the generated SQL and the final answer,
    so the frontend can show "see generated SQL" later.
    """
    sql = generate_sql(question)
    result = await run_sql_via_mcp(sql)
    return {
        "question": question,
        "sql_query": sql,
        "result": result,
    }
 
 
# ---- Standalone test ----
if __name__ == "__main__":
    test_question = "How much did I spend on dining in total?"
    output = asyncio.run(ask_sql_agent(test_question))
    print("Question:", output["question"])
    print("\nGenerated SQL:\n", output["sql_query"])
    print("\nResult:\n", output["result"])
 