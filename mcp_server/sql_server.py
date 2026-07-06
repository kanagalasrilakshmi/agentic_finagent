"""
sql_server.py
This is the MCP Server for FinAgent.
 
It exposes ONE tool: run_query(sql).
- Any MCP client (like our SQL Agent) can call this tool.
- The server connects to finance.db, runs the SQL, and returns the result.
- A guardrail blocks anything that isn't a safe, read-only SELECT statement.
"""
 
import os
import sqlite3
from mcp.server.fastmcp import FastMCP
 
# ---- 1. Create the MCP server instance ----
# This name just identifies the server; it will show up when clients connect.
mcp = FastMCP("finagent-sql-server")
 
# Build an ABSOLUTE path based on THIS file's location, so it works no matter
# which folder the server gets launched from (important since sql_agent.py
# launches this as a subprocess from different working directories).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_THIS_DIR, "..", "data", "finance.db")
 
 
# ---- 2. The guardrail function ----
# This checks that the SQL is safe BEFORE we let it run.
def is_safe_select(sql: str) -> bool:
    """
    Returns True only if the query is a read-only SELECT statement.
    Blocks anything that could modify or delete data.
    """
    normalized = sql.strip().lower()
 
    # Must start with "select"
    if not normalized.startswith("select"):
        return False
 
    # Block dangerous keywords anywhere in the query, just in case
    # someone tries something like "select ...; drop table ..."
    dangerous_keywords = ["drop", "delete", "update", "insert", "alter", "truncate", ";"]
    for keyword in dangerous_keywords:
        if keyword in normalized:
            return False
 
    return True
 
 
# ---- 3. The actual tool ----
# The @mcp.tool() decorator is what makes this function callable
# by any MCP client, following the standard MCP protocol.
@mcp.tool()
def run_query(sql: str) -> str:
    """
    Run a read-only SQL SELECT query against the finance database
    and return the results as text.
 
    Args:
        sql: A SQL SELECT statement (must be read-only).
    """
    # Step 1: Guardrail check
    if not is_safe_select(sql):
        return "ERROR: Only single, read-only SELECT statements are allowed."
 
    # Step 2: Try running the query
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        columns = [description[0] for description in cur.description]
        conn.close()
 
        if not rows:
            return "Query ran successfully but returned no rows."
 
        # Format results as simple readable text
        result_lines = [", ".join(columns)]
        for row in rows:
            result_lines.append(", ".join(str(value) for value in row))
 
        return "\n".join(result_lines)
 
    except Exception as e:
        return f"ERROR running query: {str(e)}"
 
 
# ---- 4. Start the server ----
if __name__ == "__main__":
    mcp.run()