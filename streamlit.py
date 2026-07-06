"""
streamlit_app.py
Simple frontend for FinAgent — calls the FastAPI backend's /ask endpoint
and displays the answer, along with which agents were used and the
generated SQL (if any), so you can show "under the hood" behavior.

IMPORTANT: The FastAPI backend (app.py) must be running first, in a
separate terminal:
    uvicorn app:app --reload

Then run this file with:
    streamlit run streamlit_app.py
"""

import streamlit as st
import requests

API_URL = "http://localhost:8000/ask"

st.set_page_config(page_title="FinAgent", page_icon="💰")

st.title("💰 FinAgent")
st.caption("A multi-agent fintech assistant — routes questions between SQL data lookup and knowledge-based advice.")

question = st.text_input(
    "Ask a finance question:",
    placeholder="e.g. How much did I spend on dining last month?",
)

if st.button("Ask", type="primary") and question:
    with st.spinner("Thinking..."):
        try:
            response = requests.post(API_URL, json={"question": question}, timeout=60)
            response.raise_for_status()
            data = response.json()

            st.markdown("### Answer")
            st.write(data["answer"])

            # Show which agent(s) handled this, as small badges
            badges = []
            if data.get("used_sql"):
                badges.append("🗄️ SQL Agent (via MCP)")
            if data.get("used_rag"):
                badges.append("📚 RAG Agent")
            if badges:
                st.markdown("**Agents used:** " + " · ".join(badges))

            # Show the generated SQL, if any, in a collapsible section
            if data.get("sql_query"):
                with st.expander("See generated SQL"):
                    st.code(data["sql_query"], language="sql")

        except requests.exceptions.ConnectionError:
            st.error(
                "Couldn't reach the FinAgent API. Make sure it's running with:\n\n"
                "`uvicorn app:app --reload`"
            )
        except Exception as e:
            st.error(f"Something went wrong: {e}")

# A few example questions to make the demo easy to show off
st.markdown("---")
st.markdown("**Try asking:**")
st.markdown(
    "- How much did I spend on dining in total?\n"
    "- What's a good rule of thumb for emergency savings?\n"
    "- How much am I spending on subscriptions, and is that a healthy amount?"
)