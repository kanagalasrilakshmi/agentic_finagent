"""
rag_agent.py
This is the RAG Agent — it answers CONCEPTUAL finance questions
(not ones needing real transaction numbers) using Retrieval-Augmented Generation.
 
This version uses LANGCHAIN for both the vector store and the embedding model,
which is the more industry-standard / resume-relevant way to build RAG.
 
Flow:
  1. A small set of finance knowledge documents are chunked and embedded into
     Chroma using a LangChain embedding model
  2. When a question comes in, LangChain's retriever finds the most relevant chunks
  3. Claude reads those chunks + the question and writes an answer
 
Run this file directly to test it standalone.
"""
 
import os
from dotenv import load_dotenv
from anthropic import Anthropic
 
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
 
load_dotenv()
 
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
 
# ---- 1. Finance knowledge base ----
# Small set of general finance tips/concepts, written by us (not copied from
# any copyrighted source), used purely for demo/portfolio purposes.
FINANCE_DOCS = [
    "A common rule of thumb for emergency savings is to keep 3 to 6 months "
    "of essential living expenses in an easily accessible account, such as "
    "a high-yield savings account.",
 
    "The 50/30/20 budgeting rule suggests allocating 50% of after-tax income "
    "to needs, 30% to wants, and 20% to savings and debt repayment.",
 
    "Compound interest means you earn interest not only on your original "
    "amount saved or invested, but also on the interest that has already "
    "accumulated. Starting early has an outsized effect over long time horizons.",
 
    "Credit utilization ratio is the percentage of your available credit "
    "that you're currently using. Keeping utilization below 30% is generally "
    "considered good for maintaining a healthy credit score.",
 
    "Diversification means spreading investments across different asset "
    "classes, industries, or geographies to reduce the risk of any single "
    "investment significantly hurting your overall portfolio.",
 
    "A budget deficit at the personal level occurs when your monthly "
    "expenses exceed your income. Tracking recurring subscriptions is a "
    "common first step in identifying areas to cut back.",
 
    "Automating savings transfers right after payday, sometimes called "
    "'paying yourself first,' tends to be more effective than saving "
    "whatever is left over at the end of the month.",
 
    "Reviewing recurring subscription spending every few months can reveal "
    "unused or forgotten subscriptions that quietly add up over time.",
 
    "A good general guideline is that housing costs should not exceed "
    "roughly 28-30% of gross monthly income, though this varies by location "
    "and personal circumstances.",
 
    "Building credit history takes time; factors like payment history, "
    "credit utilization, length of credit history, and credit mix all "
    "contribute to a credit score.",
]
 
CHROMA_PATH = "../chroma_db"
COLLECTION_NAME = "finance_knowledge"
 
# ---- 2. LangChain embedding model ----
# HuggingFaceEmbeddings runs a local, free sentence-transformer model
# (no API key/cost needed) — a real, named embedding model via LangChain,
# rather than Chroma's unnamed internal default.
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
 
 
def build_or_load_vectorstore():
    """
    Creates (or loads, if already built) a LangChain Chroma vector store,
    using our LangChain embedding model.
    """
    # LangChain's Chroma wrapper handles persistence automatically at CHROMA_PATH
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=CHROMA_PATH,
    )
 
    # Only ingest if the collection is currently empty
    existing_count = vectorstore._collection.count()
    if existing_count == 0:
        # Split docs into chunks (good practice even for short docs — shows
        # you understand chunking strategy, a real RAG concept)
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        documents = [Document(page_content=doc) for doc in FINANCE_DOCS]
        chunks = splitter.split_documents(documents)
 
        vectorstore.add_documents(chunks)
        print(f"Ingested {len(chunks)} chunks into Chroma via LangChain.")
    else:
        print(f"Collection already has {existing_count} chunks. Skipping ingest.")
 
    return vectorstore
 
 
def retrieve_relevant_docs(vectorstore, question: str, k: int = 3):
    """Uses LangChain's retriever interface to find the most relevant chunks."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    results = retriever.invoke(question)
    return [doc.page_content for doc in results]
 
 
RAG_ANSWER_PROMPT = """You are a helpful personal finance assistant.
Use the following relevant knowledge snippets to answer the user's question.
Only use information from the snippets provided. If the snippets don't fully
answer the question, say so honestly.
 
Relevant knowledge:
{context}
 
User question: {question}
 
Answer:"""
 
 
def generate_answer(question: str, context_docs: list) -> str:
    """Calls Claude to write a natural answer using the retrieved context."""
    context = "\n\n".join(context_docs)
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=400,
        messages=[
            {
                "role": "user",
                "content": RAG_ANSWER_PROMPT.format(context=context, question=question),
            }
        ],
    )
    return response.content[0].text.strip()
 
 
def ask_rag_agent(question: str) -> dict:
    """
    Full RAG Agent pipeline: question -> LangChain retrieval -> generate answer.
    Returns a dict with the answer and which chunks were used (for transparency).
    """
    vectorstore = build_or_load_vectorstore()
    relevant_docs = retrieve_relevant_docs(vectorstore, question)
    answer = generate_answer(question, relevant_docs)
 
    return {
        "question": question,
        "retrieved_docs": relevant_docs,
        "answer": answer,
    }
 
 
# ---- Standalone test ----
if __name__ == "__main__":
    test_question = "What's a good rule of thumb for emergency savings?"
    output = ask_rag_agent(test_question)
    print("Question:", output["question"])
    print("\nRetrieved chunks:")
    for doc in output["retrieved_docs"]:
        print(" -", doc[:80] + "...")
    print("\nAnswer:\n", output["answer"])
 