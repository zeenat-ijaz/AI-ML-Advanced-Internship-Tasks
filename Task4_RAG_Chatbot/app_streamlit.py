"""
Streamlit UI for the context-aware RAG chatbot.
Run with: streamlit run app_streamlit.py
"""

import streamlit as st

from rag_chatbot import build_chain

st.title("Context-Aware RAG Chatbot")
st.caption("Ask questions about the documents in knowledge_base/. Remembers conversation history.")

if "chain" not in st.session_state:
    with st.spinner("Loading knowledge base and models..."):
        st.session_state.chain = build_chain()

if "history" not in st.session_state:
    st.session_state.history = []

for role, message in st.session_state.history:
    with st.chat_message(role):
        st.write(message)

query = st.chat_input("Ask a question...")
if query:
    st.session_state.history.append(("user", query))
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.chain.invoke({"question": query})
            answer = result["answer"]
            sources = result.get("source_documents", [])
        st.write(answer)
        if sources:
            with st.expander("Sources"):
                for doc in sources:
                    st.write(f"- {doc.metadata.get('source', 'unknown')}: {doc.page_content[:200]}...")

    st.session_state.history.append(("assistant", answer))
