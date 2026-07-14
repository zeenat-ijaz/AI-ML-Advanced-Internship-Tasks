# Task 4: Context-Aware Chatbot Using LangChain / RAG

## Objective
Build a conversational chatbot that remembers context across turns and retrieves external
information from a document store during conversations.

## Dataset
A small custom knowledge base (`knowledge_base/*.txt`) — plain-text documents describing this
internship program. Any additional `.txt` files dropped into `knowledge_base/` are automatically
picked up and indexed on the next run (e.g. Wikipedia pages, internal docs, other knowledge
bases).

## Methodology / Approach
1. **Document ingestion**: load all `.txt` files from `knowledge_base/` with LangChain's
   `TextLoader`, then split into ~500-character overlapping chunks with
   `RecursiveCharacterTextSplitter`.
2. **Vector store**: embed chunks with a local `sentence-transformers/all-MiniLM-L6-v2` model
   (no API key required) and index them in a local **FAISS** vector store.
3. **Retrieval-Augmented Generation**: retrieve the top-3 most relevant chunks per query from
   the FAISS store, then feed them plus the running conversation to a local
   `google/flan-t5-base` sequence-to-sequence model (loaded directly via
   `AutoModelForSeq2SeqLM`/`AutoTokenizer`) to generate the answer. (LangChain 1.x removed the
   legacy `ConversationalRetrievalChain`/`HuggingFacePipeline` text2text helpers, so retrieval
   and generation are wired together manually in `RagChatbot` — LangChain is still used for
   document loading, chunking, embeddings, and the FAISS vector store.)
4. **Context memory**: `RagChatbot` keeps the last few (question, answer) turns and includes
   them in the prompt, so follow-up questions resolve correctly against prior turns.
5. **Deployment**: a Streamlit chat UI (`app_streamlit.py`) exposes the chatbot with a running
   chat history and an expandable "Sources" panel showing which document chunks were retrieved.

## How to Run

```bash
pip install -r requirements.txt

# CLI chat loop
python rag_chatbot.py

# Streamlit chat UI
streamlit run app_streamlit.py
```

## Key Results / Insights
- Using fully local, free models (MiniLM embeddings + FLAN-T5 generation) means the chatbot runs
  with no API key and no per-query cost, at the expense of some answer quality compared to a
  larger hosted LLM — a deliberate tradeoff documented here for reproducibility.
- Splitting documents into overlapping chunks (500 chars, 50 overlap) balances retrieval
  precision against losing context at chunk boundaries.
- Returning `source_documents` alongside the answer makes the RAG pipeline auditable — a user can
  verify which part of the knowledge base the answer was grounded in.

## Skills Gained
- Conversational AI development
- Document embedding and vector search (FAISS)
- Retrieval-Augmented Generation (RAG)
- LLM integration and deployment (LangChain + Streamlit)
