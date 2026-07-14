"""
Task 4: Context-Aware Chatbot Using LangChain / RAG

A conversational chatbot that:
  - Retrieves answers from a vectorized document store (local knowledge base, e.g. Wikipedia
    pages or internal docs placed in knowledge_base/*.txt).
  - Remembers conversational history (context memory) across turns.
  - Uses fully local, free models: sentence-transformers for embeddings + a local
    Hugging Face text-generation model (flan-t5) for answer generation.

Built directly against LangChain 1.x's current APIs (langchain-core / langchain-community),
with retrieval + memory wired manually since the legacy `ConversationalRetrievalChain` /
`ConversationBufferMemory` helpers were removed in LangChain 1.0.

Run the Streamlit UI with: streamlit run app_streamlit.py
This module exposes RagChatbot used by the Streamlit app.
"""

import glob
import os

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

KB_DIR = "knowledge_base"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"
TOP_K = 3
MAX_HISTORY_TURNS = 4  # how many past Q/A turns to keep in the prompt as context memory


def load_documents():
    docs = []
    for path in glob.glob(os.path.join(KB_DIR, "*.txt")):
        docs.extend(TextLoader(path, encoding="utf-8").load())
    return docs


def build_vectorstore():
    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.from_documents(chunks, embeddings)


class RagChatbot:
    """Context-aware RAG chatbot: retrieves relevant chunks + keeps chat history in the prompt."""

    def __init__(self):
        self.vectorstore = build_vectorstore()
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
        self.history = []  # list of (question, answer) tuples

    def _generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model.generate(**inputs, max_new_tokens=256)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _build_prompt(self, question, context_chunks):
        context_text = "\n\n".join(c.page_content for c in context_chunks)

        history_text = ""
        for q, a in self.history[-MAX_HISTORY_TURNS:]:
            history_text += f"Previous question: {q}\nPrevious answer: {a}\n"

        prompt = (
            "You are a helpful assistant answering questions using the context below. "
            "Use the conversation history to resolve follow-up questions (e.g. pronouns "
            "like 'it' or 'that').\n\n"
            f"Context:\n{context_text}\n\n"
            f"{history_text}"
            f"Question: {question}\nAnswer:"
        )
        return prompt

    def ask(self, question: str):
        retrieved = self.vectorstore.similarity_search(question, k=TOP_K)
        prompt = self._build_prompt(question, retrieved)
        result = self._generate(prompt)
        self.history.append((question, result))
        return {"answer": result, "source_documents": retrieved}


def build_chain():
    """Kept for API-compatibility with earlier drafts / the Streamlit app."""
    return RagChatbot()


if __name__ == "__main__":
    bot = build_chain()
    print("RAG chatbot ready. Type 'exit' to quit.\n")
    while True:
        query = input("You: ")
        if query.strip().lower() in {"exit", "quit"}:
            break
        result = bot.ask(query)
        print(f"Bot: {result['answer']}\n")
