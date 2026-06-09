import os
import time
import streamlit as st

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_community.llms import Ollama
from pdf_ingest import ingest_pdf, clean_ascii

# page config
st.set_page_config(page_title="📚 Document Q&A RAG")
st.title("📚 Document Q&A RAG")

# session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []  # lưu filename đã clean_ascii


# load models
@st.cache_resource
def load_models():
    load_dotenv()
    embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    llm = Ollama(model="qwen3:8b")
    return embedding_model, reranker, llm


embedding_model, reranker, llm = load_models()

# pinecone
load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("document-qa-rag")

# upload multifile pdf
uploaded_files = st.file_uploader(
    "Upload PDF",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    new_files = [
        f
        for f in uploaded_files
        if clean_ascii(f.name) not in st.session_state.uploaded_files
    ]
    if new_files:
        for uploaded_file in new_files:
            save_path = uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            with st.spinner(f"Indexing {uploaded_file.name}..."):
                num_chunks, clean_name = ingest_pdf(save_path)
            st.session_state.uploaded_files.append(clean_name)
            st.success(f"✅ {uploaded_file.name} indexed! ({num_chunks} chunks)")
        st.session_state.messages = []
        st.rerun()

if st.session_state.uploaded_files:
    st.info(f"📄 Loaded PDFs: {', '.join(st.session_state.uploaded_files)}")

# chat history
for msg in st.session_state.messages:
    with st.chat_message("user"):
        st.write(msg["question"])
    with st.chat_message("assistant"):
        st.write(msg["answer"])

# question input
if not st.session_state.uploaded_files:
    st.warning("Please upload a PDF first.")
    st.stop()

question = st.chat_input("Ask a question about your document")

# q&a
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.spinner("Searching..."):

        # embedding
        t0 = time.time()
        query_embedding = embedding_model.encode("query: " + question)
        embedding_time = time.time() - t0

        # pinecone search
        t0 = time.time()
        results = index.query(
            vector=query_embedding.tolist(),
            top_k=5,
            include_metadata=True,
            filter={"source": {"$in": st.session_state.uploaded_files}},
        )
        pinecone_time = time.time() - t0
        matches = results["matches"]
        if len(matches) == 0:
            st.error("No documents found.")
            st.stop()

        # re-ranking
        t0 = time.time()
        pairs = []
        for match in matches:
            text = match["metadata"]["text"]
            pairs.append([question, text])
        rerank_scores = reranker.predict(pairs)
        rerank_time = time.time() - t0
        ranked = sorted(
            zip(matches, rerank_scores),
            key=lambda x: x[1],
            reverse=True,
        )
        top_matches = ranked[:3]

        # context
        context = ""
        sources = []
        for match, score in top_matches:
            metadata = match["metadata"]
            text = metadata["text"]
            source = metadata.get("source", "Unknown")
            chunk_id = metadata.get("chunk_id", "Unknown")
            context += text + "\n\n"
            sources.append(
                {
                    "source": source,
                    "chunk_id": chunk_id,
                    "rerank_score": float(score),
                    "text": text,
                }
            )

        # conversation history
        history_text = ""
        for item in st.session_state.messages[-5:]:
            history_text += (
                f"User: {item['question']}\n" f"Assistant: {item['answer']}\n\n"
            )

        # prompt
        prompt = f"""
You are a document question answering assistant.
Use the conversation history when it helps understand follow-up questions.
Answer ONLY from the provided context.
Do NOT use outside knowledge.
If the answer is not explicitly found in the context, say:
"I cannot find the answer in the document."
Answer concisely in 3-5 sentences maximum.
Conversation History:
{history_text}
Context:
{context}
Question:
{question}
Answer:
"""

    # generate with streaming
    t0 = time.time()
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_answer = ""
        for chunk in llm.stream(prompt):
            full_answer += chunk
            response_placeholder.markdown(full_answer + "▌")
        response_placeholder.markdown(full_answer)
        answer = full_answer
    llm_time = time.time() - t0

    st.session_state.messages.append({"question": question, "answer": answer})

    # sources
    with st.expander("Sources"):
        for i, s in enumerate(sources, start=1):
            st.write(f"""
**Rank {i}**
Source: {s['source']}
Chunk ID: {s['chunk_id']}
Rerank Score: {s['rerank_score']:.4f}
""")
            with st.expander(f"Chunk {i}"):
                st.write(s["text"])

    # performance
    with st.expander("Performance"):
        st.write(f"Embedding: {embedding_time:.2f} sec")
        st.write(f"Pinecone: {pinecone_time:.2f} sec")
        st.write(f"Rerank: {rerank_time:.2f} sec")
        st.write(f"LLM: {llm_time:.2f} sec")
        st.write(
            f"Total: {embedding_time + pinecone_time + rerank_time + llm_time:.2f} sec"
        )
