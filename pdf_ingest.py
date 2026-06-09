import os
import re
import uuid
import fitz
import unicodedata

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

model = SentenceTransformer("intfloat/multilingual-e5-base")


def clean_ascii(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9_-]", "_", text)
    return text


def extract_text(filepath: str) -> str:
    pdf = fitz.open(filepath)
    text = ""
    for page in pdf:
        text += page.get_text()
    pdf.close()
    return text


def ingest_pdf(filepath: str):
    filename = clean_ascii(os.path.basename(filepath))

    # extract text
    text = extract_text(filepath)

    # chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )
    chunks = splitter.split_text(text)

    # embedding
    embeddings = model.encode(
        ["passage: " + chunk for chunk in chunks], show_progress_bar=True
    )

    # load pinecone
    load_dotenv()
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("document-qa-rag")

    # delete old vectors of same file
    results = index.query(
        vector=[0.0] * 768,
        top_k=10000,
        filter={"source": filename},
        include_metadata=False,
    )
    ids_to_delete = [m["id"] for m in results["matches"]]
    if ids_to_delete:
        index.delete(ids=ids_to_delete)
        print(f"Deleted {len(ids_to_delete)} old vectors for '{filename}'")

    # prepare vectors
    vectors = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        vectors.append(
            {
                "id": f"{filename}_{i}_{uuid.uuid4().hex[:6]}",
                "values": emb.tolist(),
                "metadata": {
                    "text": chunk,
                    "source": filename,
                    "chunk_id": i,
                },
            }
        )

    # batch upsert
    batch_size = 50
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)
        print(f"Uploaded {i + len(batch)} / {len(vectors)}")

    return len(chunks), filename
