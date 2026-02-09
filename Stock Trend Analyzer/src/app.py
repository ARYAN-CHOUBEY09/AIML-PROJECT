import os
import streamlit as st
import pickle
import numpy as np
import faiss
from dotenv import load_dotenv

from groq import Groq
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================
# LOAD API KEY
# =========================
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# =========================
# STREAMLIT UI
# =========================
st.title("📰 News Research Assistant ")
st.sidebar.title("Enter 3 News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_{i}")
    urls.append(url)

process_clicked = st.sidebar.button("Process URLs")


# File save path
file_path = "news_faiss.pkl"
main_placeholder = st.empty()

# Load SentenceTransformer Model
model = SentenceTransformer("all-MiniLM-L6-v2")


# =========================
# FUNCTION → Groq LLM
# =========================
def ask_groq(question, context):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Answer using ONLY the given context."},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
    )
    return response.choices[0].message.content


# =========================
# PROCESS URLs + BUILD VECTOR DB
# =========================
if process_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("🔄 Loading data...")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80
    )

    main_placeholder.text("✂ Splitting text into chunks...")
    chunks = text_splitter.split_documents(data)

    texts = [c.page_content for c in chunks]

    main_placeholder.text("⚙ Creating embeddings...")
    embeddings = model.encode(texts)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    vector_data = {"index": index, "texts": texts}
    with open(file_path, "wb") as f:
        pickle.dump(vector_data, f)

    main_placeholder.text("✅ FAISS Vector DB Saved Successfully!")


# =========================
# USER QUERY SECTION
# =========================
query = st.text_input("Ask a question about these articles:", key="user_query")

if query:
    if os.path.exists(file_path):
        # Load FAISS database
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        index = data["index"]
        texts = data["texts"]

        # Encode the query
        
        # Convert query into correct shape (1, 384)
        q_emb = model.encode([query])
        q_emb = np.array(q_emb).astype("float32").reshape(1, -1)


        # Perform FAISS search
        distances, ids = index.search(q_emb, k=3)

        # Build final context
        context = "\n\n".join([texts[i] for i in ids[0]])

        # Get answer from Groq LLM
        answer = ask_groq(query, context)

        st.subheader("📌 Answer")
        st.write(answer)

        
