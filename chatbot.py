# chatbot.py
# pip install pandas chromadb sentence-transformers openai streamlit

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# Optional: import for OpenAI backend
import openai


# =========== CONFIG ==========
DATA_PATH = "Training_Dataset.csv"  # Adjust filename if needed
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3

# If using OpenAI
USE_OPENAI = True
openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

# If using HF local:
USE_HF_LOCAL = False
# ================================

@st.cache_data
def load_corpus(path):
    df = pd.read_csv(path)
    df.fillna("", inplace=True)
    # Create a combined text per row (context chunk)
    corpus = df.apply(lambda r: " | ".join([str(r[col]) for col in df.columns]), axis=1).tolist()
    return corpus

corpus = load_corpus(DATA_PATH)

@st.cache_resource
def build_retriever(corpus):
    client = chromadb.Client()
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL_NAME)
    coll = client.create_collection(name="loan_docs", embedding_function=embed_fn)
    ids = [str(i) for i in range(len(corpus))]
    coll.add(documents=corpus, ids=ids)
    return coll

retriever = build_retriever(corpus)

def retrieve(query, k=TOP_K):
    res = retriever.query(query_texts=[query], n_results=k)
    return res["documents"][0]

def answer_openai(query):
    docs = retrieve(query)
    prompt = f"Use the following context to answer the question.\n\nContext:\n{docs}\n\nQuestion: {query}\nAnswer:"
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message["content"]


def answer_hf(query):
    from transformers import pipeline
    llm = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1", max_new_tokens=200)
    docs = retrieve(query)
    prompt = f"Context:\n{docs}\n\nQ: {query}\nA:"
    return llm(prompt)[0]["generated_text"]
def generate_answer(query):
    if USE_OPENAI:
        try:
            return answer_openai(query)
        except Exception as e:
            st.warning(f"⚠️ OpenAI failed: {e}\nSwitching to Hugging Face...")
            return answer_hf(query)
    else:
        return answer_hf(query)

# === Streamlit UI ===
st.title("📊 Loan Approval Dataset Q&A Bot")

query = st.text_input("Ask about the dataset (e.g., 'How many applicants had credit history=0?')")

if query:
    with st.spinner("Generating answer..."):
        ans = generate_answer(query)
    st.markdown("### Answer")
    st.write(ans)

    with st.expander("Retrieved Context"):
        for i, doc in enumerate(retrieve(query), 1):
            st.markdown(f"**Doc {i}:** {doc}")
