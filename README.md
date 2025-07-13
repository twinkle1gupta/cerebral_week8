# 📊 Loan Approval Dataset Q&A Bot

This is a Streamlit-based chatbot that answers natural language queries about the **Loan Approval Prediction dataset** using **Retrieval-Augmented Generation (RAG)**.

---

## 🔍 Features

- ✅ Ask questions like:
  - *"How many applicants had credit history = 0?"*
  - *"What is the loan approval rate for self-employed applicants?"*
- 🔁 Document retrieval via `chromadb`
- 🤖 Answer generation using:
  - OpenAI's GPT (via `gpt-3.5-turbo`)
  - OR HuggingFace models (e.g. `mistralai/Mistral-7B-Instruct-v0.1`)
- 🧠 Embeddings via `sentence-transformers`

---

## 📁 Folder Structure

