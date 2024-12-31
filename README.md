# 🧠 Codebase RAG Assistant

**Codebase RAG Assistant** is a Streamlit-based application that lets you quickly index a GitHub repository, store the code as vector embeddings in Pinecone, and then retrieve relevant context to answer questions about the codebase using an LLM (GPT-4o). Perfect for large projects that need fast, intelligent codebase querying.

---

## 🚀 Key Features

- **Clone** – User enters a GitHub repo URL. The app clones the repo locally, reads relevant files, and splits them into chunks.  
- **Embed & Store** – Vector embeddings are created for each chunk and stored in Pinecone under a unique namespace.  
- **Wait for Ingestion** – The app polls Pinecone to ensure the vector count matches the number of chunks added.  
- **Query** – Once ingestion completes, users can ask questions via the chat. The top results from Pinecone are pulled into a GPT-4o prompt, which returns a context-based answer.  

---

## 💻 Tech Stack

- **Streamlit** for the UI and user interactions.
- **GitPython** for cloning GitHub repositories locally.
- **LangChain** modules:
  - `langchain_openai` for GPT-4o embeddings and chat completion,
  - `langchain.text_splitter` for chunking code into smaller pieces,
  - `langchain.docstore.document` to structure code chunks as documents.
- **Pinecone** for vector database storage and retrieval.

---

### 🚀 Give it a try:
[**Codebase RAG Assistant**](https://codebase-rag-app.streamlit.app)

---

### 📸 Screenshots:

<img width="1710" alt="Screenshot 2024-12-31 at 1 17 15 AM" src="https://github.com/user-attachments/assets/4b764eaf-1459-435e-9306-635f344bba77" />
<img width="1710" alt="Screenshot 2024-12-31 at 1 20 40 AM" src="https://github.com/user-attachments/assets/e0fe1bd1-2e24-457d-b976-41643dc4410a" />
