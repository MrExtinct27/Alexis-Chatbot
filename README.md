# Alexis ChatBot: News Research Tool 📈

Alexis is an AI-powered Streamlit chatbot designed to help users extract insights and answers from **online news articles** by combining the power of **document loading, semantic search, and large language models**.

---

## 🚀 Project Purpose

The goal of Alexis is to make **news comprehension and research** more efficient. Rather than manually reading multiple articles, users can:

- **Paste URLs of news articles**
- **Ask questions** about their contents
- Instantly receive **accurate answers** with **source references**

---

## 🧠 How It Works

Alexis uses LangChain components along with OpenRouter (ChatGPT API) to build a retrieval-based QA system:

### 1. 🔗 Data Loading — `UnstructuredURLLoader`

- The `UnstructuredURLLoader` fetches and parses raw HTML content from the provided URLs.
- It extracts the **main article content** (not ads or sidebars), enabling structured processing of unstructured web data.

### 2. 🧩 Chunk Creation — `RecursiveCharacterTextSplitter`

- The text is broken into **overlapping chunks** using LangChain’s `RecursiveCharacterTextSplitter`.
- This approach:
  - Prioritizes natural boundaries (`\n\n`, `\n`, `.`, `,`)
  - Ensures each chunk is within the model’s context window
- Helps maintain **semantic consistency** and **retrievability**.

### 3. 🔍 Embedding — `HuggingFaceEmbeddings`

- Each chunk is transformed into a vector using the **`sentence-transformers/all-MiniLM-L6-v2`** model.
- This enables **semantic similarity** search later on.
- HuggingFace embeddings are lightweight, fast, and work offline.

### 4. 🧠 Vector Database — `FAISS`

- Vectors are stored using **FAISS (Facebook AI Similarity Search)** for **efficient nearest-neighbor search**.
- When a user asks a question, FAISS retrieves the most relevant chunks to pass to the LLM.

### 5. 💬 Answer Generation — `ChatOpenAI (via OpenRouter)`

- The retrieved content is passed to **ChatGPT (via OpenRouter)** using `RetrievalQAWithSourcesChain`.
- The model:
  - Synthesizes an answer
  - Returns **citations** of which article sections were used

---

## 🌟 Features

- 📰 Accepts up to **3 live article URLs**
- ✂️ Dynamically splits and embeds content into chunks
- 🔍 Semantic search using FAISS + HuggingFace
- 🤖 Answers questions based on article content
- 📌 Displays **source references** for transparency
- 🧠 Powered by OpenRouter’s **GPT-3.5-Turbo**

---

## 🛠 Tech Stack

| Layer        | Tool/Library                         |
|--------------|--------------------------------------|
| Frontend     | Streamlit                            |
| Backend      | LangChain                            |
| LLM          | OpenRouter (`gpt-3.5-turbo`)         |
| Embeddings   | HuggingFace (`all-MiniLM-L6-v2`)     |
| Vector Store | FAISS (langchain_community)          |
| Data Loader  | `UnstructuredURLLoader`              |
| Text Splitter| `RecursiveCharacterTextSplitter`     |
| Environment  | Python `.env` via `python-dotenv`    |

---

