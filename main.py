import os
import streamlit as st
import pickle
import time

from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

print("API Key:", os.getenv("OPENROUTER_API_KEY"))

# Streamlit UI setup
st.title("Alexis ChatBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"
main_placeholder = st.empty()

# Use OpenRouter-compatible LLM
llm = ChatOpenAI(
    temperature=0.9,
    model="gpt-3.5-turbo",  # or any model OpenRouter supports
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY")
)

if process_url_clicked:
    # Load data from URLs
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=1000)
    main_placeholder.text("Text Splitting...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # Use HuggingFace Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Building Embedding Vector Store...✅✅✅")
    time.sleep(2)

    # Save vector store
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

# Ask questions
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)

        st.header("Answer")
        st.write(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                st.write(source)
