import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.llms import Ollama

from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
__import__('pysqlite3')
import sqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

llm = Ollama(model="llama3")
# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.title("Chat with Your Documents Using Chroma and LangChain")

# Initialize Persisted ChromaDB
def initialize_chroma_db(persist_directory='db'):
    if not os.path.exists(persist_directory):
        # Load documents from URLs
        url_loader = WebBaseLoader([
            "https://eshop.se.com/in/blog/post/what-is-a-sensor-how-does-it-work-and-what-are-the-various-types-of-sensors.html",
            "https://builtin.com/articles/iot-sensors"
        ])
        url_docs = url_loader.load()

        # Load documents from PDFs
        pdf_loader = PyPDFDirectoryLoader("./docs")
        pdf_docs = pdf_loader.load()

        # Combine all documents
        documents = pdf_docs + url_docs

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        # Embed and store the texts
        #embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        embedding = OllamaEmbeddings(model='nomic-embed-text')
        vectordb = Chroma.from_documents(documents=texts, 
                                         embedding=embedding, 
                                         persist_directory=persist_directory)
    else:
        # Load the persisted database from disk
        #embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        embedding = OllamaEmbeddings(model='nomic-embed-text')
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

    return vectordb

# Create the Chain
def create_qa_chain(vectordb):
    retriever = vectordb.as_retriever()
    #llm = OpenAI(openai_api_key=OPENAI_API_KEY)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa

# Initialize the database and QA chain
vectordb = initialize_chroma_db()
qa_chain = create_qa_chain(vectordb)
st.write("Chroma database is ready.")

# Streamlit interface
input_prompt = st.text_input("Enter Your Question")

if st.button("Ask Question"):
    response = qa_chain.invoke({"query": input_prompt})
    st.write("Answer:", response['result'])
