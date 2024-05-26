import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_objectbox.vectorstores import ObjectBox
from chromadb.config import Settings
from chromadb.client import ChromaClient
from chromadb.utils import Document
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFDirectoryLoader, WebBaseLoader

from dotenv import load_dotenv
load_dotenv()

## load the Groq And OpenAI Api Key
os.environ['OPEN_API_KEY'] = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv('GROQ_API_KEY')

st.title("Chat with your urls and documents With Llama3")

llm = Ollama(model="llama3")

#llm = ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}
    """
)



## Vector Embedding and Objectbox Vectorstore db
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        
        # Load documents from URLs
        url_loader = WebBaseLoader(["https://eshop.se.com/in/blog/post/what-is-a-sensor-how-does-it-work-and-what-are-the-various-types-of-sensors.html",
                                    "https://builtin.com/articles/iot-sensors"])
        url_docs = url_loader.load()
        st.write(f"Loaded {len(url_docs)} documents from URLs.")
        
        # Load documents from PDFs
        pdf_loader = PyPDFDirectoryLoader("./docs")
        pdf_docs = pdf_loader.load()
        st.write(f"Loaded {len(pdf_docs)} documents from PDFs.")

        # Combine both PDF and URL documents
        all_docs = pdf_docs + url_docs
        st.write(f"Total documents to process: {len(all_docs)}")

        
        # Text splitting
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(all_docs[:20])
        
        # Create vector store
        st.session_state.vectors = ObjectBox.from_documents(st.session_state.final_documents, st.session_state.embeddings, embedding_dimensions=768)

input_prompt = st.text_input("Enter Your Question From Documents")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("ObjectBox Database is ready")

import time

if input_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()

    response = retrieval_chain.invoke({'input':input_prompt})

    st.write("Response time: ", time.process_time() - start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
