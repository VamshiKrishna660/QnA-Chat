import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time


from dotenv import load_dotenv
load_dotenv()

## load GROQ API
groq_api_key = os.environ['GROQ_API_KEY']
model_name = "BAAI/bge-large-en-v1.5"  
model_kwargs = {'device': 'cpu'}  
encode_kwargs = {'normalize_embeddings': True}  



st.title("ChatGroq Demo")

prompt_link = st.text_input("Enter the Link to Chat with Website!!")


if prompt_link:
    if "vector" not in st.session_state:
        st.session_state.embeddings = HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
        st.session_state.loader =  WebBaseLoader(prompt_link)
        st.session_state.docs = st.session_state.loader.load()

        st.session_state.chunk_documents = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents = st.session_state.chunk_documents.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) 


    llm = ChatGroq(groq_api_key=groq_api_key,
                model_name="meta-llama/llama-4-scout-17b-16e-instruct")
    prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Questions:{input}

    <Note>
    * Talk to the user in a Friendly manner.
    </Note>
    """
    )

    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    prompt = st.text_input("Input your Prompt here")

    if prompt:
        start = time.process_time()
        response = retrieval_chain.invoke({"input":prompt})
        print("Response time : ",time.process_time()-start)
        st.write(response['answer'])

        with st.expander("Document similarity search"):
            for i,doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------")
