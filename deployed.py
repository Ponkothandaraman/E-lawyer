import streamlit as st
import os
import time
from langchain_community.vectorstores import LanceDB
from langchain.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import LanceDB
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, DirectoryLoader , CSVLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from prettytable import PrettyTable

HF_TOKEN = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN


# Initialize the model
llm_repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
model_kwargs = {"temperature": 0.5, "max_length": 4096, "max_new_tokens": 2048}
model = HuggingFaceHub(repo_id=llm_repo_id, model_kwargs=model_kwargs)

# Specify Embedding Model
embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={'device': 'cpu'})


vectordb_file_path = "faiss_index"



# Create the vector database
def create_vector_db():

   loader = CSVLoader(file_path=r"C:\Users\Ram\Desktop\Cases_processed.csv", csv_args={
    'delimiter': ',',
    'quotechar': '"',
    'fieldnames': ['case title', 'citation', 'bench','info','Judgement']
    })

   pdf = PyPDFLoader(r"C:\Users\Ram\Desktop\IPC_186045.pdf")
   document = loader.load()
   pages = pdf.load()
   docs = document + pages
   chunk_size = 256
   chunk_overlap = 20
   text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
   chunks = text_splitter.split_documents(docs)
   vectordb = FAISS.from_documents(documents=chunks,embedding=embeddings)
   
   # Save vector database locally
   vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, embeddings,allow_dangerous_deserialization=True)

    # Create a retriever for querying the vector database
    search_kwargs = {"k": 10}
    retriever = vectordb.as_retriever(search_kwargs = {"k": 10})

    template = """I want you to act as an IPC lawyer. I will provide you with a list of IPC cases and you will need to come up with a summary of the judgements based on the IPC section. 
    You will also need to provide me with the details of the cases and their conclusions. My first request is "I need a lawyer who is well versed in IPC law. I need someone who can summarize judgements in a concise manner. 
    My second request is that I need a person who is knowledgeable about IPC laws. My third request is, "I want someone who is able to summarize IPC judgements."
    Given the following context and a question, generate an answer based on this context only.
    Understand the meaning of the sentence  and go through similar sentence potraying the same meaning and get the judgement, bench and citation
    In the answer try to provide judgement and bench as much text as possible from given feilds such as bench, citation and judgement in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}
    
    QUESTION: {query}  """

    prompt = ChatPromptTemplate.from_template(template)
    #Build the rag chain
    rag_chain = (
    {"context": retriever, "query": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser())

    return rag_chain
    


if __name__ == "__main__":
    create_vector_db()
    rag_chain = get_qa_chain()