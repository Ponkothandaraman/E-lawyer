import streamlit as st
from deployed import get_qa_chain, create_vector_db

st.set_page_config(layout="wide")

st.title("E-Lawyer")
# btn = st.button("Create Knowledgebase")
# if btn:
#     create_vector_db()

question = st.text_input("Question: ")

if question:
    rag_chain = get_qa_chain()
    response = rag_chain.invoke(question)
    st.header("Answer")
    st.write(response[1103:])