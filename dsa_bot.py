import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# Set up page
st.set_page_config(page_title="PDF QA with TinyLlama", layout="wide")
st.title("💬 Chat with your PDF using TinyLlama")

# Step 1: Set up TinyLlama LLM
@st.cache_resource
def load_llm():
    return HuggingFacePipeline.from_model_id(
        model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        task='text-generation',
        pipeline_kwargs=dict(
            temperature=0.5,
            max_new_tokens=1000
        )
    )

# Step 2: Prompt template
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know — don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# Step 3: Load FAISS DB and QA chain
@st.cache_resource
def load_qa_chain():
    DB_FAISS_PATH = "vectorstore/db_faiss"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    llm = load_llm()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
    )
    return qa_chain

qa_chain = load_qa_chain()

# Step 4: User input and query execution
query = st.text_input("Ask a question based on your PDF:")

if query:
    with st.spinner("Generating answer..."):
        response = qa_chain.invoke({'query': query})
        st.subheader("📄 Answer")
        st.write(response["result"])

        st.subheader("📚 Source Documents")
        for i, doc in enumerate(response["source_documents"]):
            st.markdown(f"**Document {i+1}:**")
            st.markdown(doc.page_content[:1000] + "...")  # Limit content for readability
