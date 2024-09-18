import json
import os
import sys
import boto3

## We will be using Titan embeddings model to generate embedding
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

## Data Ingestion
import numpy as np 
from langchain.text_splitter import RecursiveCharacterSplitter
from langchain.document_loaders import PyPDFDirectoryLoader

# Vector Embedding and Vector Store

from langchain.vectorstores import FAISS
## LLM models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

import streamlit as st

## Bedrock Clients
bedrock = boto3.client(service_name='bedrock-runtime')
bedrock_embeddings = BedrockEmbeddings(model_id='amazon.titan-embed-text-v1', client=bedrock)

## Data Ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader('data')
    documents = loader.load()
    
    text_splitter = RecursiveCharacterSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

## Vector Store and Vector Embeddings

def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs,
                                             bedrock_embeddings
                                             )
    vectorstore_faiss.save_local('faiss_index')

def get_claude_llm():
    ## Create an anthropic model
    llm=Bedrock(model_id='ai21.j2-mid-v1', client=bedrock,
                model_kwargs={'maxTokens':512})
    return llm

def get_llama2_llm():
    ## Create an llama2 model
    llm=Bedrock(model_id='meta.llama2-70b-chat-v1', client=bedrock,
                model_kwargs={'maxTokens':512})
    return llm

prompt_template = """
Human: Use the following pieces of context to provide a concise answer to the question at the end
but atleast provide answer with 250 words.if you don't know answer you can say don't know.
<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, 
    input_variables=['context','question']
)

def get_response_llm(llm, vector_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=vector_faiss.as_retriever(
            search_type='similarity', search_kwargs={'k':3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': PROMPT}
    )
    response = qa({'query':query})
    return response['result']


def main():
    st.title('LangChain Based QA System with PDF Files')
    st.subheader('A system that uses LangChain, Bedrock, and FAISS to answer questions')
    user_question = st.text_input('Ask a question from PDF files')

    with st.sidebar:
        st.title('Update or Create Vector Store')
        if st.button('Vector Update'):
            with st.spinner('Processing...'):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success('Done')

    if st.button('Output'):
        with st.spinner('Processing...'):
            vector_faiss = FAISS.load_local('faiss_index', bedrock_embeddings)
            llm = get_claude_llm()
            response = get_response_llm(llm, vector_faiss, user_question)
            st.write(response)
            
            #llm = get_llama2_llm()
            #response = get_response_llm(llm, vector_faiss, 'What is the history of the Eiffel Tower?')
            #st.write(response)
            #st.text('Note: The responses are based on the provided context and may not be entirely accurate.')


