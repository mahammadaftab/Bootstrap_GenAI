import os
from typing import List

from dotenv import load_dotenv 

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ ERROR: OPENAI_API_KEY not found. Please check your .env file.")
else:
    print("✅ API Key loaded successfully.")

app = FastAPI(title="Basic RAG API")

DOC_PATH = "knowledge_base.txt"

def initialize_knowledge_base():
    if not os.path.exists(DOC_PATH):
        with open(DOC_PATH, "w") as f:
            f.write("""
            The Apollo 11 mission was the spaceflight that first landed humans on the Moon. 
            Commander Neil Armstrong and lunar module pilot Buzz Aldrin formed the American crew 
            that landed the Apollo Lunar Module Eagle on July 20, 1969.
            """)
    
    loader = TextLoader(DOC_PATH)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(documents)

try:
    chunks = initialize_knowledge_base()
    embeddings = OpenAIEmbeddings() 
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever()
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    
    system_prompt = (
        "Answer the question based strictly on the context below:\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create a simple RAG chain using the pipe operator
    rag_chain = (
        {"context": retriever, "input": lambda x: x["input"]}
        | question_answer_chain
    )

except Exception as e:
    print(f"Error initializing RAG: {e}")
    rag_chain = None
    retriever = None

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    if not rag_chain:
        raise HTTPException(status_code=500, detail="RAG system failed to initialize.")
    
    response = rag_chain.invoke({"input": request.question})
    return QueryResponse(answer=response)

