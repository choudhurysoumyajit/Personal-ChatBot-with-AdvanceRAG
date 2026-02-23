import os
import json
import tempfile
from langchain_groq import ChatGroq
#from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import shutil

os.environ["CHROMA_TELEMETRY"] = "false"
# Initialize sentence transformer embeddings (free)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#initialize Vector Store Path

VECTOR_STORE_PATH = "vector_store"
HISTORY_FILE = os.path.join(VECTOR_STORE_PATH, "conversation_history.json")

def vector_db():
    return Chroma(embedding_function=embeddings, persist_directory=VECTOR_STORE_PATH)

load_dotenv()

chromadb.api.client.SharedSystemClient.clear_system_cache()

#File Loader & Processing

def process_files(files, chunk_size=1000, chunk_overlap=100):
# Check if the directory exists
   
    if os.path.exists(VECTOR_STORE_PATH) == False:
        os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

    docs = []
    for file in files:
        file_ext = os.path.splitext(file.name)[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        
        if file_ext == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif file_ext == ".csv":
            loader = CSVLoader(tmp_path)
        elif file_ext == ".txt":
            loader = TextLoader(tmp_path)
        else:
            continue
        docs.extend(loader.load())

# Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    vectordb = vector_db()

# ⚠️ This deletes ALL vectors in the default collection

    collection = vectordb._collection
    ids = collection.get()["ids"]
    if ids:
        collection.delete(ids=ids)

    vectordb.add_documents(documents=chunks)


def ask_question(query, k=3, file_flag=False):

    llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7
    )

    if file_flag:

        vectordb = vector_db()
        collection = vectordb._collection  
        data = collection.get()
        ids = data["ids"]

        for i, _id in enumerate(ids, 1):
            print(f"{i}: {_id}")

        prompt_template = """
        As a highly knowledgeable chat assistant, your role is to accurately interpret queries, respond to greetings or generic conversations and 
        provide responses by strictly using the context. Do not hallucinate. Follow these directives to ensure optimal user interactions:
        1. Precision in Answers: If any context is provided, Respond solely with information directly relevant to the user's query from the database. 
        2. Avoiding Duplication: Ensure no response is repeated within the same interaction, maintaining uniqueness and 
            relevance to each user query.
        3. Streamlined Communication: Eliminate any unnecessary comments or closing remarks from responses. Focus on
            delivering clear, concise, and direct answers.
        4. Avoid Non-essential Sign-offs: Do not include any sign-offs like "Best regards" or "FashionBot" in responses.
        5. One-time Use Phrases: Avoid using the same phrases multiple times within the same response. Each 
            sentence should be unique and contribute to the overall message without redundancy.

        Query:
        {context}

        Question: {question}

        Answer:
        """
        custom_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        print(file_flag)
        retriever = vectordb.as_retriever(search_kwargs={"k": k})

        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff", chain_type_kwargs={"prompt": custom_prompt}, return_source_documents=True,)

        result = qa.invoke({"query": query})

        response_text=result["result"]
        sources = result["source_documents"]

        print("\nAnswer:\n", response_text)
        print("\nContext used:")
        for i, doc in enumerate(sources, 1):
            print(f"\n--- Chunk {i} ---")
            print(doc.page_content)
            print(result["result"])

    else:

        prompt_template = """
        As a highly knowledgeable chat assistant, your role is to accurately interpret queries and 
        provide appropriate responses.

        Question: {question}
        """
        custom_prompt = PromptTemplate(template=prompt_template, input_variables=["question"])

        prompt_text = custom_prompt.format(
            question=query
        )
        
        # Invoke the chain with input data and include the callback
        result = llm.invoke(prompt_text)
        print(result.content)
        print(file_flag)
        response_text = result.content

    return response_text

    

