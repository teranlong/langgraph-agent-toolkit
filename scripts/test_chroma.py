import csv
import os
import shutil

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


# Load environment variables from the .env file
load_dotenv()

def load_chroma_db(db_path: str):
    # Create the embedding function for our project description database
    try:
        embeddings = OpenAIEmbeddings()
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize OpenAIEmbeddings. Ensure the OpenAI API key is set."
        ) from e

    # Load the stored vector database
    chroma_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    retriever = chroma_db.as_retriever(search_kwargs={"k": 5})
    return retriever


if __name__ == "__main__":
    # Path to the folder containing the documents
    tsv_path = "./data/cards.tsv"

    # Create the Chroma database
    # chroma = create_chroma_db_from_cards(tsv_path=tsv_path, db_name="./chroma_db_cards")
    retriever = load_chroma_db("./chroma_db_cards")

    # Create retriever from the Chroma database
    # retriever = chroma.as_retriever(search_kwargs={"k": 3})

    # Perform a similarity search
    query = "Tell me about the bobbit worm card."
    similar_docs = retriever.invoke(query)

    # Display results
    for i, doc in enumerate(similar_docs, start=1):
        print(f"\n🔹 Result {i}:\n{doc.page_content}\nTags: {doc.metadata.get('source', [])}")
