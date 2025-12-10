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


def create_chroma_db(
    folder_path: str,
    db_name: str = "./chroma_db",
    delete_chroma_db: bool = True,
    chunk_size: int = 2000,
    overlap: int = 500,
):
    embeddings = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])

    # Initialize Chroma vector store
    if delete_chroma_db and os.path.exists(db_name):
        shutil.rmtree(db_name)
        print(f"Deleted existing database at {db_name}")

    chroma = Chroma(
        embedding_function=embeddings,
        persist_directory=f"./{db_name}",
    )

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

    # Iterate over files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Load document based on file extension
        # Add more loaders if required, i.e. JSONLoader, TxtLoader, etc.
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            continue  # Skip unsupported file types

        # Load and split document into chunks
        document = loader.load()
        chunks = text_splitter.split_documents(document)

        # Add chunks to Chroma vector store
        for chunk in chunks:
            chunk_id = chroma.add_documents([chunk])
            if chunk_id:
                print(f"Chunk added with ID: {chunk_id}")
            else:
                print("Failed to add chunk")

        print(f"Document {filename} added to database.")

    print(f"Vector database created and saved in {db_name}.")
    return chroma


def create_chroma_db_from_cards(
    tsv_path: str,
    db_name: str = "./chroma_db_cards",
    delete_chroma_db: bool = True,
):
    embeddings = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])

    if delete_chroma_db and os.path.exists(db_name):
        shutil.rmtree(db_name)

    chroma = Chroma(embedding_function=embeddings, persist_directory=db_name)

    docs = []
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            content = (
                f"{row['name']} ({row['type']}, {row['rarity']}) — "
                f"{row['album']} / {row['collection']} #{row['number']}\n"
                f"Release: {row['release_date']} | Energy {row['energy']} | "
                f"Power {row['power']} | PPE {row['ppe']}\n"
                f"Ability: {row['ability_name']} — {row['ability_description']}\n"
                f"Tags: {row['tags']}"
            )
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": row["url"],
                        "name": row["name"],
                        "album": row["album"],
                        "collection": row["collection"],
                        "number": row["number"],
                        "type": row["type"],
                        "rarity": row["rarity"],
                        "release_date": row["release_date"],
                        "tags": row["tags"],
                    },
                )
            )

    chroma.add_documents(docs)
    print(f"Loaded {len(docs)} cards into {db_name}")
    return chroma


if __name__ == "__main__":
    # Path to the folder containing the documents
    tsv_path = "./data/cards.tsv"

    # Create the Chroma database
    chroma = create_chroma_db_from_cards(tsv_path=tsv_path, db_name="./chroma_db_cards_run")

    # Create retriever from the Chroma database
    retriever = chroma.as_retriever(search_kwargs={"k": 3})

    # Perform a similarity search
    query = "Tell me about the bobbit worm card."
    similar_docs = retriever.invoke(query)

    # Display results
    for i, doc in enumerate(similar_docs, start=1):
        print(f"\n🔹 Result {i}:\n{doc.page_content}\nTags: {doc.metadata.get('source', [])}")
