import math
import re

import numexpr
from langchain_chroma import Chroma
from langchain_core.tools import BaseTool, tool
from langchain_openai import OpenAIEmbeddings

from core.settings import settings


def calculator_func(expression: str) -> str:
    """Calculates a math expression using numexpr.

    Useful for when you need to answer questions about math using numexpr.
    This tool is only for math questions and nothing else. Only input
    math expressions.

    Args:
        expression (str): A valid numexpr formatted math expression.

    Returns:
        str: The result of the math expression.
    """

    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") raised error: {e}.'
            " Please try again with a valid numerical expression"
        )


calculator: BaseTool = tool(calculator_func)
calculator.name = "Calculator"


# Format retrieved documents
def format_contexts(docs):
    return "\n\n".join(doc.page_content for doc in docs)


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


def database_search_func(query: str) -> str:
    """Searches chroma_db for information in the company's handbook."""
    # Get the chroma retriever
    retriever = load_chroma_db("./chroma_db")

    # Search the database for relevant documents
    documents = retriever.invoke(query)

    # Format the documents into a string
    context_str = format_contexts(documents)

    return context_str


def cards_search_func(query: str) -> str:
    """Searches the Chroma server for information in the cards database."""

    # load chroma connection details from settings to handle docker host automatically
    conn = settings.chroma_connection()

    embeddings = OpenAIEmbeddings()
    vector_store = Chroma(
        collection_name="cards-v1__openai__text-embedding-3-large__v1",
        embedding_function=embeddings,
        host=conn["host"],
        port=conn["port"],
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    documents = retriever.invoke(query)

    # Display results
    for i, doc in enumerate(documents, start=1):
        print(
            f"\n🔹 Result {i}:\n{doc.page_content}\nTags: {doc.metadata.get('source', [])}"
        )

    # Format the documents into a string
    context_str = format_contexts(documents)

    return context_str

database_search: BaseTool = tool(database_search_func)
database_search.name = "Database_Search"  # Update name with the purpose of your database


cards_search: BaseTool = tool(cards_search_func)
cards_search.name = "Cards_Search"  # Update name with the purpose of your database
