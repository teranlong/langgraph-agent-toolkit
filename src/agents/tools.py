import math
import re
from typing import Any, Dict, Union

import numexpr
from langchain_chroma import Chroma
from langchain_core.tools import BaseTool, tool
from langchain_openai import OpenAIEmbeddings

from core.settings import settings
from .card_filters import CardFilter


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


def load_default_cards_chroma_db():

    # load chroma connection details from settings to handle docker host automatically
    conn = settings.chroma_connection()

    # This collection was built with text-embedding-3-large (3072 dims); using a
    # smaller embedding model will trigger a dimension mismatch error.
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=(
            settings.CHROMA_OPENAI_API_KEY.get_secret_value()
            if settings.CHROMA_OPENAI_API_KEY
            else None
        ),
    )
    vector_store = Chroma(
        collection_name="cards-v1__openai__text-embedding-3-large__v1",
        embedding_function=embeddings,
        host=conn["host"],
        port=conn["port"],
    )
    return vector_store


def cards_search_func(query: str) -> str:
    """Searches the Chroma server for information in the cards database."""

    vector_store = load_default_cards_chroma_db()

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    documents = retriever.invoke(query)

    # Display results (keep ASCII to avoid encoding errors on Windows shells)
    for i, doc in enumerate(documents, start=1):
        print(
            f"\n* Result {i}:\n{doc.page_content}\nTags: {doc.metadata.get('source', [])}"
        )

    # Format the documents into a string
    context_str = format_contexts(documents)

    return context_str


def cards_search_with_power_filter_func(query: str, filter_value: int) -> str:
    """Searches the Chroma server for information in the cards database."""

    filters = {"power": {"$gte": filter_value}}

    # FILTER = {"rarity": "Legendary"}
    # FILTER = {"power": 49}

    vector_store = load_default_cards_chroma_db()

    retriever = vector_store.as_retriever(search_kwargs={"k": 5, "filter": filters})

    documents = retriever.invoke(query)

    # Display results (keep ASCII to avoid encoding errors on Windows shells)
    for i, doc in enumerate(documents, start=1):
        print(
            f"\n* Result {i}:\n{doc.page_content}\nTags: {doc.metadata.get('source', [])}"
        )

    # Format the documents into a string
    context_str = format_contexts(documents)

    return context_str


def cards_search_with_filter_func(
    query: str, filters: Union[CardFilter, Dict[str, Any]], k: int = 5
) -> str:
    """Search the cards database with validated metadata filters.

    LLM usage:
    - Always pass a `filters` dict matching the CardFilter schema:
      { "rarity": <str or {"$in": [...]}>,
        "collection": <str or {"$in": [...]}>,
        "power": <number or {"$gte"/"$lte"/"$gt"/"$lt"/"$eq"/"$in": number|[numbers]}>,
        "energy": <number or {"$gte"/"$lte"/"$gt"/"$lt"/"$eq"/"$in": number|[numbers]}> }
    - Use numerics for power/energy (no strings/booleans); comparison ops belong inside the nested dict.
    - Set `k` if you want a different top-k (default 5).
    Examples:
    - filters={"rarity": {"$in": ["Legendary", "Epic"]}, "power": {"$gte": 50}}
    - filters={"collection": "Deep Sea", "power": {"$gt": 5, "$lt": 10}, "rarity": "Rare"}
    - filters={"power": {"$gt": 50}, "energy": {"$lt": 5}}
    """

    card_filter = filters if isinstance(filters, CardFilter) else CardFilter(**filters)
    validated_filter = card_filter.to_chroma_filter()

    vector_store = load_default_cards_chroma_db()

    retriever = vector_store.as_retriever(
        search_kwargs={"k": k, "filter": validated_filter}
    )

    documents = retriever.invoke(query)

    for i, doc in enumerate(documents, start=1):
        print(
            f"\n* Result {i}:\n{doc.page_content}\nTags: {doc.metadata.get('source', [])}"
        )

    return format_contexts(documents)


database_search: BaseTool = tool(database_search_func)
database_search.name = "Database_Search"


cards_search: BaseTool = tool(cards_search_func)
cards_search.name = "Cards_Search"


cards_search_with_power_filter: BaseTool = tool(cards_search_with_power_filter_func)
cards_search_with_power_filter.name = "Cards_Search_with_power_filter"


cards_search_with_filter: BaseTool = tool(cards_search_with_filter_func)
cards_search_with_filter.name = "Cards_Search_with_filter"
