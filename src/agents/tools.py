import math
import os
import re
from typing import Any, Dict, List, Tuple, Union

import numexpr
import psycopg
from langchain_chroma import Chroma
from langchain_core.tools import BaseTool, tool
from langchain_openai import OpenAIEmbeddings
from psycopg.rows import dict_row

from core.settings import settings, running_in_docker
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


SQL_OPERATOR_MAP = {
    "$gt": ">",
    "$gte": ">=",
    "$lt": "<",
    "$lte": "<=",
    "$eq": "=",
    "$in": "IN",
}


def _cards_postgres_conn_kwargs() -> Dict[str, Any]:
    """Build connection kwargs for the cards Postgres instance with sensible defaults."""

    default_host = "host.docker.internal" if running_in_docker() else "localhost"

    def pick_value(key: str, default: str) -> str:
        card_specific = os.getenv(f"CARDS_{key}")
        if card_specific:
            return card_specific

        env_value = os.getenv(key)
        if env_value:
            return env_value

        setting_value = getattr(settings, key, None)
        if hasattr(setting_value, "get_secret_value"):
            return setting_value.get_secret_value()
        if setting_value:
            return setting_value

        return default

    host = pick_value("POSTGRES_HOST", default_host)
    if running_in_docker() and host in {"localhost", "127.0.0.1"}:
        host = "host.docker.internal"

    return {
        "host": host,
        "port": int(pick_value("POSTGRES_PORT", "56432")),
        "dbname": pick_value("POSTGRES_DB", "cuecards"),
        "user": pick_value("POSTGRES_USER", "cuecards"),
        "password": pick_value("POSTGRES_PASSWORD", "cuecards"),
    }


def _build_filter_clause(
    filters: Dict[str, Any],
) -> Tuple[str, List[Any]]:
    """Translate a CardFilter output dict into a SQL WHERE fragment and parameters."""

    def render_condition(field: str, condition: Any) -> Tuple[str, List[Any]]:
        params: List[Any] = []
        if isinstance(condition, dict):
            fragments = []
            for op, value in condition.items():
                sql_op = SQL_OPERATOR_MAP.get(op)
                if sql_op is None:
                    raise ValueError(f"Unsupported operator for SQL filter: {op}")
                if op == "$in":
                    if not isinstance(value, list):
                        raise ValueError("Value for $in must be a list")
                    placeholders = ", ".join(["%s"] * len(value))
                    fragments.append(f"{field} IN ({placeholders})")
                    params.extend(value)
                else:
                    fragments.append(f"{field} {sql_op} %s")
                    params.append(value)
            return " AND ".join(fragments), params

        # For simple equality matches, use case-insensitive comparison on strings.
        operator = "ILIKE" if isinstance(condition, str) else "="
        return f"{field} {operator} %s", [condition if not isinstance(condition, str) else condition]

    if "$and" in filters:
        clauses: List[str] = []
        params: List[Any] = []
        for clause in filters["$and"]:
            if not isinstance(clause, dict):
                raise ValueError("Invalid $and clause in filter")
            (field, condition), = clause.items()
            fragment, values = render_condition(field, condition)
            clauses.append(f"({fragment})")
            params.extend(values)
        return " AND ".join(clauses), params

    clauses = []
    params: List[Any] = []
    for field, condition in filters.items():
        fragment, values = render_condition(field, condition)
        clauses.append(f"({fragment})")
        params.extend(values)
    return " AND ".join(clauses), params


def _format_card_rows(rows: List[Dict[str, Any]]) -> str:
    """Format card rows as RAG-ready context text."""

    if not rows:
        return "No matching cards were found in the SQL database."

    formatted_rows = []
    for row in rows:
        ability_name = row.get("ability_name") or "None"
        ability_description = row.get("ability_description") or ""
        formatted_rows.append(
            "\n".join(
                [
                    f"{row.get('name', 'Unknown')} [{row.get('rarity', 'Unknown')} - {row.get('collection', 'Unknown')}]",
                    f"Power: {row.get('power')} | Energy: {row.get('energy')} | PPE: {row.get('ppe')}",
                    f"Ability: {ability_name}",
                    ability_description,
                    f"Album: {row.get('album', '')} | Release: {row.get('release_date', '')}",
                    f"URL: {row.get('url', '')}",
                    f"Tags: {row.get('tags', '')}",
                ]
            )
        )
    return "\n\n".join(formatted_rows)


def _sql_cards_query(
    query: str, card_filter: Union[CardFilter, Dict[str, Any], None] = None, k: int = 5
) -> str:
    """Core SQL query helper shared by the SQL RAG tools."""

    validated_filters: Dict[str, Any] | None = None
    if card_filter:
        filter_model = (
            card_filter if isinstance(card_filter, CardFilter) else CardFilter(**card_filter)
        )
        validated_filters = filter_model.to_chroma_filter()

    search_pattern = f"%{query}%"
    where_clauses = [
        "("
        "name ILIKE %s OR "
        "ability_name ILIKE %s OR "
        "ability_description ILIKE %s OR "
        "collection ILIKE %s OR "
        "album ILIKE %s OR "
        "tags ILIKE %s"
        ")"
    ]
    params: List[Any] = [search_pattern] * 6

    if validated_filters:
        filter_clause, filter_params = _build_filter_clause(validated_filters)
        where_clauses.append(filter_clause)
        params.extend(filter_params)

    base_sql = """
    SELECT id,
           name,
           album,
           collection,
           rarity,
           release_date,
           energy,
           power,
           ppe,
           ability_name,
           ability_description,
           url,
           tags
    FROM cards
    WHERE {where}
    ORDER BY power DESC NULLS LAST, rarity DESC, name ASC
    LIMIT %s;
    """

    conn_kwargs = _cards_postgres_conn_kwargs()
    attempts = []
    # First pass: text + filters (if any)
    attempts.append((" AND ".join(where_clauses), list(params)))
    # Fallback pass: drop the text clause if nothing returned but we had filters; this mirrors
    # vector search behavior (Chroma may return semantically related items even if the keyword
    # isn't present literally).
    if validated_filters:
        filter_clause, filter_params = _build_filter_clause(validated_filters)
        attempts.append((filter_clause, list(filter_params)))

    for where_clause, param_values in attempts:
        query_params = list(param_values) + [k]
        try:
            with psycopg.connect(**conn_kwargs, row_factory=dict_row) as conn:
                with conn.cursor() as cur:
                    cur.execute(base_sql.format(where=where_clause), query_params)
                    rows: List[Dict[str, Any]] = cur.fetchall()
        except Exception as exc:  # pragma: no cover - surface connection issues clearly
            raise RuntimeError(
                "Failed to query the cards Postgres database. "
                "Verify that the container is running and connection settings are correct."
            ) from exc

        if rows:
            return _format_card_rows(rows)

    return _format_card_rows([])


def cards_sql_search_func(query: str, k: int = 5) -> str:
    """Search cards in Postgres using text fields (name, abilities, collection, album, tags)."""

    return _sql_cards_query(query=query, k=k)


def cards_sql_search_with_filter_func(
    query: str, filters: Union[CardFilter, Dict[str, Any]], k: int = 5
) -> str:
    """Search cards in Postgres using text fields plus validated metadata filters."""

    return _sql_cards_query(query=query, card_filter=filters, k=k)


cards_sql_search: BaseTool = tool(cards_sql_search_func)
cards_sql_search.name = "Cards_SQL_Search"


cards_sql_search_with_filter: BaseTool = tool(cards_sql_search_with_filter_func)
cards_sql_search_with_filter.name = "Cards_SQL_Search_with_filter"
