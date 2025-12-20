"""Standalone Chroma retrieval tester using the HTTP thin client."""

import asyncio
import json
import logging
from typing import Any

import chromadb
import streamlit as st

from core.logging_utils import setup_logging
from core.settings import settings

DEFAULT_QUERY = "Tell me about the bobbit worm card."

chroma_client = None


@st.cache_resource(show_spinner=False)
def get_client(
    host: str, port: int, ssl: bool, path: str | None
) -> chromadb.HttpClient:
    """Create a single shared Chroma HTTP client instance."""

    client_kwargs: dict[str, Any] = {"host": host, "port": port, "ssl": ssl}
    if path:
        client_kwargs["path"] = path
    return chromadb.HttpClient(**client_kwargs)


async def connect_and_list(
    conn_settings: dict[str, Any],
) -> tuple[chromadb.HttpClient, list]:
    """Connect to Chroma and list collections asynchronously."""

    client = await asyncio.to_thread(get_client, **conn_settings)
    collections = await asyncio.to_thread(client.list_collections)
    return client, collections


def render_results(results: dict[str, Any]) -> None:
    """Render query results in a simple table."""

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    ids = (results.get("ids") or [[]])[0]

    rows = []
    for rank, (doc, meta, dist, doc_id) in enumerate(
        zip(documents, metadatas, distances, ids, strict=False),
        start=1,
    ):
        rows.append(
            {
                "rank": rank,
                "id": doc_id,
                "distance": f"{dist:.4f}" if isinstance(dist, (int, float)) else dist,
                "document": doc,
                "metadata": json.dumps(meta or {}, ensure_ascii=True),
            }
        )

    if not rows:
        st.info("No results returned.")
        return

    st.dataframe(rows, use_container_width=True, hide_index=True)


def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)

    st.set_page_config(page_title="Chroma Retrieval", page_icon=":mag_right:")
    st.title("Chroma Retrieval")
    st.caption("Query an existing Chroma collection via the HTTP thin client.")

    conn_settings = settings.chroma_connection()
    status = st.status("Connecting to Chroma...", expanded=True, state="running")
    try:
        status.write("Initializing HTTP client...")
        client, collections = asyncio.run(connect_and_list(conn_settings))
        status.write(f"Connected to {conn_settings['host']}:{conn_settings['port']}")
        status.update(label="Connected to Chroma", state="complete", expanded=False)
    except Exception as exc:  # pragma: no cover - user-facing error
        status.update(label="Connection failed", state="error", expanded=True)
        st.error(f"Unable to connect to Chroma: {exc}")
        return

    collection_names = [c.name for c in collections]
    if not collection_names:
        st.info("No collections found on the Chroma server.")
        return

    with st.sidebar:
        st.subheader("Connection")
        st.code(f"{conn_settings['host']}:{conn_settings['port']}", language="text")

    selected_collection = st.selectbox(
        "Collection",
        options=collection_names,
        index=0,
    )
    n_results = st.slider(
        "Results to return (k)", min_value=1, max_value=10, value=5, step=1
    )

    with st.form("chroma-query", clear_on_submit=False):
        query_text = st.text_area("Query text", value=DEFAULT_QUERY, height=80)
        submitted = st.form_submit_button("Run query", use_container_width=True)

    if not submitted:
        return

    try:
        collection = client.get_collection(name=selected_collection)
        with st.spinner("Querying collection..."):
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
        logger.info(
            "Chroma query completed",
            extra={"collection": selected_collection, "n_results": n_results},
        )
        render_results(results)
    except Exception as exc:  # pragma: no cover - user-facing error
        st.error(f"Query failed: {exc}")


if __name__ == "__main__":
    main()
