"""Standalone page to query cards retrieval via the backend API."""

import logging
import os

import httpx
import streamlit as st
from dotenv import load_dotenv

from core.logging_utils import setup_logging

def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)

    load_dotenv()

    st.set_page_config(page_title="Retrieval Tester", page_icon=":mag:")
    st.title("Retrieval Tester")
    st.caption("Queries the backend cards retriever endpoint.")

    k = st.slider("Top k", min_value=1, max_value=10, value=3, step=1)
    query_text = st.text_area("Query", value="Tell me about the bobbit worm card.", height=80)

    backend_url = os.getenv("AGENT_URL", "http://localhost:8080").rstrip("/")
    endpoint = f"{backend_url}/retrieval/cards"

    logger.info("Retrieval tester page loaded", extra={"endpoint": endpoint, "k": k})
    if st.button("Run retrieval", use_container_width=True):
        logger.debug("Running retrieval", extra={"query": query_text, "k": k, "endpoint": endpoint})
        try:
            resp = httpx.post(endpoint, json={"query": query_text, "k": k}, timeout=30)
            resp.raise_for_status()
            results = resp.json()
            logger.debug("Retrieval results", extra={"count": len(results) if results else 0})
            if not results:
                st.info("No results returned.")
            for i, doc in enumerate(results, start=1):
                st.markdown(f"**Result {i}**")
                st.write(doc.get("content", ""))
                metadata = doc.get("metadata", {})
                if metadata:
                    st.json(metadata)
        except Exception as exc:  # pragma: no cover - surfaced to UI
            st.error(f"Retrieval failed: {exc}")


if __name__ == "__main__":
    main()
