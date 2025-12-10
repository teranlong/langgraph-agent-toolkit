"""Standalone page to query Chroma stores for quick testing."""

import logging
import os

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings

from core.logging_utils import setup_logging



@st.cache_resource(show_spinner=False)
def get_retriever(db_path: str, k: int) -> VectorStoreRetriever:
    """Return a Chroma retriever for the given persisted DB path."""
    embeddings = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY", ""))
    chroma = Chroma(persist_directory=db_path, embedding_function=embeddings)
    return chroma.as_retriever(search_kwargs={"k": k})


def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)

    load_dotenv()

    st.set_page_config(page_title="Retrieval Tester", page_icon=":mag:")
    st.title("Retrieval Tester")
    st.caption("Quickly query a Chroma store without the chat UI.")

    db_options = ["./chroma_db_cards", "./chroma_db_cards2", "Custom path..."]
    selection = st.selectbox("Chroma DB directory", options=db_options, index=0)
    if selection == "Custom path...":
        db_path = st.text_input("Custom DB path", value="./chroma_db_cards")
    else:
        db_path = selection

    k = st.slider("Top k", min_value=1, max_value=10, value=3, step=1)
    query_text = st.text_area("Query", value="Tell me about the bobbit worm card.", height=80)

    logger.info(
        "Retrieval tester page loaded", extra={"db_path": db_path, "k": k, "query_text": query_text}
    )
    if st.button("Run retrieval", use_container_width=True):
        logger.debug("Running retrieval", extra={"query": query_text, "db_path": db_path, "k": k})
        try:
            retriever = get_retriever(db_path, k)
            results = retriever.invoke(query_text)
            logger.debug("Retrieval results", extra={"count": len(results) if results else 0})
            if not results:
                st.info("No results returned.")
            for i, doc in enumerate(results, start=1):
                st.markdown(f"**Result {i}**")
                st.write(doc.page_content)
                if doc.metadata:
                    st.json(doc.metadata)
        except Exception as exc:  # pragma: no cover - surfaced to UI
            st.error(f"Retrieval failed: {exc}")


if __name__ == "__main__":
    main()
