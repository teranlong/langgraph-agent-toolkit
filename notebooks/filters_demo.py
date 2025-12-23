# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: agent-service-toolkit
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Chroma Metadata Filters Demo (Cards DB)
#
# Shows how to run metadata filters against the existing cards collection using:
# 1) LangChain `Chroma` retriever
# 2) Chroma HTTP client
#
# It covers multiple filter shapes (equality and range) and highlights that the
# `power` field is stored as a **string** in this collection, so string filters
# are required unless you re-ingest with numeric power.

# %%
import sys
import pathlib

from dotenv import load_dotenv

load_dotenv(".env")


def repo_root(marker: str = "pyproject.toml") -> pathlib.Path:
    """Return repository root by walking parents looking for marker."""

    start = pathlib.Path.cwd().resolve()
    for parent in [start, *start.parents]:
        if (parent / marker).exists():
            return parent
    raise RuntimeError(f"Repo root not found (looked for {marker})")


# Make repo modules importable regardless of working directory
root = repo_root()
sys.path.insert(0, str(root / "src"))

import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from core.settings import settings

conn = settings.chroma_connection()
HOST, PORT = conn["host"], conn["port"]
COLLECTION = "cards-v1__openai__text-embedding-3-large__v1"
QUERY = "Tell me about the bobbit worm card."
K = 5


def build_embeddings():
    """Match the embedding model used by the cards collection (3072 dims)."""

    return OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=(
            settings.CHROMA_OPENAI_API_KEY.get_secret_value()
            if settings.CHROMA_OPENAI_API_KEY
            else settings.OPENAI_API_KEY.get_secret_value()
        ),
    )


def show_docs(label, docs):
    print(f"\n{label} ({len(docs)} results)")
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        print(
            f"{i}. {meta.get('name')} | rarity={meta.get('rarity')} | collection={meta.get('collection')} | power={meta.get('power')}\n"
            f"   {doc.page_content[:160]}..."
        )


# %% [markdown]
# ## Inspect power field type
# Power is currently stored as a string in this collection, so filters should use string values (e.g., `"49"`).

# %%
client = chromadb.HttpClient(host=HOST, port=PORT)
collection = client.get_collection(name=COLLECTION)

sample = collection.get(where={"name": {"$eq": "Bobbit Worm"}}, include=["metadatas"], limit=1)
meta = (sample.get("metadatas") or [None])[0] or {}
print("Bobbit Worm power:", meta.get("power"), "type:", type(meta.get("power")))


# %% [markdown]
# ## LangChain `Chroma` filters (string power, rarity)
# Filters belong in `search_kwargs` for this client version.

# %%
vector_store = Chroma(
    collection_name=COLLECTION,
    embedding_function=build_embeddings(),
    host=HOST,
    port=PORT,
)

filters = {
    "rarity_eq": {"rarity": "Legendary"},
    "power_eq_str": {"power": 49},  # string because metadata stores power as string
    "power_ge_str": {"power": {"$gte": 100}},  # string range comparison
}

retrievers = {
    name: vector_store.as_retriever(search_kwargs={"k": K, "filter": f})
    for name, f in filters.items()
}

for name, r in retrievers.items():
    show_docs(f"LangChain filter={name}", r.invoke(QUERY))


# %% [markdown]
# ## Chroma HTTP client filters (same filters, same collection)

# %%
def render_http(label, results):
    docs = list(zip(results.get("documents", [[]])[0], results.get("metadatas", [[]])[0]))
    print(f"\n{label} ({len(docs)} results)")
    for i, (doc, meta) in enumerate(docs, start=1):
        meta = meta or {}
        print(
            f"{i}. {meta.get('name')} | rarity={meta.get('rarity')} | collection={meta.get('collection')} | power={meta.get('power')}\n"
            f"   {doc[:160]}..."
        )


filters_http = filters  # same definitions as above

for name, f in filters_http.items():
    res = collection.query(
        query_texts=[QUERY],
        n_results=K,
        include=["documents", "metadatas"],
        where=f,
    )
    render_http(f"HTTP filter={name}", res)


# %% [markdown]
# ## Notes
# - If you want numeric comparisons on `power`, re-ingest the collection with `power` stored as a number (int/float).
# - Until then, use string filters for `power` (e.g., `{"power": "49"}` or `{"power": {"$gte": "49"}}`).
