# %% [markdown]
# # Cards Retrieval Demo (LangChain + Chroma HTTP)
#
# Minimal retrieval examples against the existing cards collection using:
# - LangChain `Chroma` retriever
# - Chroma HTTP client
#
# Includes unfiltered queries and simple metadata filters (rarity and power).
# Note: `power` is stored as a **string** in this collection, so string filters are used.

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


# %% [markdown]
# ## LangChain `Chroma` retriever
# Filters live in `search_kwargs` for this client version.

# %%
vector_store = Chroma(
    collection_name=COLLECTION,
    embedding_function=build_embeddings(),
    host=HOST,
    port=PORT,
)

filters = {
    "rarity_legendary": {"rarity": "Legendary"},
    "power_eq_str": {"power": "49"},  # power stored as string
}

retrievers = {
    name: vector_store.as_retriever(search_kwargs={"k": K, "filter": f})
    for name, f in filters.items()
}

for name, r in retrievers.items():
    print(f"\n=== LangChain filter: {name} ===")
    for i, doc in enumerate(r.invoke(QUERY), start=1):
        meta = doc.metadata or {}
        print(
            f"{i}. {meta.get('name')} | rarity={meta.get('rarity')} | collection={meta.get('collection')} | power={meta.get('power')}"
        )


# %% [markdown]
# ## Chroma HTTP client
# Same filters, applied via `where`.

# %%
client = chromadb.HttpClient(host=HOST, port=PORT)
collection = client.get_collection(name=COLLECTION)

for name, f in filters.items():
    res = collection.query(
        query_texts=[QUERY],
        n_results=K,
        include=["documents", "metadatas"],
        where=f,
    )
    print(f"\n=== HTTP filter: {name} ===")
    for i, meta in enumerate(res.get("metadatas", [[]])[0], start=1):
        meta = meta or {}
        print(
            f"{i}. {meta.get('name')} | rarity={meta.get('rarity')} | collection={meta.get('collection')} | power={meta.get('power')}"
        )


# %% [markdown]
# ## Notes
# - `power` is currently stored as string; use string filters until re-ingested as numeric.
# - Adjust `filters` to test other metadata fields (e.g., `collection`, `rarity`, ranges on string values).
