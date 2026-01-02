import os

os.environ.setdefault("OPENAI_API_KEY", "sk-fake-openai-key")
os.environ.setdefault("CARDS_POSTGRES_HOST", "localhost")
os.environ.setdefault("CARDS_POSTGRES_PORT", "56432")
os.environ.setdefault("CARDS_POSTGRES_DB", "cuecards")
os.environ.setdefault("CARDS_POSTGRES_USER", "cuecards")
os.environ.setdefault("CARDS_POSTGRES_PASSWORD", "cuecards")

import pytest

from agents.tools import cards_sql_search_func, cards_sql_search_with_filter_func

pytestmark = pytest.mark.docker


def test_cards_sql_search_returns_results():
    context = cards_sql_search_func("Dragon", k=3)

    assert "Blue Dragon" in context
    assert "Power:" in context
    assert "URL:" in context


def test_cards_sql_search_with_filter_respects_metadata():
    context = cards_sql_search_with_filter_func(
        query="Dragon",
        filters={"power": {"$gte": 80}},
        k=3,
    )

    assert "Blue Dragon" in context
    assert "Power: 91" in context
    assert "Uther Pendragon" not in context


def test_cards_sql_search_real_worm_query():
    context = cards_sql_search_func("worm", k=5)

    assert "Christmas Tree Worm" in context
    assert "Blue Dragon" in context  # collection includes Worms


def test_cards_sql_search_dog_with_power_filter():
    context = cards_sql_search_with_filter_func(
        query="dog",
        filters={"power": {"$gte": 40}},
        k=5,
    )

    assert "Rottweiler" in context
    assert "Doge" not in context  # power 21, should be filtered out


def test_cards_sql_search_dog_with_power_energy_filters_returns_fallback():
    context = cards_sql_search_with_filter_func(
        query="dog",
        filters={"power": {"$gt": 50}, "energy": {"$lt": 5}},
        k=5,
    )

    # No literal "dog" matches meet these constraints; fallback should still return high-power animals.
    assert any(name in context for name in ["Dusky Dolphin", "Cheetah", "Candy Crab"])
