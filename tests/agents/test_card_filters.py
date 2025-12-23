import pytest
from pydantic import ValidationError

from agents.card_filters import CardFilter, CardOperatorExpr


def test_power_accepts_numeric_and_casts_numeric_strings():
    f1 = CardFilter(power=50)
    assert f1.power == 50
    assert f1.to_chroma_filter() == {"power": 50}

    f2 = CardFilter(power="50")
    assert f2.power == 50

    f3 = CardFilter(power="50.5")
    assert pytest.approx(f3.power) == 50.5


def test_power_rejects_boolean_and_non_numeric_strings():
    with pytest.raises(ValidationError):
        CardFilter(power=True)
    with pytest.raises(ValidationError):
        CardFilter(power="not-a-number")


def test_power_operator_dict_coerces_values():
    f = CardFilter(power={"$gte": "10", "$lt": 20})
    assert f.power.as_dict == {"$gte": 10, "$lt": 20}
    assert f.to_chroma_filter() == {"power": {"$gte": 10, "$lt": 20}}


def test_power_operator_expr_with_list():
    expr = CardOperatorExpr({"$in": ["1", 2, "3.5"]})
    f = CardFilter(power=expr)
    assert f.power.as_dict["$in"] == [1, 2, 3.5]
    assert f.to_chroma_filter() == {"power": {"$in": [1, 2, 3.5]}}


def test_string_fields_are_cast_to_strings():
    f = CardFilter(rarity=1, collection=False)
    assert f.rarity == "1"
    assert f.collection == "False"
    assert f.to_chroma_filter() == {"rarity": "1", "collection": "False"}


def test_string_field_operator_values_are_stringified():
    expr = CardOperatorExpr({"$in": [1, True, "Legendary"]})
    f = CardFilter(rarity=expr)
    assert f.rarity.as_dict["$in"] == ["1", "True", "Legendary"]
    assert f.to_chroma_filter() == {"rarity": {"$in": ["1", "True", "Legendary"]}}


def test_extra_fields_are_rejected():
    with pytest.raises(ValidationError):
        CardFilter(rarity="Legendary", foo="bar")  # type: ignore[arg-type]


def test_invalid_operator_is_rejected():
    with pytest.raises(ValidationError):
        CardOperatorExpr({"$foo": 1})  # type: ignore[arg-type]


def test_power_mixed_operator_types():
    f = CardFilter(power={"$gte": "10", "$lte": "25.5", "$eq": 15})
    assert f.to_chroma_filter() == {"power": {"$gte": 10, "$lte": 25.5, "$eq": 15}}


def test_power_in_list_with_numeric_strings():
    f = CardFilter(power={"$in": ["1", "2", 3.5]})
    assert f.to_chroma_filter() == {"power": {"$in": [1, 2, 3.5]}}


def test_rarity_in_list_and_collection_single():
    f = CardFilter(rarity={"$in": ["Legendary", 5]}, collection="Deep Sea")
    assert f.to_chroma_filter() == {
        "$and": [
            {"rarity": {"$in": ["Legendary", "5"]}},
            {"collection": "Deep Sea"},
        ]
    }


def test_combined_fields_all_present():
    f = CardFilter(
        rarity={"$in": ["Legendary", "Epic"]},
        collection={"$eq": "Aether"},
        power={"$gte": 75},
    )
    assert f.to_chroma_filter() == {
        "$and": [
            {"rarity": {"$in": ["Legendary", "Epic"]}},
            {"collection": {"$eq": "Aether"}},
            {"power": {"$gte": 75}},
        ]
    }


def test_power_range_gt_and_lt():
    f = CardFilter(power={"$gt": 5, "$lt": "10"})
    assert f.to_chroma_filter() == {"power": {"$gt": 5, "$lt": 10}}


def test_power_range_with_other_fields():
    f = CardFilter(
        power={"$gt": 5, "$lt": 10},
        rarity="Rare",
        collection={"$eq": "Deep Sea"},
    )
    assert f.to_chroma_filter() == {
        "$and": [
            {"power": {"$gt": 5, "$lt": 10}},
            {"rarity": "Rare"},
            {"collection": {"$eq": "Deep Sea"}},
        ]
    }


def test_power_in_list_rejects_non_numeric():
    with pytest.raises(ValidationError):
        CardFilter(power={"$in": [1, "two", 3]})


def test_power_in_list_rejects_boolean():
    with pytest.raises(ValidationError):
        CardFilter(power={"$in": [1, True, 3]})


def test_power_rejects_unknown_operator():
    with pytest.raises(ValidationError):
        CardFilter(power={"gt": 5})  # missing $


def test_combined_logic_edge_case_all_ops():
    f = CardFilter(
        power={"$gt": "1", "$gte": 2, "$lt": "10.5", "$lte": 11, "$eq": 5},
        rarity={"$in": ["Common", "Uncommon"]},
        collection={"$in": ["Deep Sea"]},
    )
    assert f.to_chroma_filter() == {
        "$and": [
            {"power": {"$gt": 1, "$gte": 2, "$lt": 10.5, "$lte": 11, "$eq": 5}},
            {"rarity": {"$in": ["Common", "Uncommon"]}},
            {"collection": {"$in": ["Deep Sea"]}},
        ]
    }


def test_energy_numeric_and_range():
    f = CardFilter(energy={"$gt": "8", "$lt": 50})
    assert f.to_chroma_filter() == {"energy": {"$gt": 8, "$lt": 50}}


def test_energy_with_power_compound():
    f = CardFilter(power={"$gt": 50}, energy={"$lt": 5})
    assert f.to_chroma_filter() == {"$and": [{"power": {"$gt": 50}}, {"energy": {"$lt": 5}}]}


def test_energy_rejects_boolean_and_non_numeric():
    with pytest.raises(ValidationError):
        CardFilter(energy=True)
    with pytest.raises(ValidationError):
        CardFilter(energy="low")


def test_energy_rejects_unknown_operator():
    with pytest.raises(ValidationError):
        CardFilter(energy={"lt": 5})  # missing $


def test_single_field_does_not_wrap_with_and():
    f = CardFilter(power={"$gt": 5})
    assert f.to_chroma_filter() == {"power": {"$gt": 5}}


def test_three_field_and_wrap():
    f = CardFilter(
        power={"$gt": 5},
        energy={"$lt": 10},
        rarity="Epic",
    )
    assert f.to_chroma_filter() == {
        "$and": [
            {"power": {"$gt": 5}},
            {"energy": {"$lt": 10}},
            {"rarity": "Epic"},
        ]
    }
