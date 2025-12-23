"""Pydantic models for validating card metadata filters."""

from typing import Any, Dict, List, Literal, Union

from pydantic import BaseModel, ConfigDict, RootModel, field_validator

CardPrimitive = Union[str, int, float, bool]
CardOperator = Literal["$eq", "$gt", "$gte", "$lt", "$lte", "$in"]
ALLOWED_OPERATORS = {"$eq", "$gt", "$gte", "$lt", "$lte", "$in"}


class CardOperatorExpr(RootModel[Dict[CardOperator, Union[CardPrimitive, List[CardPrimitive]]]]):
    """A single field's operator expression (e.g., {"$gte": 50})."""

    @property
    def as_dict(self) -> Dict[str, Union[CardPrimitive, List[CardPrimitive]]]:
        return self.root


class CardFilter(BaseModel):
    """Metadata filter for cards (safe to send directly to Chroma)."""

    rarity: Union[str, CardOperatorExpr, None] = None
    collection: Union[str, CardOperatorExpr, None] = None
    power: Union[int, float, CardOperatorExpr, None] = None
    energy: Union[int, float, CardOperatorExpr, None] = None

    model_config = ConfigDict(extra="forbid")

    @staticmethod
    def _cast_numeric(value, field_name: str):
        if isinstance(value, bool):
            raise ValueError(f"{field_name} must be numeric, not boolean")
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            try:
                return int(value) if value.isdigit() else float(value)
            except ValueError as exc:
                raise ValueError(f"{field_name} must be numeric") from exc
        raise ValueError(f"{field_name} must be numeric")

    @staticmethod
    def _cast_str(value):
        if isinstance(value, (int, float, bool)):
            return str(value)
        return str(value)

    @field_validator("rarity", "collection", mode="before")
    @classmethod
    def _ensure_string_fields(cls, value):
        if value is None:
            return value
        if isinstance(value, CardOperatorExpr):
            # Cast nested values to strings
            data = {}
            for op, val in value.as_dict.items():
                if op not in ALLOWED_OPERATORS:
                    raise ValueError(f"unsupported operator for string field: {op}")
                data[op] = [cls._cast_str(v) for v in val] if isinstance(val, list) else cls._cast_str(val)
            return CardOperatorExpr(data)
        if isinstance(value, dict):
            return {
                op: [cls._cast_str(v) for v in val] if isinstance(val, list) else cls._cast_str(val)
                for op, val in value.items()
                if cls._validate_operator(op, numeric=False)
            }
        return cls._cast_str(value)

    @field_validator("power", "energy", mode="before")
    @classmethod
    def _coerce_numeric_fields(cls, value, info):
        """Ensure numeric fields are numeric for correct comparison semantics."""

        if value is None:
            return value

        field_name = info.field_name  # power or energy

        if isinstance(value, CardOperatorExpr):
            data = {}
            for op, val in value.as_dict.items():
                if op not in ALLOWED_OPERATORS:
                    raise ValueError(f"unsupported operator for {field_name}: {op}")
                data[op] = [
                    cls._cast_numeric(v, field_name) for v in val
                ] if isinstance(val, list) else cls._cast_numeric(val, field_name)
            return CardOperatorExpr(data)

        if isinstance(value, dict):
            return {
                op: [cls._cast_numeric(v, field_name) for v in val]
                if isinstance(val, list)
                else cls._cast_numeric(val, field_name)
                for op, val in value.items()
                if cls._validate_operator(op, numeric=True, field=field_name)
            }

        return cls._cast_numeric(value, field_name)

    @staticmethod
    def _validate_operator(op: str, numeric: bool, field: str = "") -> bool:
        if op not in ALLOWED_OPERATORS:
            target = field if field else ("numeric" if numeric else "string")
            raise ValueError(f"unsupported operator for {target}: {op}")
        return True

    def to_chroma_filter(self) -> Dict[str, Any]:
        """Return a plain dict that Chroma's `where`/`filter` parameter expects."""

        def unwrap(val):
            if isinstance(val, CardOperatorExpr):
                return val.as_dict
            return val

        raw = self.model_dump(exclude_none=True)
        unwrapped = {field: unwrap(val) for field, val in raw.items()}
        if len(unwrapped) <= 1:
            return unwrapped
        # Chroma requires a single top-level operator when combining multiple fields.
        return {"$and": [{k: v} for k, v in unwrapped.items()]}
