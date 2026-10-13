"""One provider-independent, GT-independent native output schema.

Numerical limits, valid material identifiers and voxel/touch bounds deliberately
remain in server_v2.core. This schema constrains shape only and allows emptiness.
"""
from __future__ import annotations

from copy import deepcopy

from offline_evaluation.server_v2.core import ValidationError
from offline_evaluation.server_v2.generation import object_hash


def _operation(name: str, coordinates: tuple[str, ...], material: bool) -> dict:
    properties = {"op": {"type": "string", "enum": [name]}}
    properties.update({key: {"type": "integer"} for key in coordinates})
    if material:
        properties["block"] = {"type": "string"}
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


_BOUNDS = ("x1", "y1", "z1", "x2", "y2", "z2")
ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "operations": {
            "type": "array",
            "items": {"anyOf": [
                _operation("set", ("x", "y", "z"), True),
                _operation("fill", _BOUNDS, True),
                _operation("carve", _BOUNDS, False),
            ]},
        },
    },
    "required": ["operations"],
    "additionalProperties": False,
}


def action_schema() -> dict:
    """Return a copy so payload construction cannot mutate shared schema state."""
    return deepcopy(ACTION_SCHEMA)


def schema_sha256() -> str:
    return object_hash(ACTION_SCHEMA)


def _matches(value, schema: dict) -> bool:
    """Validate just the fixed subset used above; perform no coercion or repair."""
    if "anyOf" in schema:
        return any(_matches(value, option) for option in schema["anyOf"])
    kind = schema["type"]
    if kind == "object":
        return (isinstance(value, dict)
                and set(value) == set(schema["required"])
                and all(_matches(value[key], child)
                        for key, child in schema["properties"].items()))
    if kind == "array":
        return isinstance(value, list) and all(_matches(item, schema["items"]) for item in value)
    if kind == "integer":
        return type(value) is int
    if kind == "string":
        return isinstance(value, str) and ("enum" not in schema or value in schema["enum"])
    raise RuntimeError("unsupported schema in fixed local validator")


def validate_document(document) -> list[dict]:
    """Check a previously strict-parsed document; leave local semantics to core."""
    if not _matches(document, ACTION_SCHEMA):
        raise ValidationError("response does not match the common action schema")
    return document["operations"]
