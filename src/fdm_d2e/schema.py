from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SCHEMA_DIR = Path(__file__).resolve().parents[2] / 'schemas'


class SchemaError(ValueError):
    pass


def load_schema(name: str) -> dict[str, Any]:
    path = SCHEMA_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _type_ok(value: Any, expected: str | list[str]) -> bool:
    if isinstance(expected, list):
        return any(_type_ok(value, t) for t in expected)
    return {
        'object': isinstance(value, dict),
        'array': isinstance(value, list),
        'string': isinstance(value, str),
        'integer': isinstance(value, int) and not isinstance(value, bool),
        'number': (isinstance(value, int | float) and not isinstance(value, bool)),
        'boolean': isinstance(value, bool),
        'null': value is None,
    }.get(expected, True)


def validate(instance: Any, schema: dict[str, Any], path: str = '$') -> None:
    expected = schema.get('type')
    if expected and not _type_ok(instance, expected):
        raise SchemaError(f"{path}: expected {expected}, got {type(instance).__name__}")
    if 'enum' in schema and instance not in schema['enum']:
        raise SchemaError(f"{path}: expected one of {schema['enum']}, got {instance!r}")
    if isinstance(instance, dict):
        for key in schema.get('required', []):
            if key not in instance:
                raise SchemaError(f"{path}: missing required key {key!r}")
        props = schema.get('properties', {})
        for key, subschema in props.items():
            if key in instance:
                validate(instance[key], subschema, f"{path}.{key}")
    if isinstance(instance, list):
        item_schema = schema.get('items')
        if item_schema:
            for idx, item in enumerate(instance):
                validate(item, item_schema, f"{path}[{idx}]")


def validate_named(instance: Any, schema_filename: str) -> None:
    validate(instance, load_schema(schema_filename))
