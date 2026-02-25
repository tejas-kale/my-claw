"""Registry for safe tool registration and execution."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError, create_model

from assistant.db import Database
from assistant.tools.base import Tool


class ToolRegistry:
    """Explicit registry of safe tools."""

    def __init__(self, db: Database) -> None:
        self._db = db
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def list_tool_specs(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters_schema,
                },
            }
            for tool in self._tools.values()
        ]

    async def execute(self, group_id: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        tool = self._tools.get(tool_name)
        if tool is None:
            raise KeyError(f"Unknown tool: {tool_name}")

        validated = _validate_json_schema(tool.parameters_schema, arguments)
        try:
            result = await tool.run(**validated)
            self._db.log_tool_execution(group_id, tool_name, validated, result, succeeded=True)
            return result
        except Exception as exc:  # noqa: BLE001
            self._db.log_tool_execution(group_id, tool_name, validated, {"error": str(exc)}, succeeded=False)
            raise


def _validate_json_schema(schema: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields: dict[str, tuple[type[Any], Any]] = {}
    for name, config in props.items():
        typ = _python_type(config.get("type", "string"))
        default = ... if name in required else None
        fields[name] = (typ, default)

    model = create_model("ToolInputModel", **fields)
    try:
        value = model(**payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid input for tool: {exc}") from exc
    return value.model_dump(exclude_none=True)


def _python_type(schema_type: str) -> type[Any]:
    mapping: dict[str, type[Any]] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    return mapping.get(schema_type, str)
