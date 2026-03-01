"""Price tracking tool: extract grocery receipt items via LLM vision and persist to BigQuery."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from assistant.llm.base import LLMProvider

LOGGER = logging.getLogger(__name__)

_EXTRACTION_SYSTEM_PROMPT = """You extract structured data from German supermarket receipts.
Return ONLY valid JSON with this exact shape:
{
  "supermarket": "string (store name)",
  "date": "YYYY-MM-DD",
  "total_price": 0.00,
  "items": [
    {"name_german": "string", "name_english": "string", "price": 0.00}
  ]
}
All prices must use international decimal format (e.g. 2.5, not 2,5).
English item names must be in title case (e.g. "Whole Milk", not "whole milk").
No markdown, no explanation, only the JSON object."""

_TABLE_SCHEMA = [
    {"name": "supermarket", "type": "STRING"},
    {"name": "date", "type": "DATE"},
    {"name": "item_name_german", "type": "STRING"},
    {"name": "item_name_english", "type": "STRING"},
    {"name": "price", "type": "FLOAT64"},
    {"name": "total_price", "type": "FLOAT64"},
    {"name": "inserted_at", "type": "TIMESTAMP"},
]


class PriceTrackerTool:
    """Extract receipt items via LLM vision and persist to BigQuery."""

    def __init__(
        self,
        llm: LLMProvider,
        bq_project: str,
        bq_dataset: str,
        bq_table: str,
    ) -> None:
        self._llm = llm
        self._bq_project = bq_project
        self._bq_dataset = bq_dataset
        self._bq_table = bq_table

    async def run(self, attachment_path: str, content_type: str) -> dict[str, Any]:
        """Process a receipt image/PDF: extract items, persist to BigQuery, return preview."""
        image_bytes = await asyncio.to_thread(
            self._encode_attachment, attachment_path, content_type
        )
        extraction = await self._call_llm(image_bytes)
        if "error" in extraction:
            return extraction
        rows = self._build_rows(extraction)
        errors = await asyncio.to_thread(self._insert_rows, rows)
        if errors:
            return {"error": f"BigQuery insert failed: {errors}"}
        preview_rows = await asyncio.to_thread(self._query_preview)
        return {
            "status": "ok",
            "message": self._format_preview(extraction, preview_rows),
        }

    def _encode_attachment(self, path: str, content_type: str) -> bytes:
        """Return PNG bytes: converts first PDF page via PyMuPDF, or reads image directly."""
        if content_type == "application/pdf":
            import fitz  # type: ignore[import-untyped]

            doc = fitz.open(path)
            page = doc.load_page(0)
            return page.get_pixmap().tobytes("png")
        with open(path, "rb") as f:
            return f.read()

    async def _call_llm(self, image_bytes: bytes) -> dict[str, Any]:
        """Send a vision message to the LLM and parse the JSON receipt extraction."""
        b64 = base64.b64encode(image_bytes).decode()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {"type": "text", "text": "Extract all items from this receipt."},
                ],
            },
        ]
        response = await self._llm.generate(messages)
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            LOGGER.error("LLM returned non-JSON: %r", response.content[:200])
            return {"error": f"LLM returned invalid JSON: {response.content[:100]}"}

    def _build_rows(self, extraction: dict[str, Any]) -> list[dict[str, Any]]:
        """Build BigQuery row dicts from the LLM extraction result."""
        now = datetime.now(timezone.utc).isoformat()
        rows = []
        for item in extraction.get("items", []):
            rows.append(
                {
                    "supermarket": extraction.get("supermarket", ""),
                    "date": extraction.get("date", ""),
                    "item_name_german": item.get("name_german", ""),
                    "item_name_english": item.get("name_english", ""),
                    "price": float(item.get("price", 0)),
                    "total_price": float(extraction.get("total_price", 0)),
                    "inserted_at": now,
                }
            )
        return rows

    def _insert_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Insert rows into BigQuery, creating the table if it does not yet exist."""
        from google.cloud import bigquery  # type: ignore[import-untyped]

        client = bigquery.Client(project=self._bq_project)
        table_ref = f"{self._bq_project}.{self._bq_dataset}.{self._bq_table}"

        try:
            client.get_table(table_ref)
        except Exception:
            schema = [
                bigquery.SchemaField(col["name"], col["type"]) for col in _TABLE_SCHEMA
            ]
            table = bigquery.Table(table_ref, schema=schema)
            client.create_table(table, exists_ok=True)

        errors = client.insert_rows_json(table_ref, rows)
        return list(errors) if errors else []

    def _query_preview(self) -> list[dict[str, Any]]:
        """Return the 5 most recently inserted rows from BigQuery."""
        from google.cloud import bigquery  # type: ignore[import-untyped]

        client = bigquery.Client(project=self._bq_project)
        table_ref = f"{self._bq_project}.{self._bq_dataset}.{self._bq_table}"
        query = (
            f"SELECT supermarket, date, item_name_german, item_name_english, price "
            f"FROM `{table_ref}` "
            f"ORDER BY inserted_at DESC "
            f"LIMIT 5"
        )
        result = client.query(query).result()
        return [dict(row) for row in result]

    def _format_preview(
        self, extraction: dict[str, Any], rows: list[dict[str, Any]]
    ) -> str:
        """Format a confirmation message + tabular preview for Signal."""
        supermarket = extraction.get("supermarket", "?")
        date = extraction.get("date", "?")
        total = float(extraction.get("total_price", 0))
        n_items = len(extraction.get("items", []))
        lines = [
            f"Saved {n_items} items from {supermarket} ({date}), total: \u20ac{total:.2f}",
            "",
            "Last 5 rows:",
            f"{'German':<25} {'English':<25} {'Price':>7}",
            "-" * 60,
        ]
        for row in rows:
            german = str(row.get("item_name_german", ""))[:24]
            english = str(row.get("item_name_english", ""))[:24]
            price = float(row.get("price", 0))
            lines.append(f"{german:<25} {english:<25} {price:>7.2f}")
        return "\n".join(lines)
