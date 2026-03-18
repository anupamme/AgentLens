#!/usr/bin/env python3
"""Export the SessionTrace Pydantic model as a JSON Schema file."""

import json
from pathlib import Path

from agentlens.schema.trace import SessionTrace


def main() -> None:
    schema = SessionTrace.model_json_schema()
    output_path = Path(__file__).parent.parent / "schemas" / "trace_schema_v1.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(schema, indent=2) + "\n")
    print(f"Schema exported to {output_path}")


if __name__ == "__main__":
    main()
