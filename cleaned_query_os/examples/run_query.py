from __future__ import annotations

import os

from query_os import QueryOS


agent = QueryOS(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="gpt-4.1-mini",
)

result = agent.generate(
    question="Which customer has the highest total order amount?",
    db_path="path/to/database.sqlite",
    # Optional. If omitted, QueryOS introspects the SQLite schema directly.
    # schema_metadata_path="path/to/database_description",
)

print(result.final_sql)
print(result.columns)
print(result.rows)

