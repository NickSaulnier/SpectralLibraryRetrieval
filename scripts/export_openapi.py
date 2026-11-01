"""Write the viewer API's OpenAPI schema to a file (for the React app's generated types).

uv run python scripts/export_openapi.py src/specret/viewer/web/src/api/openapi.json
"""

import json
import sys
from pathlib import Path

from specret.viewer.app import create_app

out = Path(sys.argv[1] if len(sys.argv) > 1 else "openapi.json")
out.write_text(json.dumps(create_app().openapi(), indent=1) + "\n", encoding="utf-8")
print(out)
