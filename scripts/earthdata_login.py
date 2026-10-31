"""Save NASA Earthdata credentials for earthaccess (prompts for username and password).

Run once: uv run python scripts/earthdata_login.py
The login is written to ~/.netrc (~/_netrc on Windows) and reused by `specret download emit`.
"""

import earthaccess

auth = earthaccess.login(strategy="interactive", persist=True)
print("Earthdata login saved." if auth and auth.authenticated else "Login failed.")
