"""Repo-wide test-collection setup.

Several agent modules (``vision.perception``, ``orchestrator.orchestrator_llm``) resolve
``OPENAI_API_URL`` at import time via ``agent_core.llm.endpoint_creds``, which requires the URL to
carry an explicit port (see ``agent_core.llm.normalize_endpoint_root``). Developer machines load a
real ``secrets.env`` (gitignored, shape not guaranteed - some predate the port-owning change and
are still a bare host), so importing those modules during collection must not depend on it. Set a
well-formed placeholder before anything else runs; ``load_dotenv`` never overrides an
already-set variable.
"""

import os

os.environ.setdefault("OPENAI_API_URL", "http://127.0.0.1:8000")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
