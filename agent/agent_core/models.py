"""Model ids for the agent runtime, declared in config.env rather than in code.

Every reasoner in this repo talks to an **OpenAI API compatible endpoint** selected by
`$LLM_PROVIDER` (`vllm` or `vertex`). Which model that
endpoint should be asked for is configuration, not a code constant, so it lives in the repo-root
`config.env`:

    OPENAI_MODEL            the model every agent-runtime reasoner uses (orchestrator, actor,
                            associative memory, perception, item resolver, planners, probes)
    OPENAI_ANNOTATOR_MODEL  the offline annotator's model when it runs on the OpenAI-compatible
                            endpoint (`--backend endpoint`); falls back to OPENAI_MODEL when unset.
                            The `claude-cli` annotator backend is unaffected — it carries its own
                            `--model` (sonnet, the frozen quality baseline; see CLAUDE.md).

Both are named for the OpenAI Chat Completions `model` request field they populate, and pair with
the `$OPENAI_API_URL`/`$OPENAI_API_KEY` that address the endpoint (renamed from `SARI_MODEL` /
`SARI_ANNOTATOR_MODEL` 2026-08-18; the old names are no longer read anywhere).

`DEFAULT_MODEL` below is the fallback for a checkout with no `config.env` entry, and is the ONLY
model id hardcoded in the tree. Point the endpoint somewhere else by setting the env var; no code
change is needed.

Deliberately dependency-light (dotenv + stdlib) so leaf scripts and the mapping tools can import
it without pulling in the whole agent runtime.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Repo-root config.env (agent/agent_core/ -> repo root is three parents up), resolved from
# __file__ so it loads regardless of CWD or checkout location. Mirrors agent_core.agent.
load_dotenv(Path(__file__).resolve().parent.parent.parent / "config.env")

# The one hardcoded model id in the tree; OPENAI_MODEL in config.env overrides it.
DEFAULT_MODEL = "Qwen/Qwen3.6-27B"
VERTEX_DEFAULT_MODEL = "google/gemini-3.1-flash-lite"


def agent_model() -> str:
    """The model every agent-runtime reasoner asks the endpoint for ($OPENAI_MODEL)."""
    configured = os.getenv("OPENAI_MODEL")
    if configured:
        return configured
    if (os.getenv("LLM_PROVIDER") or "vllm").strip().lower() == "vertex":
        return VERTEX_DEFAULT_MODEL
    return DEFAULT_MODEL


def annotator_model() -> str:
    """The offline annotator's model on the OpenAI-compatible endpoint ($OPENAI_ANNOTATOR_MODEL,
    defaulting to the agent model)."""
    return os.getenv("OPENAI_ANNOTATOR_MODEL") or agent_model()
