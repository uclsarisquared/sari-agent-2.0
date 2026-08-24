# AGENTS.md

Guidance for AI coding agents working in this repository.

## Versioning (SemVer via conventional commits)

This repo uses **python-semantic-release**. It reads the git commit history since the last tag
(`vX.Y.Z`) and derives the next version from the commit message prefixes. A GitHub Actions workflow
(`.github/workflows/release.yml`) runs it automatically on every push to `main`, so:

- `feat:`  -> minor bump (1.0.0 -> 1.1.0)
- `fix:`   -> patch bump (1.0.0 -> 1.0.1)
- `feat!:` or a `BREAKING CHANGE:` footer -> major bump (1.0.0 -> 2.0.0)
- `chore:`, `docs:`, `refactor:`, `test:`, `build:`, `ci:` -> no bump

### Every commit message MUST follow conventional commits

Because the version is derived from commit prefixes, **every commit message must use the
conventional format** or the version silently stops advancing (the tool skips it with no error).

Rules:
- Start with a type: `feat`, `fix`, `chore`, `docs`, `refactor`, `test`, `build`, `ci`, `perf`, `style`.
- An optional scope in parens is allowed: `feat(agent): ...`.
- Use the imperative mood ("Add", "Fix", not "Adds"/"Added"/"Adding").
- One logical change per commit.
- For a breaking change, append `!` to the type (`feat!`) or add a `BREAKING CHANGE:` line in the body.

Examples:
```
feat(agent): add adaptive leg replanning
fix: recover orphaned capture frames and replays
docs: document the versioning workflow
chore(release): 1.0.1 [skip ci]
```

Do NOT create a new tag or bump the version yourself — the workflow does that automatically.
Just write conventional commits.

## Deprecated: moondream pointing path

The moondream-based pointing path (`agent/vision/md_tools.py`, cloud API, `$MDREAM_API_KEY`) is
**deprecated** in favor of the VL model on the configured OpenAI-compatible endpoint
(bbox → center). Do not extend or re-promote it.

## VLM bounding-box support (read before switching the agent's model)

The agent's per-step VLM emits bounding boxes whose **coordinate order is per-model**, and the
parser (`agent/vision/perception.py`, `BBOX_YMIN_FIRST`) only understands **two conventions**:

- **Gemini (Vertex)** — ymin-first `[y1, x1, y2, x2]` (provider `vertex`).
- **Qwen-VL (vLLM)** — xmin-first `[x1, y1, x2, y2]`.

These are trained conventions the models emit regardless of what the prompt asks, so they are not
negotiable. **Only these two are supported.** Before switching the agent to any other VLM model,
confirm how that model reports bounding-box coordinates (the axis order and the value range — this
code assumes normalized 0–1000). If it is neither Gemini's nor Qwen's convention, the parser will
silently misread every box; the model needs to be added to the `BBOX_YMIN_FIRST` branch and verified
on an elongated/off-centre box, not just a near-centred one.
