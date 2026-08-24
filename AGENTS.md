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
