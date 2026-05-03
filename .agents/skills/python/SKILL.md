---
name: python
description: "Best practices for python code"
---

- Function arguments should be either argument-only or keyword-only (deterministic), by using `/` and `*` in function signature. Do not add too many argument-only arguments, make it argument-only only if it is very obvious. Ideally argument-only arguments should be 1 (best) or 2, avoid making it 0 or more than 3.
- Do not import within function. If the package is not installed in the current environment, install them via `uv add <package>` or tell the user to do so.
- To run python commands, use `uv run python`, `uv run pytest`, etc. Never run `python` directly. You may run `uv run pytest` on your own.
