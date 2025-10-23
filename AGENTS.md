# Repository Guidelines

These guidelines help contributors work efficiently with **wp-embeddings**.  Follow them to keep the codebase consistent and easy to maintain.

## Project Structure & Module Organization
```
.
├─ wme_sdk/          # Modified Wikimedia Enterprise SDK. Avoid changes here.
├─ downloaded/       # Raw .tar.gz chunk files per namespace
├─ extracted/        # Extracted NDJSON files (temporary)
├─ command.py        # CLI entry point
├─ database.py       # SQLite helper
├─ progress_utils.py # Helper to show progress bars on long activities
├─ test_*.py         # PyTest unit/integration tests and other manual tests
└─ pyproject.toml    # Project metadata & dependencies
```
Source lives in the top‑level package files; tests start with `test_` and reside alongside the code they verify.

## Build, Test, and Development Commands
| Command | Description |
|---------|-------------|
| `uv sync` | Install/upgrade all dependencies into the virtual environment. |
| `python -m command <subcommand>` | Run the interactive or one‑off CLI (e.g. `status`, `download`). |
| `pytest -q` | Execute the test suite. |
| `uv run black .` | Reformat code with Black (if needed). |

## Coding Style & Naming Conventions
* **Indentation** – 4 spaces, no tabs.
* **Line length** – 88 characters (Black default).
* **File names** – snake_case for modules, `CamelCase` for classes.
* **Functions/variables** – lower_case_with_underscores.
* **Formatting/Linting** – Black for formatting, flake8 for linting (`uv run flake8`).

## Testing Guidelines
* **Framework** – PyTest (declared in `pyproject.toml`).
* **Naming** – Test files start with `test_`; test functions start with `test_`.
* **Running** – `pytest` runs all tests; add `-k <expr>` to filter.
    * **Coverage** – Aim for ≥80 % line coverage on new code (use `pytest --cov`).

**Agent Note:** No testing is required for code inside `wme_sdk`. Tests for that package are intentionally excluded from CI.

## Commit & Pull Request Guidelines
* **Commit messages** – Follow the conventional format:
  ```
  type(scope): short description

  Optional longer description.
  ```
  Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`.
* **PR description** – Brief overview, related issue numbers (`Closes #123`), and any required screenshots or performance notes.
* **CI checks** – All tests and linting must pass before merging.

## Security & Configuration Tips (optional)
* Store API credentials in a `.env` file – never commit it.
* Review third‑party SDK changes in `wme_sdk/` for licensing compliance.

---
Happy hacking! If anything is unclear, open an issue or ask the maintainers.
