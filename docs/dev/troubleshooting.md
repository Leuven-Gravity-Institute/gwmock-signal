---
title: Troubleshooting
description: Common problems when developing or documenting gwmock-signal.
---

# Troubleshooting

This page is for **contributors and maintainers** working from a **git clone**
of [gwmock-signal](https://github.com/Leuven-Gravity-Institute/gwmock-signal).
End-user install help is in **[Installation](../user_guide/installation.md)**.

The repo is **uv-first**: use `uv sync` and `uv run` rather than ad hoc
`pip install -e ".[…]"` — there are **no PyPI optional extras**; dev and docs
tools live in **`dependency-groups`** in `pyproject.toml`.

## Environment and `uv`

### “Command not found” or wrong packages after clone

**Symptoms:** `uv run pytest` fails, imports fail, or tools are missing.

1. Work from the **repository root** (directory that contains `pyproject.toml`
   and `uv.lock`).
2. Create or refresh a venv with a **supported Python** (3.12 or 3.13):

    ```bash
    uv venv --python 3.12
    source .venv/bin/activate   # Windows: .venv\Scripts\activate
    ```

3. Install what you need with **groups**, not extras:

    ```bash
    uv sync --group dev              # tests, ruff, pytest, pre-commit, …
    uv sync --group docs             # zensical, mkdocstrings-python
    uv sync --group dev --group docs # both
    ```

4. Prefer **`uv run …`** so commands use the project environment even if you
   forgot to activate the venv:

    ```bash
    uv run pytest
    uv run pre-commit run --all-files
    ```

### Clean slate

```bash
rm -rf .venv
uv venv --python 3.12
source .venv/bin/activate
uv sync --group dev --frozen
```

Use `--frozen` in CI-like debugging to reproduce the lockfile exactly.

### Python version

**Supported:** 3.12 and 3.13 (`requires-python = ">=3.12,<3.14"` in
`pyproject.toml`).

**Symptoms:** resolver errors, missing wheels for `pycbc` / `gwpy`, or CI
mismatch.

- Pin the interpreter: `uv venv --python 3.12` or `uv venv --python 3.13`.
- CI runs the same matrix as `.github/workflows/ci.yml` (Ubuntu and macOS ×
  those two versions).

## Pre-commit

### Hooks not installed or not running

1. `uv sync --group dev` (pre-commit is in the `dev` group).
2. From repo root:

    ```bash
    uv run pre-commit install
    ```

3. Run everything once:

    ```bash
    uv run pre-commit run --all-files
    ```

### Hook failures (ruff, typos, taplo, bandit, prettier, markdownlint, gitleaks)

- **Ruff / ruff-format:** fix or run `uv run ruff check .` /
  `uv run ruff format .` from the root; config is in `pyproject.toml`.
- **typos:** correct spelling or add an exception in `_typos.toml` if justified.
- **taplo:** `pyproject.toml` TOML formatting — run the hook or fix manually.
- **bandit:** security scan; config under `[tool.bandit]` in `pyproject.toml`.
- **prettier / markdownlint-cli2:** Markdown and YAML in `docs/` and elsewhere;
  see `.markdownlint.yaml`.
- **gitleaks:** may flag secrets in test fixtures; only bypass with maintainer
  agreement.
- **uv-lock:** lockfile out of date — run `uv lock` and commit `uv.lock`.

### Skip a hook once (local only)

```bash
SKIP=ruff git commit -m "…"
```

Do not use `SKIP` to bypass failing checks on shared branches without team
agreement.

## Tests (`pytest`)

### Nothing collected or import errors

- Tests live under `tests/`; discovery is configured in
  `[tool.pytest.ini_options]` in `pyproject.toml`
  (`pythonpath = ["src", "scripts"]`).
- Run from **repo root**:

    ```bash
    uv run pytest
    uv run pytest --collect-only -q
    ```

- Default `addopts` include **`-m 'not integration'`** — integration tests need
  `-m integration` explicitly.

### Coverage looks wrong

CI uses the same `addopts` as `pyproject.toml` (`--cov src`, XML report, etc.).
Locally:

```bash
uv run pytest --cov-report=term-missing
```

## Documentation (`zensical`)

### `zensical serve` / `zensical build` fails

1. `uv sync --group docs`.
2. Validate **`zensical.toml`** (TOML syntax and `nav` paths — every entry must
   point to an existing file under `docs/`).
3. API pages use **mkdocstrings** directives like `::: gwmock_signal.module` in
   Markdown under `docs/api/` — there is **no** `gen_ref_pages.py`; if a page is
   empty, check the module path and that the package installs
   (`uv sync --group dev` pulls the library from `src/`).
4. Verbose build:

    ```bash
    uv run zensical build --verbose
    ```

Same command shape as **`.github/workflows/documentation.yml`**
(`uv sync --group docs --frozen` then `uv run --frozen zensical build`).

### Site on GitHub Pages not updating

1. **Actions** → workflow **Documentation** (success on `main`).
2. Repo **Settings → Pages**: source should be **GitHub Actions** (workflow
   uploads the `site/` artifact).
3. Hard-refresh the browser; allow a minute for CDN/cache.

## GitHub Actions (CI)

### PR / push CI red

`.github/workflows/ci.yml` effectively runs:

```bash
uv sync --group dev --frozen
uv run --frozen pytest
```

Reproduce locally with the same Python/OS matrix when possible. **Codecov**
upload may warn if `CODECOV_TOKEN` is unset in forks — that does not fail the
job by itself in all configurations; check the job log.

### CodeQL slow or queued

**CodeQL** is defined in `.github/workflows/codeql.yml` (Python autobuild).
First runs or large dependency trees can take several minutes; that is normal.
Only maintainers should change or remove CodeQL after a security/process review.

### Releases / PyPI

Publishing is handled by dedicated workflows (e.g. `publish.yml`,
`publish_testpypi.yml`, `scheduled_release.yml`). If a release fails, read the
failed job log and confirm tags and secrets. There is no separate **CI/CD** user
guide in `docs/`; contributor setup is in
[CONTRIBUTING.md](https://github.com/Leuven-Gravity-Institute/gwmock-signal/blob/main/CONTRIBUTING.md).

## CLI and imports

### `gwmock-signal: command not found`

Install the package (PyPI or editable from clone). The console script is
declared in `[project.scripts]` in `pyproject.toml` as **`gwmock-signal`**; the
Python import name is **`gwmock_signal`** (underscore).

After `uv pip install gwmock-signal` or `uv sync` from a clone:

```bash
gwmock-signal --help
python -c "import gwmock_signal; print(gwmock_signal.__version__)"
```

### Typer / Rich issues in the CLI

The CLI uses **Typer** and optional **Rich** logging. If stderr formatting
errors appear, ensure you are on a supported Python and that
`uv run gwmock-signal …` uses the intended environment.

## Heavy scientific dependencies (`gwpy`, `pycbc`)

Wheels and install time vary by platform. If resolution or install fails:

- Use **Python 3.12 or 3.13** only.
- Try a **clean venv** and `uv sync --group dev --frozen`.
- On unusual platforms, check upstream **PyCBC** / **GWpy** install notes; this
  project’s CI targets **Ubuntu** and **macOS** only.

## Dependency conflicts

1. `python --version` → must be 3.12 or 3.13.
2. Fresh venv + `uv sync --group dev` (add `--verbose` if needed).
3. Conflicts are usually from **mixing pip and uv** in the same venv — pick one
   tool per environment.

## Getting more help

1. Search
   [existing issues](https://github.com/Leuven-Gravity-Institute/gwmock-signal/issues).
2. Open a **new issue** with OS, exact Python, minimal commands, and full
   traceback or CI log link.
3. Cross-check **[Installation](../user_guide/installation.md)** and the
   **[Command-line interface](../user_guide/cli.md)** for behaviour that is
   intentional rather than broken tooling.
