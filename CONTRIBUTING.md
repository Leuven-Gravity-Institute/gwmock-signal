# Contributing to gwmock-signal

🎉 Thank you for your interest in contributing to `gwmock-signal`! Your ideas,
fixes, and improvements are welcome and appreciated.

Whether you’re fixing a typo, reporting a bug, suggesting a feature, or
submitting a pull request—this guide will help you get started.

## How to Contribute

<!-- prettier-ignore-start -->

1. Open an Issue

    - Have a question, bug report, or feature suggestion?
    [Open an issue](https://github.com/Leuven-Gravity-Institute/gwmock-signal/issues/new/choose)
    and describe your idea clearly.
    - Check for existing issues before opening a new one.

2. Fork and Clone the Repository

    ```shell
    git clone git@github.com:<username>/gwmock-signal.git
    cd gwmock-signal
    ```

3. Set Up Your Environment

    We recommend using uv to manage virtual environments for installing `gwmock-signal`.
    If you don't have uv installed, you can install it with pip. See the project pages for more details:

    - Install via pip: `pip install --upgrade pip && pip install uv`
    - Project pages: [uv on PyPI](https://pypi.org/project/uv/) | [uv on GitHub](https://github.com/astral-sh/uv)
    - Full documentation and usage guide: [uv docs](https://docs.astral.sh/uv/)

    ```shell
    # Create a virtual environment (recommended with uv)
    uv venv --python 3.13
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    uv sync --group dev
    ```

4. Set Up Pre-commit Hooks

    We use **pre-commit** to ensure code quality and consistency.
    After installing dependencies, run:

    ```shell
    uv run prek install
    ```

    This ensures checks like code formatting, linting, and basic hygiene run automatically when you commit.

5. Create a New Branch

    Give it a meaningful name like fix-typo-in-docs or feature-add-summary-option.

6. Make Changes

    - Write clear, concise, and well-documented code.
    - Follow [PEP 8](https://pep8.org/) style conventions.
    - Add or update unit tests when applicable.
    - **Keep changes atomic and focused**: one type of change per commit
      (e.g., do not mix refactoring with feature addition).

7. Run Tests

    Ensure that all tests pass before opening a pull request:

    ```shell
    uv run pytest
    ```

8. Open a Pull Request

    Clearly describe the motivation and scope of your change. Link it to the relevant issue if applicable.
    The pull request titles should match the [Conventional Commits spec](https://www.conventionalcommits.org/).

    **Pull request guidelines:**

    - Always use the provided [pull request template](.github/pull_request_template.md)
        and complete all relevant sections.
    - The pull request title must follow the Conventional Commits format, using the appropriate type prefix
        (for example, `feat:`, `fix:`, `docs:`, `refactor:`).
    - Keep each pull request focused on a single type of change (for example, do not mix refactoring with new
        features or documentation-only changes in the same PR).

<!-- prettier-ignore-end -->

## 💡 Tips

- Be kind and constructive in your communication.
- Keep PRs focused and atomic—smaller changes are easier to review.
- Document new features and update existing docs if needed.
- Tag your PR with relevant labels if you can.

## Licensing

By contributing, you agree that your contributions will be licensed under the
same terms as the project: **GPL-3.0-or-later** (see the repository `LICENSE`
file).

---

Thanks again for being part of the `gwmock-signal` community!

---
