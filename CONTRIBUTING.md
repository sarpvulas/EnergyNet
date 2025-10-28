Contributing
============

Setup
-----

1. Create a virtual environment and install the package in editable mode:

```
pip install -e .
```

2. Install pre-commit hooks:

```
pip install pre-commit
pre-commit install
```

3. Run tests:

```
pytest -q
```

Conventions
-----------

- Keep visualization code under `src/energynet/viewer/`.
- Avoid writing artifacts into the repo; prefer `artifacts/` or `results/`.
- Run linters and tests locally before submitting PRs.





