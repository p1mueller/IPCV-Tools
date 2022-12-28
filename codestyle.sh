python3 -m isort . && \
python3 -m flake8 ipcv_tools examples --max-line-length=90 --ignore=E203,D202,D105,W503 --docstring-convention=google && \
python3 -m mypy ipcv_tools examples --ignore-missing-imports --disallow-untyped-defs --follow-imports skip