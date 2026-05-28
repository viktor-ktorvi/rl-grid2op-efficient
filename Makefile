pre-commit:
	uv run pre-commit run --all-files

mypy:
	uv run mypy .

test:
	uv run pytest

check-everything: pre-commit mypy test
