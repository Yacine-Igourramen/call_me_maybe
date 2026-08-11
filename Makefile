# Define the variable macros
COMPILER = uv run
FILE ?= src

# Ensure Make uses a real hard tab (\t) for recipe lines, not spaces
run:
	@$(COMPILER) python3 -m $(FILE)

install:
	@uv sync
	@uv pip install ./llm_sdk

debug:
	@$(COMPILER) -m pdb $(FILE)

clean:
	@rm -rf src/__pycache__ .mypy_cache llm_sdk/__pycache__

lint:
	@uv run flake8 src/*
	@uv run mypy src/* --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --follow-imports=silent

.PHONY: run install debug clean lint