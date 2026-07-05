# Define the variable macros
COMPILER = uv run
DEPENDENCY = flake8 mypy
FILE ?= src

# Ensure Make uses a real hard tab (\t) for recipe lines, not spaces
run:
	$(COMPILER) $(FILE)

install:
	uv sync
	uv pip install $(DEPENDENCY)

debug:
	$(COMPILER) -m pdb $(FILE)

clean:
	rm -rf src/__pycache__ src/.mypy_cache

lint:
	uv run flake8 src
	uv run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

.PHONY: run install debug clean lint