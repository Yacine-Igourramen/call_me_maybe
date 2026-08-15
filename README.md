*This project has been created as part of the 42 curriculum by yigourra.*

# Constrained Function Calling

## Description

This project implements constrained decoding for a local language model. Its goal is to generate valid JSON function calls instead of unrestricted text.

The program reads a list of function definitions and a list of prompts. For each prompt, it selects one valid function and produces a JSON object containing:

- the original user prompt,
- the chosen function name,
- the function parameters as a JSON object.

The output format is:

```json
{
  "prompt": "",
  "name": "",
  "parameters": {}
}
```

This project relies on a local Hugging Face causal language model accessed through a lightweight wrapper named `llm_sdk`.

## Instructions

### Requirements

- Python 3.10 or newer
- `uv`
- internet access for the initial model download

### Installation

From the repository root:

```bash
uv sync
make install
```

The `make install` target performs the project setup and installs the local SDK package:

```bash
uv sync
uv pip install ./llm_sdk
```

### Execution

Run the project with the default input and output files:

```bash
make run
```

Equivalent direct invocation:

```bash
uv run python3 -m src
```

Custom input and output paths can be provided with flags:

```bash
uv run python3 -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Project structure

```text
call_me_maybe/
├── data/
│   ├── input/
│   │   ├── function_calling_tests.json
│   │   └── functions_definition.json
│   └── output/
├── llm_sdk/
│   └── __init__.py
├── moulinette/
│   ├── pyproject.toml
│   └── moulinette/
│       ├── __main__.py
│       ├── extract_functions_infos.py
│       ├── functions_definition.py
│       ├── generate_tests_and_corrections.py
│       └── output_formatter.py
├── src/
│   ├── __main__.py
│   ├── fsm.py
│   └── parsing.py
├── Makefile
├── pyproject.toml
├── README.md
├── uv.lock
└── .gitignore
```

## Algorithm explanation

The project uses constrained decoding instead of letting the model freely generate any text.

At each generation step, the decoder operates in several stages:

1. Prefix mode
   - The model is forced to follow the fixed JSON structure before the function name.
   - This includes the literal JSON keys such as `"prompt"`, `"name"`, and the start of the `"parameters"` object.

2. Function-name selection
   - The allowed function names are stored in a prefix tree.
   - During this phase, the model can only emit tokens that continue at least one valid function name.
   - This prevents invalid or unknown function names from being generated.

3. Static argument layout
   - After the function name is accepted, the decoder enforces JSON punctuation and argument keys.
   - Braces, commas, quotes, and key names are generated deterministically.

4. Dynamic parameter generation
   - The model is allowed to generate values for the current parameter.
   - The parameter type is checked against the finite-state machine to ensure the generated value matches the expected pattern.
   - When the value is complete, the decoder switches back to a static state for the next parameter or closes the object.

5. Token masking
   - Invalid tokens receive a mask that blocks them from being selected.
   - The model then chooses the highest-scoring valid token among the remaining candidates.

This approach reduces malformed output while still letting the model contribute meaningful parameter values.

## Design decisions

### Finite-state machine for valid JSON

The core design choice is to model the output format as a finite-state machine. Static JSON structure and parameter types are encoded into the state graph, which makes generation both constrained and predictable.

### Prefix-tree matching for function names

Multiple function names may share the same prefix. A prefix tree lets the decoder validate every generated token incrementally instead of waiting for the full name to be assembled.

### Type-aware parameter states

Parameter values are not generated as arbitrary text. Instead, each parameter type (`string`, `integer`, `number`, `boolean`) has its own state logic. This makes it easier to enforce valid JSON and type-correct values.

### Final validation with `json.loads()`

Before writing the output file, the generated text is parsed using Python's JSON parser. If decoding fails, the program treats the result as invalid and stops instead of writing malformed JSON.

## Performance analysis

### Accuracy

The constrained decoder significantly reduces invalid JSON and prevents the model from selecting names outside the allowed function list. However, the content of the argument values still depends on the language model and the quality of the prompt.

### Speed

The implementation precomputes valid transitions for the static layout and the parameter types. This avoids re-checking the entire vocabulary from scratch at every step. The main remaining cost is the model inference itself.

### Reliability

The JSON validation step improves reliability by rejecting malformed output before saving results. The main remaining reliability risk is semantic correctness of generated parameter values, which depends on the model’s reasoning capability and the prompt formulation.

## Challenges faced

### Tokenizer mismatch between tokens and characters

Language models generate tokenizer tokens, not always plain characters. Some tokens contain spaces or special representations, so the implementation must normalize token strings before matching them against the finite-state graph.

### Escaping quotes in prompt strings

User prompts can contain quotes and backslashes. These must be escaped correctly so they remain valid JSON strings and do not break the generated output.

### Distinguishing JSON values from Python values

The output must ultimately be written as valid JSON, not as a Python representation string. This required careful handling of `json.loads()` and `json.dump()` so the final file contains real JSON objects, not serialized strings.

### Type-specific numeric handling

Integer and floating-point values require different constraints. The project had to distinguish between `integer` and `number` generation to avoid invalid numeric outputs.

## Testing strategy

The implementation was validated through:

- valid function-definition JSON files,
- invalid function-definition JSON files,
- prompts that map to different functions,
- functions with no parameters,
- functions with one parameter,
- functions with multiple parameters,
- prompts containing single and double quotes,
- malformed generated JSON to confirm failure handling,
- missing input files and invalid output paths,
- CLI flags for custom function-definition, input, and output paths.

The testing focused on two things:

1. checking that the static JSON structure is respected,
2. ensuring that generated function names remain within the allowed set.

## Example usage

Example function definition:

```json
[
  {
    "name": "fn_multiply_numbers",
    "description": "Multiply two numbers together and return their product.",
    "parameters": {
      "a": { "type": "number" },
      "b": { "type": "number" }
    },
    "returns": { "type": "number" }
  }
]
```

Example prompt file:

```json
[
  {
    "prompt": "What is the product of 3 and 5?"
  }
]
```

Example generated result:

```json
[
  {
    "prompt": "What is the product of 3 and 5?",
    "name": "fn_multiply_numbers",
    "parameters": {
      "a": 3,
      "b": 5
    }
  }
]
```

## Resources

- Python JSON documentation: https://docs.python.org/3/library/json.html
- Constrained decoding overview: https://huggingface.co/blog/constrained-beam-search
- finite state machine: https://www.youtube.com/watch?v=e0qB-jFavrM
- a guide: https://www.aidancooper.co.uk/constrained-decoding/


### AI usage

AI was used as a learning and debugging aid during this project.
