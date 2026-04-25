# Function Detective

`function_detective` is a dialogue-based Clembench game where a guesser probes a hidden Python function with test inputs, observes outputs, and then submits Python code for the inferred function.

## Game Files

```text
function_detective/
+-- clemgame.json
+-- functions.py
+-- instancegenerator.py
+-- master.py
+-- protocol.py
+-- utils.py
+-- in/
|   +-- instances.json
+-- resources/
    +-- initial_prompts/
```

## Generate Instances

From the parent `Code` directory:

```bash
python function_detective/instancegenerator.py
```

## Run

From the parent `Code` directory:

```bash
clem run -g function_detective -m gpt-4.1-nano
clem transcribe -g function_detective
clem score -g function_detective
clem eval
```

## Development Notes

Hidden functions and metadata are registered in `functions.py`. The game master logic is implemented in `master.py`, and initial player prompts are stored under `resources/initial_prompts`.
