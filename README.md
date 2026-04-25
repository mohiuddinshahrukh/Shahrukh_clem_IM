# Shahrukh Clembench Workspace

This repository contains my local Clembench workspace and the custom game `function_detective`.

`function_detective` is a single-player program induction game built with Clembench/Clemcore. A model receives the signature of a hidden Python function, probes it with test inputs, observes the outputs, and then submits Python code that should behave like the hidden function.

## Repository Layout

- `Code/`: active Clembench workspace used for development and benchmark runs
- `Code/function_detective/`: the custom game implementation by Shahrukh Mohiuddin
- `Code/model_registry.json`: local model registry entries
- `Code/key.json`: local API/backend credential file
- `Code/results/`: generated benchmark outputs kept out of version control

## Main Game

The primary custom game in this repository is `function_detective`.

Core files:

- `master.py`: game master, scorer, and benchmark wiring
- `instancegenerator.py`: generation of benchmark instances
- `functions.py`: hidden target functions and registry metadata
- `utils.py`: parsing, test generation, and solution validation helpers
- `protocol.py`: interaction tags used by the game
- `resources/initial_prompts/`: prompt templates
- `in/instances.json`: generated instances used by Clembench

## Running

From `Code/`:

```bash
clem run -g function_detective -m <model_name>
```

To regenerate instances:

```bash
cd function_detective
python instancegenerator.py
```

## Notes

- Benchmark results are intentionally not tracked in the main repository.
- This repository is for Clembench game development and experimentation, not just for a single package snapshot.
