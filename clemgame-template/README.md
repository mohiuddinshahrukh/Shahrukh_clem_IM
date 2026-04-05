# Clemgame Template Workspace

This directory is the active Clembench workspace used in this repository.

It contains the custom game `function_detective`, local model and key configuration files, optional benchmark result folders, and an embedded copy of upstream `clembench` for reference.

## Important Files and Folders

- `function_detective/`: custom single-player game implemented by Shahrukh Mohiuddin
- `model_registry.json`: local model definitions used by `clem`
- `key.json`: API and backend credentials
- `game_registry.json`: optional explicit game registry entries
- `results/`: default output directory for benchmark runs
- `clembench/`: upstream benchmark code copied into the workspace

## Running Games

From this directory, list available games:

```bash
clem list games
```

Run `function_detective` with a model:

```bash
clem run -g function_detective -m <model_name>
```

Score completed runs:

```bash
clem score -g function_detective
```

Generate transcripts:

```bash
clem transcribe -g function_detective
```

## Workspace Notes

- `clem` operates relative to the current working directory, so this folder is intended to be the workspace root.
- Generated outputs such as `results/`, logs, and caches should stay local and out of version control.
- The detailed game-specific documentation lives in `function_detective/README.md`.

## `function_detective` Package Shape

```text
function_detective/
|-- clemgame.json
|-- master.py
|-- instancegenerator.py
|-- functions.py
|-- utils.py
|-- protocol.py
|-- resources/
|   `-- initial_prompts/
`-- in/
    `-- instances.json
```
