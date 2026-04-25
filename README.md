# Function Detective

This repository contains the `function_detective` Clembench game. The active game code lives in the `Code` workspace directory.

## Project Layout

```text
Code/
+-- function_detective/
|   +-- clemgame.json
|   +-- functions.py
|   +-- instancegenerator.py
|   +-- master.py
|   +-- protocol.py
|   +-- utils.py
|   +-- in/
|   |   +-- instances.json
|   +-- resources/
|       +-- initial_prompts/
+-- game_registry.json
+-- key.json
+-- model_registry.json
+-- run_benchmark.sh
```

Historical run outputs may also be present under `Code/results`, `Code/old_results`, or `Code/SHARHUKH_IM_CLEM_RESULTS`.

## Setup

Create and activate a Python virtual environment, then install Clemcore:

```bash
pip install clemcore
```

For API-backed models, add credentials to `Code/key.json`. Custom or local model definitions belong in `Code/model_registry.json`.

## Generate Instances

From the repository root:

```bash
cd Code
python function_detective/instancegenerator.py
```

This regenerates `Code/function_detective/in/instances.json` from the function registry.

## Run The Game

Run Clembench commands from the `Code` directory so Clemcore can discover the game and local registries:

```bash
cd Code
clem list games
clem run -g function_detective -m gpt-4.1-nano
clem transcribe -g function_detective
clem score -g function_detective
clem eval
```

You can replace `gpt-4.1-nano` with any model configured in `model_registry.json` or available through Clemcore.

## Add Or Change Functions

1. Edit `Code/function_detective/functions.py`.
2. Update or add entries in `FUNCTION_REGISTRY`.
3. Regenerate instances with `python function_detective/instancegenerator.py`.
4. Run a small benchmark pass to verify parsing, execution, and scoring.
