# Functionigma: The Black Box Function Game

**Functionigma** is a dialogue-based puzzle game benchmark for the [Clembench](https://github.com/clembench/clembench) framework. In this game, an AI Model (the **Guesser**) acts as a scientist trying to infer the logic of a hidden Python function by probing it with inputs and observing the outputs.

## 🎮 Game Overview

The game simulates a "Black Box" testing scenario:

1. **The Challenge:** The AI is given the *signature* of a hidden function (e.g., `(x: int, y: int) -> int`) and a hint about its category (e.g., "Math", "String").
2. **The Process:** The AI generates inputs (e.g., `Input: 5, 3`).
3. **The Feedback:** The Game Master executes the hidden function and returns the real output (e.g., `Function Output: 2`).
4. **The Goal:** The AI must deduce the exact logic and submit a Python code block that is functionally equivalent to the hidden function.

## 📂 Project Structure

```text
functionigma/
├── functions.py           # Registry of hidden functions (The "Ground Truths")
├── instance_generator.py  # Script to generate instances.json from the registry
├── master.py              # The Game Master (Main logic & execution engine)
├── utils.py               # Utilities for dynamic validation, parsing, and data generation
├── requirements.txt       # Dependencies
└── resources/
    └── initial_prompts/   # System prompts/templates for the AI players

```

## 🚀 How to Run

### 1. Prerequisites

Ensure you have the `clembench` framework installed and your environment configured.

### 2. Generate Game Instances

Before running the benchmark, you must generate the `instances.json` file. This script reads your `functions.py`, sorts puzzles by difficulty, and prepares the game metadata.

```bash
python functionigma/instance_generator.py

```

### 3. Run the Benchmark

Use the standard clembench CLI to run the game. Replace `<model_name>` with your target model (e.g., `gpt-4`, `claude-3`).

```bash
clem run Functionigma -m <model_name>

```

### 4. Score the Results

After the run completes, calculate the metrics (Success Rate, etc.).

```bash
clem score Functionigma

```

## 🛠️ Adding New Puzzles

Functionigma is designed to be easily extensible. You can add new hidden functions without touching the core game engine.

1. Open `functionigma/functions.py`.
2. Import the helper enums: `Category`, `Difficulty`, and the `@register` decorator.
3. Write your function and decorate it:

```python
@register(category=Category.STRING, difficulty=Difficulty.EASY)
def f_reverse_upper(x: str) -> str:
    """Reverses the string and converts it to uppercase."""
    return x[::-1].upper()

```

4. Re-run `python functionigma/instance_generator.py`.
5. Your new puzzle is now live and ready to be played!

## 🧠 Logic Validation System

Unlike simple text-based benchmarks, Functionigma does **not** rely on string matching to check if the AI is correct. (Because `x + y` is the same logic as `y + x`).

* **Behavioral Equivalence:** The system compiles the AI's guessed code and runs it against the Ground Truth function using **dynamic property-based testing**.
* **Robustness:**
* **Math:** Automatically tests edge cases like `0`, negative numbers, large integers, and floats.
* **Strings:** Tests empty strings, whitespace, and special characters.
* **Lists:** Recursively generates lists of random integers or strings for complex inputs.


* **Safety:** The validation runs in a restricted scope to prevent unsafe execution.

## 📊 Categories

Puzzles are grouped to test specific reasoning capabilities:

* **MATH:** Arithmetic, modular arithmetic, number theory.
* **STRING:** Manipulation, slicing, concatenation, pattern matching.
* **LIST:** Aggregation, sorting, filtering, indexing.
* **BITWISE:** Binary logic (AND, OR, XOR), shifting operations.
