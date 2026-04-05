# Function Detective

Implemented by: Sherzod et al.

In this game a single player probes a hidden Python function through input-output interaction and must infer a functionally equivalent implementation. The player receives the function signature, can test candidate inputs, observes the resulting outputs, and eventually submits Python code as the final solution.

The game targets inductive reasoning over program behavior. It measures whether a model can choose informative probes, extract a rule from sparse observations, and produce code that matches the hidden function on held-out tests.

### Instantiation

We instantiate the game with function families grouped by difficulty and domain. Each experiment specifies a maximum number of turns and includes pre-generated hidden-function test cases used for evaluation. Instances are created from the registry in `functions.py`, and each instance stores the hidden callable name, signature, category, difficulty, candidate metadata, and evaluation tests.

### Evaluation

We measure the following metrics at the episode level:

1. **Success**: Whether the submitted solution functionally matches the hidden function on static tests.
2. **Abort**: Whether the interaction was aborted because of invalid behavior.
3. **Efficiency**: How quickly the player solved the task relative to the turn budget.
4. **Accuracy**: The final held-out test accuracy of the submitted solution.
5. **Internal Consistency**: Whether the submitted solution agrees with the outputs observed during probing.
6. **Probe Diversity**: How varied the tested inputs and resulting outputs were across the episode.

### Package Structure

The game package follows the standard Clembench layout:

- `master.py`: game master, scorer, benchmark, and runtime validation logic
- `instancegenerator.py`: instance generation from the hidden-function registry
- `functions.py`: hidden functions and registry metadata
- `utils.py`: test generation, parsing, and code validation helpers
- `protocol.py`: interaction tags used in prompts and responses
- `resources/`: prompt templates and other game assets
- `in/instances.json`: generated benchmark instances

### Running

Generate instances:

```bash
python function_detective/instancegenerator.py
```

Run the game with Clembench:

```bash
clem run function_detective -m <model_name>
```

Score the results:

```bash
clem score function_detective
```
