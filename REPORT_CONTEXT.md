# Report Context: `function_detective`

## Project Overview

- Project: `function_detective`
- Repository root: `e:\UNI\Semester 5\IM - Sherzod\Code`
- Primary workspace: [`clemgame-template/function_detective`](e:\UNI\Semester 5\IM%20-%20Sherzod\Code\clemgame-template\function_detective)
- Author of the game: Shahrukh Mohiuddin
- Main objective:
  - align `function_detective` structurally with the Clembench `taboo` game while preserving the game’s own mechanics
- Secondary objectives:
  - keep the game compatible with the current Clembench / `clemcore` usage pattern
  - improve benchmark metadata and repository hygiene
  - improve transcript readability
  - improve failure handling for Docker sandbox issues
  - simplify benchmark layout from `difficulty x domain` to domain-only evaluation

## Initial State And Problems

At the beginning of this work, `function_detective` existed as a custom Clembench game but diverged from `taboo` in several structural and maintenance-related ways.

Main issues identified early:

- folder and file layout did not closely follow the standard Clembench game pattern used by `taboo`
- naming was inconsistent and still contained remnants of `Functionigma`
- `README` files were stale, misleading, or broken
- game metadata was incomplete or not aligned with the expected Clembench style
- imports in `master.py` and `instancegenerator.py` were fragile under Clembench’s game loading mechanism
- game state implementation was incompatible with Clemcore expectations
- runtime function loading failed in some solve/reveal paths
- Docker sandbox failures were being treated as ordinary gameplay losses
- transcripts were noisy and contained too many raw internal structures
- benchmark experiments were modeled as a `3 x 5` matrix (`difficulty x domain`), even though difficulty calibration was not reliable
- generated result artifacts and local noise polluted the repository state

## Reference Used For Alignment

The structural reference point for the refactor was the upstream Clembench `taboo` game:

- Clembench repo: <https://github.com/clp-research/clembench>
- Taboo folder: <https://github.com/clp-research/clembench/tree/main/taboo>

The goal was not to clone `taboo` mechanically, but to align:

- naming conventions
- file roles
- folder structure
- benchmark metadata style
- generator / master organization
- overall Clembench “shape”

## Major Commit History

### `4086d19` `Align function_detective structure with clembench conventions`

This was the first major structural normalization pass.

Key changes:

- aligned public class naming in [`master.py`](e:\UNI\Semester 5\IM%20-%20Sherzod\Code\clemgame-template\function_detective\master.py)
- aligned generator naming in [`instancegenerator.py`](e:\UNI\Semester 5\IM%20-%20Sherzod\Code\clemgame-template\function_detective\instancegenerator.py)
- improved [`clemgame.json`](e:\UNI\Semester 5\IM%20-%20Sherzod\Code\clemgame-template\function_detective\clemgame.json)
- added missing game-level support files such as:
  - [`requirements.txt`](e:\UNI\Semester 5\IM%20-%20Sherzod\Code\clemgame-template\function_detective\requirements.txt)
  - [`.gitignore`](e:\UNI\Semester 5\IM%20-%20Sherzod\Code\clemgame-template\function_detective\.gitignore)
  - [`test_function_detective.py`](e:\UNI\Semester 5\IM%20-%20Sherzod\Code\clemgame-template\function_detective\test_function_detective.py)
  - prompt template files
- moved the game closer to the Clembench package pattern used by `taboo`

Why this mattered:

- it established a cleaner public surface for the game
- it reduced the gap between `function_detective` and standard Clembench games
- it created a better base for future debugging and reporting

### `d77526d` `Ignore and untrack generated benchmark results`

This commit focused on repository hygiene.

Key changes:

- updated root ignore rules so benchmark result artifacts do not bloat the main repository
- untracked generated `results/` content without deleting local files

Why this mattered:

- benchmark outputs belong in analysis workflows, not in the main source repository
- this reduced noise and prevented the repository from becoming large and messy

### `73bc848` `Clean function_detective structure and docs`

This commit focused on documentation and package cleanup.

Key changes:

- rewrote the root repository `README`
- rewrote [`clemgame-template/README.md`](e:\UNI\Semester 5\IM%20-%20Sherzod\Code\clemgame-template\README.md)
- fixed [`function_detective/README.md`](e:\UNI\Semester 5\IM%20-%20Sherzod\Code\clemgame-template\function_detective\README.md)
- documented the game as implemented by Shahrukh Mohiuddin

Why this mattered:

- the repository and game were no longer accurately described by the old documentation
- the report and future maintenance depend on docs matching the real codebase

### `bf7044c` `Simplify function_detective game loop`

This commit reduced leftover complexity in the game logic.

Key changes:

- removed obsolete `TRY` / iterative-guess style pathways
- simplified the core interaction path
- removed dead protocol surface and leftover template logic

Why this mattered:

- the game loop became closer to the smaller, clearer `taboo` style
- dead paths were a source of confusion and maintenance cost

### `d1b92b3` `Fix function_detective runtime imports and prompt examples`

This commit fixed critical runtime issues and prompt formatting issues.

Key changes:

- replaced fragile runtime imports with file-local function loading
- fixed `function_detective.*` package-style import problems under Clembench loading
- fixed prompt example formatting
- regenerated [`in/instances.json`](e:\UNI\Semester 5\IM%20-%20Sherzod\Code\clemgame-template\function_detective\in\instances.json)

Why this mattered:

- the game had concrete runtime failures during `SOLVE` handling
- the prompt examples had duplicate / awkward formatting in transcripts

## Important Post-Commit Progress Not Yet Captured In A Commit

After `d1b92b3`, substantial additional work was completed in the working tree. These changes are important for the report even if they are not yet represented in the visible commit log above.

### 1. `master.py` Refactor Into Smaller Helpers

The main game master in [`master.py`](e:\UNI\Semester 5\IM%20-%20Sherzod\Code\clemgame-template\function_detective\master.py) was further refactored to make it structurally cleaner and closer to the compact organization style of `taboo`.

Refactor highlights:

- extracted prompt-building helpers
- extracted input parsing helpers
- extracted probe recording helpers
- extracted solve handling helpers
- extracted probe summary helpers
- removed dead / unreachable code from `_on_after_game`
- removed the broad process-wide JSON encoder monkey patch

Why this mattered:

- `master.py` had become the biggest divergence from `taboo`
- the file is now more modular and easier to reason about without changing the game mechanics

### 2. Quality Score Normalized To `0-100`

The scorer in [`master.py`](e:\UNI\Semester 5\IM%20-%20Sherzod\Code\clemgame-template\function_detective\master.py) was changed so the benchmark score is explicitly on a `0-100` scale.

Current logic:

- `quality_score = 100 * efficiency * binary_success`
- aborted episodes produce `NaN`

Companion metrics logged:

- `quality_score`
- `binary_success`
- `efficiency`

Why this mattered:

- the score became easier to interpret and report
- it better matches expected benchmark-style presentation

### 3. Docker Sandbox Failure Handling

Docker / sandbox failures were originally showing up as ordinary gameplay losses. That behavior was corrected.

Changes:

- added Docker sandbox availability checks in [`utils.py`](e:\UNI\Semester 5\IM%20-%20Sherzod\Code\clemgame-template\function_detective\utils.py)
- added sandbox failure classification helpers
- made the game abort instead of lose when Docker is unavailable
- printed clear error messages to the console
- logged infrastructure failures to transcript/internal logs
- intentionally did **not** send infrastructure messages to the LLM as dialogue

Why this mattered:

- infrastructure failures are not gameplay failures
- benchmark scores should not penalize models for missing Docker
- transcripts should record the issue for humans without contaminating the LLM interaction

### 4. Transcript Cleanup

Transcript generation is primarily handled by Clembench / `clemcore`, but the visible transcript content depends heavily on what the game logs into interactions.

The game logs were cleaned to improve readability:

- raw dict dumps for probe results were replaced by readable probe summaries
- raw dict dumps for internal consistency were removed
- separate metric spam entries such as `test_accuracy`, `test_efficiency`, and `test_slack` were collapsed into a single `episode_summary`
- final solution logging was renamed and clarified via `submitted_solution`
- overall transcript flow now emphasizes:
  - player action
  - game output
  - final outcome
  - revealed hidden function
  - submitted solution
  - compact summary

Why this mattered:

- previous transcripts were technically informative but hard to read
- the report needs clean artifacts to discuss gameplay and outcomes

### 5. Benchmark Matrix Removed: Domain-Only Evaluation

Originally, the game used a `Difficulty x Domain` matrix. This was removed because difficulty calibration was not considered reliable enough for meaningful evaluation.

Current benchmark layout:

- `scalar_math`
- `pair_math`
- `string`
- `logic_formal`

Implemented in:

- [`instancegenerator.py`](e:\UNI\Semester 5\IM%20-%20Sherzod\Code\clemgame-template\function_detective\instancegenerator.py)
- regenerated [`in/instances.json`](e:\UNI\Semester 5\IM%20-%20Sherzod\Code\clemgame-template\function_detective\in\instances.json)

Additional changes:

- removed the unused empty `LIST_SEQUENCE` experiment bucket
- standardized generation settings:
  - `MAX_TURNS = 5`
  - `NUM_TESTS = 30`

Why this mattered:

- the evaluation is now easier to interpret
- the benchmark structure better matches what can actually be defended analytically

## Main Design Decisions And Rationale

### Use Local Imports Rather Than Package Imports For Game Loading

Decision:

- favor local file imports or dual-mode import fallbacks over assuming the game is installed as a package

Reason:

- Clembench loads game files directly from the folder
- package-style imports caused `ModuleNotFoundError` failures

### Abort On Infrastructure Failure Instead Of Recording A Loss

Decision:

- Docker or sandbox failures abort the episode

Reason:

- infrastructure unavailability is not a model failure
- counting it as a loss corrupts benchmark validity

### Do Not Send Infrastructure Failure Messages To The LLM

Decision:

- infrastructure failures are printed to console and logged internally, but not sent as player-facing game context

Reason:

- if sent to the LLM, the model may start responding with Docker troubleshooting instructions
- that corrupts the game interaction and transcript

### Keep Transcript Cleanup In The Game Logs Rather Than Rewriting Clembench Rendering

Decision:

- improve transcript readability by changing what the game logs, not by modifying Clembench HTML generation

Reason:

- transcript rendering appears to be owned by `clemcore`
- changing game log content is the most practical and least invasive way to improve presentation

### Remove Difficulty As A Benchmark Axis

Decision:

- move from `difficulty x domain` to domain-only experiments

Reason:

- difficulty was not being measured in a defensible way
- domain-only evaluation is easier to explain and more robust for reporting

## Current Game State

### Structure

Main files:

- [`master.py`](e:\UNI\Semester 5\IM%20-%20Sherzod\Code\clemgame-template\function_detective\master.py)
- [`instancegenerator.py`](e:\UNI\Semester 5\IM%20-%20Sherzod\Code\clemgame-template\function_detective\instancegenerator.py)
- [`utils.py`](e:\UNI\Semester 5\IM%20-%20Sherzod\Code\clemgame-template\function_detective\utils.py)
- [`functions.py`](e:\UNI\Semester 5\IM%20-%20Sherzod\Code\clemgame-template\function_detective\functions.py)
- [`protocol.py`](e:\UNI\Semester 5\IM%20-%20Sherzod\Code\clemgame-template\function_detective\protocol.py)
- [`clemgame.json`](e:\UNI\Semester 5\IM%20-%20Sherzod\Code\clemgame-template\function_detective\clemgame.json)
- [`README.md`](e:\UNI\Semester 5\IM%20-%20Sherzod\Code\clemgame-template\function_detective\README.md)
- [`in/instances.json`](e:\UNI\Semester 5\IM%20-%20Sherzod\Code\clemgame-template\function_detective\in\instances.json)

### Current Benchmark Layout

Experiments now correspond to domains only:

- `scalar_math`
- `pair_math`
- `string`
- `logic_formal`

Current domain counts from generated instances:

- `scalar_math`: 3 instances
- `pair_math`: 3 instances
- `string`: 1 instance
- `logic_formal`: 2 instances

### Current Scoring

Official episode score:

- `quality_score = 100 * efficiency * binary_success`

Implications:

- fully wrong final solutions score `0`
- late correct solutions still score low
- aborted infrastructure failures become `NaN`

### Current Transcript Behavior

Visible transcript now emphasizes:

- probes
- outputs
- final solve
- outcome
- hidden function reveal
- submitted solution
- compact end summary

### Current Test Status

Unit tests in [`test_function_detective.py`](e:\UNI\Semester 5\IM%20-%20Sherzod\Code\clemgame-template\function_detective\test_function_detective.py) were used repeatedly throughout the work to verify refactors and runtime fixes.

## Current Results Interpretation

The latest clean result snapshot shows:

- `% Played = 100`
- low quality score despite perfect protocol compliance

Interpretation:

- protocol formatting is no longer the core problem
- the main issue is that the score is very strict:
  - failure = `0`
  - late success = low score
- some episodes are nearly correct but still receive `0` under the current scoring rule
- some domains, especially string and logic-formal cases, still need better probing strategies or more turn budget

## Results And Reporting Artifacts

Relevant output files:

- [`results.csv`](e:\UNI\Semester 5\IM%20-%20Sherzod\Code\clemgame-template\results\results.csv)
- [`raw.csv`](e:\UNI\Semester 5\IM%20-%20Sherzod\Code\clemgame-template\results\raw.csv)
- [`results.html`](e:\UNI\Semester 5\IM%20-%20Sherzod\Code\clemgame-template\results\results.html)

Example transcript files:

- [`transcript.html` string example](e:\UNI\Semester 5\IM%20-%20Sherzod\Code\clemgame-template\results\gpt-4.1-2025-04-14\function_detective\string\instance_00006\transcript.html)
- [`transcript.html` pair-math example](e:\UNI\Semester 5\IM%20-%20Sherzod\Code\clemgame-template\results\gpt-4.1-2025-04-14\function_detective\pair_math\instance_00003\transcript.html)

## Open Questions / Remaining Work

The following items remain good candidates for future work or discussion in the report:

- whether the official score should include partial correctness rather than pure binary success
- whether `max_turns = 5` is too strict for some domains
- whether difficulty metadata in [`functions.py`](e:\UNI\Semester 5\IM%20-%20Sherzod\Code\clemgame-template\function_detective\functions.py) should be removed or retained only as annotation
- whether transcript-visible internal analytics should be trimmed even further
- whether old archived result folders should be cleaned up once Windows file locks are no longer an issue

## Suggested Prompt For ChatGPT Report Writing

Suggested usage:

> Use this context file plus the relevant commit history to write a report on the development and refinement of the `function_detective` Clembench game. The report should explain the project goals, initial issues, major refactor steps, important commits, runtime/debugging fixes, scoring changes, transcript improvements, benchmark redesign from difficulty x domain to domain-only, current results, and remaining limitations or next steps.

