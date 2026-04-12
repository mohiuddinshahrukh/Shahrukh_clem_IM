from typing import Dict, Tuple, List, Any
from dataclasses import dataclass, field
import ast
import importlib.util
import inspect
import json
import logging
import os

import numpy as np

from clemcore.backends import Model
from clemcore.clemgame import GameSpec, GameBenchmark, GameMaster, Player, DialogueGameMaster, GameScorer, \
    GameError, ParseError
from clemcore.clemgame.master import GameState
from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE, BENCH_SCORE

try:
    from protocol import TEST_TAG, SOLVE_TAG, OUTPUT_TAG
    from utils import parse_signature_with_types
    from utils import extract_function_code, validate_function_logic, get_sandbox_status, is_sandbox_failure
except ModuleNotFoundError:
    from function_detective.protocol import TEST_TAG, SOLVE_TAG, OUTPUT_TAG
    from function_detective.utils import parse_signature_with_types
    from function_detective.utils import (
        extract_function_code,
        validate_function_logic,
        get_sandbox_status,
        is_sandbox_failure,
    )


logger = logging.getLogger(__name__)


@dataclass
class FunctionDetectiveGameState(GameState):
    max_turns: int

    function_signature: str
    function_callable: str

    success: bool = False
    failure: bool = False
    aborted: bool = False

    probe_count: int = 0
    unique_output_count: int = 0
    output_novelty_rate: float = 0.0
    unique_input_count: int = 0
    unique_input_rate: float = 0.0

    probe_args: str = ""
    probe_round: int = 0
    probe_output: str = ""

    test_slack: float = 0.0
    test_accuracy: float = 0.0
    test_efficiency: float = 0.0

    parse_error_count: int = 0
    runtime_error_count: int = 0

    observed_pairs: List[Dict] = field(default_factory=list)

    internal_consistency_score: float = 0.0
    internal_consistency_all_observed: bool = False
    internal_consistency_violations: int = 0

    def __post_init__(self):
        # Initialize Clemcore's base fields such as outcome/current_turn.
        super().__init__()


class FunctionGuesser(Player):
    def __init__(self, model: Model):
        super().__init__(model)

    def _custom_response(self, messages):
        """
        Convert any UserMessage or non-string to string before returning.
        This prevents JSON serialization errors.
        """
        sanitized = []
        for m in messages:
            if hasattr(m, "content"):
                sanitized.append(str(m.content))
            else:
                sanitized.append(str(m))
        return " ".join(sanitized)


class FunctionDetective(DialogueGameMaster):

    def __init__(self, game_spec: GameSpec, experiment: Dict, player_models: List[Model]):
        super().__init__(game_spec, experiment, player_models)
        self.experiment = experiment

    def _format_arg(self, value: Any) -> str:
        if isinstance(value, str):
            return repr(value)
        if isinstance(value, bool):
            return "True" if value else "False"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    def _format_args_for_log(self, args: List[Any]) -> str:
        return ", ".join(self._format_value(arg) for arg in args)

    def _build_example_lines(self) -> Tuple[str, str]:
        example_test_line = f"{TEST_TAG} <example>"
        example_output_line = f"{OUTPUT_TAG} <example>"

        if not self.test_cases:
            return example_test_line, example_output_line

        example_case = self.test_cases[0]
        example_args = example_case.get("args", [])
        example_expected = example_case.get("expected")

        if isinstance(example_args, list):
            if len(example_args) == 1:
                arg_text = self._format_arg(example_args[0])
            else:
                arg_text = ", ".join(self._format_arg(arg) for arg in example_args)
        else:
            arg_text = self._format_arg(example_args)

        return (
            f"{TEST_TAG} {arg_text}",
            f"{OUTPUT_TAG} {self._format_value(example_expected)}",
        )

    def _build_initial_prompt(self, example_test_line: str, example_output_line: str) -> str:
        param_list_str = ", ".join(
            f"{name}: {param_type}" for name, param_type in zip(self.param_names, self.param_types)
        ) if self.param_names else "none"
        variables = ", ".join(f"<value{i + 1}>" for i in range(self.num_params))

        prompt = self.experiment["guesser_initial_prompt"]
        replacements = {
            "$TEST_TAG$": TEST_TAG,
            "$SOLVE_TAG$": SOLVE_TAG,
            "$OUTPUT_TAG$": OUTPUT_TAG,
            "$SIGNATURE_OF_CURRENT_FUNCTION$": self.signature,
            "$NUM_PARAMS$": str(self.num_params),
            "$PARAM_LIST$": param_list_str,
            "$CATEGORY$": str(self.category),
            "$MAX_TURNS$": str(self.max_turns),
            "$VARIABLES$": variables,
            "$EXAMPLE_TEST_LINE$": example_test_line,
            "$EXAMPLE_OUTPUT_LINE$": example_output_line,
        }

        for placeholder, value in replacements.items():
            prompt = prompt.replace(placeholder, value)

        return prompt

    def _parse_input_values(self, response: str):
        raw = response[len(TEST_TAG):].strip()
        if self.num_params == 0:
            return tuple()

        tuple_text = f"({raw})" if self.num_params > 1 else raw
        parsed = ast.literal_eval(tuple_text)

        if self.num_params == 1:
            return (parsed,)
        if isinstance(parsed, tuple):
            return parsed
        return tuple(parsed)

    def _inputs_match_signature(self, parsed_values: Tuple[Any, ...]) -> bool:
        if len(parsed_values) != self.num_params:
            self.log_to_self("Input parameter count mismatch.", "end game")
            return False

        for value, expected in zip(parsed_values, self.param_types):
            if expected in ("int", "Integer"):
                if not isinstance(value, int) or isinstance(value, bool):
                    return False
            elif expected in ("float", "double"):
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    return False
            elif expected in ("str", "string"):
                if not isinstance(value, str):
                    return False
            elif expected in ("bool", "boolean"):
                if not isinstance(value, bool):
                    return False

        return True

    def _record_probe(self, args: List[Any], result: Any) -> None:
        self.state.probe_args = args
        self.state.probe_round += 1
        self.state.probe_output = result
        self.state.observed_pairs.append({
            "round": self.state.probe_round,
            "args": self.state.probe_args,
            "output": self.state.probe_output,
        })

    def _set_efficiency_metrics(self) -> None:
        denominator = max(1, self.state.max_turns - 1)
        efficiency = 1 - (self.state.probe_round - 1) / denominator
        self.state.test_efficiency = efficiency
        self.state.test_slack = self.state.max_turns - self.state.probe_round

    def _build_observed_cases(self) -> List[Dict[str, Any]]:
        return [
            {"args": pair["args"], "expected": pair["output"]}
            for pair in (self.state.observed_pairs or [])
        ]

    def _compute_internal_consistency(self, guessed_code: str) -> None:
        observed_cases = self._build_observed_cases()
        if not observed_cases:
            self.state.internal_consistency_score = 1.0
            self.state.internal_consistency_all_observed = True
            self.state.internal_consistency_violations = 0
        else:
            obs_all_correct, obs_accuracy, feedback = validate_function_logic(guessed_code, observed_cases)
            if is_sandbox_failure(feedback):
                raise RuntimeError(feedback)
            self.state.internal_consistency_score = float(obs_accuracy)
            self.state.internal_consistency_all_observed = bool(obs_all_correct)
            self.state.internal_consistency_violations = int(round((1.0 - obs_accuracy) * len(observed_cases)))

    def _compute_probe_summary(self) -> Dict[str, float]:
        pairs = self.state.observed_pairs or []
        probe_count = len(pairs)

        unique_input_keys = {self._stable_key(pair.get("args")) for pair in pairs}
        unique_output_keys = {self._stable_key(pair.get("output")) for pair in pairs}

        unique_input_count = len(unique_input_keys)
        unique_output_count = len(unique_output_keys)
        novelty_rate = (unique_output_count / probe_count) if probe_count else 0.0
        input_diversity = (unique_input_count / probe_count) if probe_count else 0.0

        output_entropy_bits = 0.0
        if probe_count:
            from collections import Counter
            import math

            out_counts = Counter(self._stable_key(pair.get("output")) for pair in pairs)
            for count in out_counts.values():
                probability = count / probe_count
                output_entropy_bits -= probability * math.log2(probability)

        return {
            "probe_count": probe_count,
            "unique_input_count": unique_input_count,
            "unique_output_count": unique_output_count,
            "novelty_rate": float(novelty_rate),
            "input_diversity": float(input_diversity),
            "output_entropy_bits": float(output_entropy_bits),
        }

    def _log_probe_summary(self) -> None:
        summary = self._compute_probe_summary()
        for key, value in summary.items():
            self.log_key(key, value)

        self.log_key("internal_consistency_score", getattr(self.state, "internal_consistency_score", 0.0))
        self.log_key(
            "internal_consistency_all_observed",
            getattr(self.state, "internal_consistency_all_observed", False),
        )
        self.log_key(
            "internal_consistency_violations",
            getattr(self.state, "internal_consistency_violations", 0),
        )

        self.log_to_self(
            "probe_information_summary",
            (
                f"Probes: {summary['probe_count']} | "
                f"Unique inputs: {summary['unique_input_count']} "
                f"(diversity={summary['input_diversity']:.2f}) | "
                f"Unique outputs: {summary['unique_output_count']} "
                f"(novelty={summary['novelty_rate']:.2f}, "
                f"entropy={summary['output_entropy_bits']:.2f} bits)"
            )
        )

    def _log_probe_result(self) -> None:
        self.log_to_self(
            "probe_result",
            (
                f"Probe {self.state.probe_round}\n"
                f"Inputs: {self._format_args_for_log(self.state.probe_args)}\n"
                f"Output: {self._format_value(self.state.probe_output)}"
            )
        )

    def _log_episode_summary(self) -> None:
        self.log_to_self(
            "episode_summary",
            (
                f"Turns used: {self.state.probe_round}/{self.state.max_turns}\n"
                f"Turns remaining: {self.state.max_turns - self.state.probe_round}\n"
                f"Accuracy: {self.state.test_accuracy:.3f}\n"
                f"Efficiency: {self.state.test_efficiency:.3f}\n"
                f"Slack: {int(self.state.test_slack)}\n"
                f"Observed-probe consistency: {self.state.internal_consistency_score:.3f}\n"
                f"Parse errors: {self.state.parse_error_count}\n"
                f"Runtime errors: {self.state.runtime_error_count}"
            )
        )

    def _handle_call_action(self, input_line: str) -> None:
        try:
            output = self.run_dynamic_function(self.state.function_callable, input_line)
            self.set_context_for(self.guesser_player, f"{OUTPUT_TAG} {output}")
            self._log_probe_result()
        except Exception as error:
            self.state.runtime_error_count += 1
            self.set_context_for(self.guesser_player, f"Error executing function: {error}")

    def _handle_solve_action(self, guessed_code: str) -> None:
        actual_func = self.load_game_function(self.state.function_callable)
        actual_code = self._clean_source_for_reveal(actual_func)
        is_correct, accuracy, feedback = validate_function_logic(guessed_code, self.test_cases)

        if is_sandbox_failure(feedback):
            self.state.aborted = True
            self.state.failure = False
            self.state.success = False
            abort_message = (
                "Infrastructure Failure: Docker sandbox is unavailable.\n"
                f"{feedback}\n\n"
                "Start Docker Desktop and rerun the benchmark."
            )
            logger.error(abort_message)
            print(abort_message)
            self.log_to_self("outcome", "Game Verdict: ABORTED (sandbox unavailable)")
            self.log_to_self("infrastructure_error", abort_message)
            return

        self._compute_internal_consistency(guessed_code)
        self.state.test_accuracy = accuracy
        self._set_efficiency_metrics()

        if is_correct:
            self.state.success = True
            self.log_to_self("outcome", "Game Verdict: WIN")
            reveal_msg = f"That is correct!\nThe hidden function was:\n\n{actual_code}"
        else:
            self.state.success = False
            self.state.failure = True
            self.log_to_self("outcome", "Game Verdict: LOSS")
            reveal_msg = f"Incorrect.\n{feedback}\n\nHidden function:\n\n{actual_code}"

        self.log_to_self("reveal_function", reveal_msg)
        self.log_to_self("submitted_solution", guessed_code)
        self._log_episode_summary()

    def _on_setup(self, **game_instance):

        self.game_instance = game_instance
        self.signature = game_instance["signature"]
        self.max_turns = self.experiment["max_turns"]
        self.test_cases = game_instance.get("test_cases", [])
        example_test_line, example_output_line = self._build_example_lines()
        self.param_names, self.param_types, self.return_type = parse_signature_with_types(self.signature)
        self.num_params = len(self.param_names)
        self.category = game_instance.get("category", "MATH")
        prompt = self._build_initial_prompt(example_test_line, example_output_line)
        self.guesser_player = FunctionGuesser(self.player_models[0])
        self.add_player(self.guesser_player, initial_context=prompt)

        self.state = FunctionDetectiveGameState(
            max_turns=self.max_turns,
            function_signature=self.signature,
            function_callable=self.game_instance["callable"],
        )

        sandbox_ok, sandbox_message = get_sandbox_status()
        if not sandbox_ok:
            user_message = (
                "Docker sandbox is not available. FunctionDetective cannot validate solutions without it.\n"
                f"{sandbox_message}\n"
                "Start Docker Desktop and rerun the benchmark."
            )
            logger.error(user_message)
            print(user_message)
            self.state.aborted = True
            self.log_to_self("outcome", "Game Verdict: ABORTED (sandbox unavailable)")
            self.log_to_self("infrastructure_error", user_message)

    def _format_value(self, v):
        """Human-readable, stable formatting for transcript + context."""
        if isinstance(v, str):
            return repr(v)  # ensures quotes are shown
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    def _stable_key(self, obj: Any) -> str:
        """Stable string key for args/outputs so we can count uniques robustly."""
        try:
            return json.dumps(obj, default=str, sort_keys=True)
        except Exception:
            return str(obj)

    def check_given_inputs(self, response: str) -> bool:
        if not isinstance(response, str):
            return False
        try:
            parsed_values = self._parse_input_values(response.strip())
            return self._inputs_match_signature(parsed_values)
        except Exception:
            return False

    def extract_given_inputs(self, response: str):
        return list(self._parse_input_values(response.strip()))

    def _clean_source_for_reveal(self, func) -> str:
        """
        Return clean function source starting at `def ...` and remove docstring block.
        This prevents decorator metadata lines from appearing in the reveal.
        """
        try:
            source_lines = inspect.getsource(func).splitlines()
        except OSError:
            return "# Source code unavailable"

        # 1) Drop everything before the first `def `
        def_idx = None
        for i, ln in enumerate(source_lines):
            if ln.lstrip().startswith("def "):
                def_idx = i
                break
        if def_idx is None:
            return "# Source code unavailable"

        lines = source_lines[def_idx:]

        # 2) Remove an immediate docstring (triple-quoted) if present
        # Find first non-empty line after def
        j = 1
        while j < len(lines) and lines[j].strip() == "":
            j += 1

        if j < len(lines):
            stripped = lines[j].lstrip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                quote = stripped[:3]
                # If docstring starts and ends on same line
                if stripped.count(quote) >= 2 and stripped.strip() != quote:
                    # remove that one docstring line
                    lines.pop(j)
                else:
                    # remove until closing triple quote
                    lines.pop(j)  # remove start
                    while j < len(lines):
                        if quote in lines[j]:
                            lines.pop(j)  # remove end
                            break
                        lines.pop(j)

        return "\n".join(lines).strip()

    def load_game_function(self, function_name: str):
        module_path = os.path.join(os.path.dirname(__file__), "functions.py")
        spec = importlib.util.spec_from_file_location(
            "_function_detective_functions",
            module_path,
        )
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load module from '{module_path}'")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        try:
            return getattr(module, function_name)
        except AttributeError:
            raise ValueError(f"Function '{function_name}' not found in '{module_path}'")

    def run_dynamic_function(self, func_name: str, text: str) -> str:
        func = self.load_game_function(func_name)
        args = self.extract_given_inputs(text)

        try:
            result = func(*args)
        except Exception as e:
            raise RuntimeError(f"Function execution error: {e}")
        self._record_probe(args, result)
        return self._format_value(result)

    def _parse_response(self, player: Player, response: str) -> Tuple[str, Any]:
        print(f"PROBE: {response}")
        response_str = str(response).strip()

        # ----------------------------
        # SOLVE mode: one solve per turn; reasoning allowed after code; no TEST in same message
        # ----------------------------
        if response_str.startswith(SOLVE_TAG):
            lines = response_str.splitlines()

            # Disallow multiple actions in one message (SOLVE + TEST)
            if any(ln.strip().startswith(TEST_TAG) for ln in lines[1:]):
                raise ParseError(f"Only one action per turn. Do not include {TEST_TAG} lines with {SOLVE_TAG}.")

            extracted_code = extract_function_code(response_str)
            if not extracted_code:
                raise ParseError(f"{SOLVE_TAG} must contain a markdown code block (```python ... ```).")

            # Allow extra reasoning text after the code block.
            return "solve", extracted_code

        # ----------------------------
        # TEST mode: exactly one TEST line; reasoning allowed after; no additional TEST/SOLVE lines
        # ----------------------------
        if response_str.startswith(TEST_TAG):
            lines = response_str.splitlines()
            input_line = lines[0].strip()

            # Disallow a second TEST or SOLVE anywhere later in the same message
            for ln in lines[1:]:
                s = ln.strip()
                if s.startswith(TEST_TAG):
                    raise ParseError(f"Only ONE {TEST_TAG} is allowed per turn.")
                if s.startswith(SOLVE_TAG):
                    raise ParseError(f"Only one action per turn. Do not include {SOLVE_TAG} with {TEST_TAG}.")

            if not self.check_given_inputs(input_line):
                raise ParseError(f"{TEST_TAG} values do not match the required signature/types.")

            # Allow reasoning lines after TEST; we ignore them.
            return "call", {"input_line": input_line}

        raise ParseError(f"Invalid format. Must be '{TEST_TAG} ...' or '{SOLVE_TAG} ```python ...```'")

    def _advance_game(self, player: Player, parsed_response: Tuple[str, Any]):

        action_type, content = parsed_response

        if action_type == "call":
            self._handle_call_action(content["input_line"])

        elif action_type == "solve":
            self._handle_solve_action(content)

    def _on_parse_error(self, error: GameError):
        self.set_context_for(
            self.guesser_player,
            f"Game Violation: {error}\n"
            f"Output ONLY:\n{TEST_TAG} ...\nOR\n{SOLVE_TAG} ```python ... ```\n"
            "No extra text."
        )
        self.state.parse_error_count += 1
        print(f"Parse Error: {error}. Consuming a turn.")

    def _does_game_proceed(self):
        if self.current_round >= self.max_turns:
            return False
        return not (self.state.aborted or self.state.failure or self.state.success)

    def compute_turn_score(self):
        return 1 if self.state.success else 0

    def compute_episode_score(self):
        if self.state.success:
            return 100
        return 0

    def _on_after_game(self):
        # --- Timeout handling: if game ended due to max turns without {SOLVE_TAG} ---
        if (self.current_round >= self.max_turns) and not (
                self.state.success or self.state.failure or self.state.aborted):
            self.state.failure = True
            self.log_to_self("outcome", "Game Verdict: LOSS (turn limit reached)")

        self.log_key(METRIC_ABORTED, self.state.aborted)
        self.log_key(METRIC_LOSE, self.state.failure)
        self.log_key(METRIC_SUCCESS, self.state.success)

        # Logging for analytics, but won't be used for the main score

        efficiency = getattr(self.state, "test_efficiency", 0.0)
        self.log_key("efficiency", efficiency)

        slack = getattr(self.state, "test_slack", 0.0)
        self.log_key("slack", slack)

        accuracy = getattr(self.state, "test_accuracy", 0.0)
        self.log_key("accuracy", accuracy)

        self.log_key("parse_error_count", self.state.parse_error_count)
        self.log_key("runtime_error_count", self.state.runtime_error_count)
        self.log_key("turns_used", self.state.probe_round)  # or current_round+1 if you prefer “turns” semantics
        self.log_key("max_turns", self.state.max_turns)
        self.log_key("observed_pairs", self.state.observed_pairs)
        self._log_probe_summary()


class FunctionDetectiveScorer(GameScorer):
    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def compute_round_score(self, round_idx, round_events: List[Dict]) -> None:
        for event in round_events:
            if event["action"]["type"] == "player_response":
                self.log_round_score(round_idx, 'response_received', 1)

    def compute_episode_scores(self, interactions: Dict):
        binary_success = 1 if interactions.get(METRIC_SUCCESS, False) else 0
        efficiency = float(interactions.get("efficiency", 0.0))
        quality_score = 100.0 * efficiency * binary_success
        if interactions.get(METRIC_ABORTED, False):
            quality_score = np.nan

        self.log_episode_score(BENCH_SCORE, quality_score)
        self.log_episode_score("quality_score", quality_score)
        self.log_episode_score("binary_success", binary_success)
        self.log_episode_score("efficiency", efficiency * 100.0)


class FunctionDetectiveGameBenchmark(GameBenchmark):
    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return FunctionDetective(self.game_spec, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return FunctionDetectiveScorer(self.game_name, experiment, game_instance)
