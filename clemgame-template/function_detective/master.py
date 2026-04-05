from typing import Dict, Tuple, List, Any
from dataclasses import dataclass, field
import ast
import importlib
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
from protocol import TEST_TAG, SOLVE_TAG, TRY_TAG, NEXT_TEST_TAG, OUTPUT_TAG
from utils import parse_signature_with_types
from utils import extract_function_code, validate_function_logic

# --- FINAL ROBUST FIX: MONKEY PATCH FOR JSON SERIALIZATION ---
_original_json_default = json.JSONEncoder.default


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default


def _patched_json_default(self, obj):
    # 1. Explicitly handle the classes we've seen so far
    known_classes = [
        "UserMessage", "SystemMessage", "AssistantMessage",
        "Message", "ChatbotMessage", "ApiMeta"
    ]

    if type(obj).__name__ in known_classes:
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)

    # 2. UNIVERSAL FALLBACK:
    # If the object has a __dict__ attribute (standard Python objects),
    # serialize that instead of crashing. This catches any future unknown classes.
    if hasattr(obj, "__dict__"):
        return obj.__dict__

    # 3. Last resort for weird objects: return their string representation
    # (Only do this if it's not a basic type that json handles naturally)
    if not isinstance(obj, (int, float, str, bool, list, dict, tuple, type(None))):
        return str(obj)

    return _original_json_default(self, obj)


json.JSONEncoder.default = _patched_json_default
# -----------------------------------------------------------


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

    # --- Hypothesis tracking (candidate IDs) ---
    candidate_ids: List[str] = field(default_factory=list)
    hypothesis_history: List[Dict] = field(default_factory=list)
    hypothesis_format_error_count: int = 0

    # Episode-level hypothesis quality metrics
    hypothesis_top1_hit_at_stop: int = 0
    hypothesis_top3_hit_at_stop: int = 0
    hypothesis_prob_true_at_stop: float = 0.0
    hypothesis_avg_negative_log_prob_true: float = 0.0
    hypothesis_top1_switches: int = 0

    # Penalty-less mode
    iterative_guesses_enabled: bool = False
    guess_count: int = 0
    wrong_guess_count: int = 0

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
    """
    Template class for game master.
    """

    def __init__(self, game_spec: GameSpec, experiment: Dict, player_models: List[Model]):
        super().__init__(game_spec, experiment, player_models)
        self.experiments = experiment
        self.success = False

    ##################
    def _on_setup(self, **game_instance):

        self.game_instance = game_instance
        self.signature = game_instance["signature"]
        self.max_turns = self.experiments["max_turns"]
        self.test_cases = game_instance.get("test_cases", [])
        # ---- Example I/O pair (free, shown in initial prompt) ----
        example_test_line = f"{TEST_TAG} <example>"
        example_output_line = f"{OUTPUT_TAG} <example>"

        if self.test_cases:
            ex = self.test_cases[0]
            ex_args = ex.get("args", [])
            ex_expected = ex.get("expected", None)

            # Format args exactly like the player must type them
            def _format_arg(a):
                if isinstance(a, str):
                    return repr(a)  # quotes
                if isinstance(a, bool):
                    return "True" if a else "False"
                if isinstance(a, float):
                    return f"{a:.3f}"
                return str(a)

            if isinstance(ex_args, list):
                if len(ex_args) == 1:
                    arg_text = _format_arg(ex_args[0])
                else:
                    arg_text = ", ".join(_format_arg(a) for a in ex_args)
            else:
                # If stored as a scalar, still handle it
                arg_text = _format_arg(ex_args)

            example_test_line = f"{TEST_TAG} {arg_text}"
            example_output_line = f"{OUTPUT_TAG} {self._format_value(ex_expected)}"

        # parse signature to get param names/types
        self.param_names, self.param_types, self.return_type = parse_signature_with_types(self.signature)
        self.num_params = len(self.param_names)

        # category is optional metadata (does not get used in the simplified prompt)
        self.category = game_instance.get("category", "MATH")

        # parameter list string for prompt
        param_list_str = ", ".join(
            f"{n}: {t}" for n, t in zip(self.param_names, self.param_types)
        ) if self.param_names else "none"
        variables = ", ".join(f"<value{i + 1}>" for i in range(self.num_params))

        # Load template and replace protocol placeholders
        prompt = self.experiments["initial_prompt_guesser"]
        prompt = prompt.replace("$TEST_TAG$", TEST_TAG)
        prompt = prompt.replace("$SOLVE_TAG$", SOLVE_TAG)
        prompt = prompt.replace("$TRY_TAG$", TRY_TAG)
        prompt = prompt.replace("$NEXT_TEST_TAG$", NEXT_TEST_TAG)
        prompt = prompt.replace("$OUTPUT_TAG$", OUTPUT_TAG)

        # Replace standard placeholders
        prompt = prompt.replace("$SIGNATURE_OF_CURRENT_FUNCTION$", self.signature)
        prompt = prompt.replace("$NUM_PARAMS$", str(self.num_params))
        prompt = prompt.replace("$PARAM_LIST$", param_list_str)
        prompt = prompt.replace("$CATEGORY$", str(self.category))
        prompt = prompt.replace("$MAX_TURNS$", str(self.max_turns))
        prompt = prompt.replace("$VARIABLES$", variables)
        prompt = prompt.replace("$EXAMPLE_TEST_LINE$", example_test_line)
        prompt = prompt.replace("$EXAMPLE_OUTPUT_LINE$", example_output_line)
        # Flags
        # use_hints = _env_flag("FUNCTION_DETECTIVE_USE_HINTS", default=False)
        # iterative = _env_flag("FUNCTION_DETECTIVE_ITERATIVE_GUESSES", default=False)
        #
        # # Optional hint (only if enabled and present in instance)
        # if use_hints:
        #     hint = game_instance.get("hint", "")
        #     if hint:
        #         prompt += f"\n\nHINT: {hint}\n"
        #
        # # Mode instructions (keep short, no candidates, no hypotheses)
        # if iterative:
        #     prompt += (
        #         f"\n\nMODE: Iterative {TRY_TAG} mode is ENABLED.\n"
        #         "On EVERY turn you MUST output:\n"
        #         f"{TRY_TAG} ```python ... ```\n"
        #         f"{NEXT_TEST_TAG} <inputs>\n"
        #         f"Do NOT use {TEST_TAG} or {SOLVE_TAG} in this mode.\n"
        #     )
        # else:
        #     prompt += (
        #         f"\n\nMODE: Iterative {TRY_TAG} mode is DISABLED (strict mode).\n"
        #         f"Use {TEST_TAG} to probe. When you output {SOLVE_TAG}, the game ends immediately (WIN or LOSS).\n"
        #     )

        # Add player with initial context
        self.guesser_player = FunctionGuesser(self.player_models[0])
        self.add_player(self.guesser_player, initial_context=prompt)

        # Initialize state
        self.state = FunctionDetectiveGameState(
            max_turns=self.max_turns,
            function_signature=self.signature,
            function_callable=self.game_instance["callable"],
        )
        # self.state.iterative_guesses_enabled = iterative

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

    def _compute_hypothesis_quality_metrics(self):
        hist = self.state.hypothesis_history or []
        true_id = self.state.function_callable

        if not hist:
            self.state.hypothesis_top1_hit_at_stop = 0
            self.state.hypothesis_top3_hit_at_stop = 0
            self.state.hypothesis_prob_true_at_stop = 0.0
            self.state.hypothesis_avg_negative_log_prob_true = 0.0
            self.state.hypothesis_top1_switches = 0
            return

        # top1 switches (churn)
        top1_list = [h["top1"] for h in hist]
        self.state.hypothesis_top1_switches = sum(
            1 for i in range(1, len(top1_list)) if top1_list[i] != top1_list[i - 1]
        )

        # Metrics "at stop": use last recorded hypotheses (last probe turn)
        last = hist[-1]
        self.state.hypothesis_prob_true_at_stop = float(last.get("p_true", 0.0))

        pairs = list(zip(last["candidates"], last["confidences"]))
        pairs.sort(key=lambda x: x[1], reverse=True)

        top1 = pairs[0][0]
        top3 = [c for c, _ in pairs[:3]]

        self.state.hypothesis_top1_hit_at_stop = int(top1 == true_id)
        self.state.hypothesis_top3_hit_at_stop = int(true_id in top3)

        # Avg negative log prob of true across turns (log loss)
        import math
        eps = 1e-9
        nlls = [-math.log(max(eps, float(h.get("p_true", 0.0)))) for h in hist]
        self.state.hypothesis_avg_negative_log_prob_true = float(sum(nlls) / len(nlls))

    def _parse_hypothesis_block(self, text: str):

        """
        Parse:
          HYPOTHESES:
          - candidate: <id>
          CONFIDENCE:
          - <float>

        Returns (candidate_list, confidence_list) or (None, None) if invalid/missing.
        """

        if not text:
            return None, None

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        try:
            h_idx = lines.index("HYPOTHESES:")
            c_idx = lines.index("CONFIDENCE:")
        except ValueError:
            return None, None

        hyp_lines = lines[h_idx + 1: c_idx]
        conf_lines = lines[c_idx + 1:]

        candidates = []
        for ln in hyp_lines:
            if not ln.startswith("-"):
                continue
            item = ln.lstrip("-").strip()
            if item.lower().startswith("candidate:"):
                candidates.append(item.split(":", 1)[1].strip())

        confidences = []
        for ln in conf_lines:
            if not ln.startswith("-"):
                continue
            item = ln.lstrip("-").strip()
            try:
                confidences.append(float(item))
            except Exception:
                return None, None

        if not candidates or len(candidates) != len(confidences):
            return None, None

        s = sum(confidences)
        if s <= 0:
            return None, None

        # Require close-to-1 to avoid arbitrary scaling
        if abs(s - 1.0) > 0.05:
            return None, None

        if len(set(candidates)) != len(candidates):
            return None, None

        # Normalize tiny numeric drift
        confidences = [x / s for x in confidences]
        return candidates, confidences

    def _record_hypotheses(self, candidates, confidences):
        """
        Log hypotheses for this round and keep a compact history.
        """
        true_id = self.state.function_callable
        cand_to_p = {c: p for c, p in zip(candidates, confidences)}

        top1 = max(cand_to_p.items(), key=lambda kv: kv[1])[0]
        p_true = float(cand_to_p.get(true_id, 0.0))

        self.state.hypothesis_history.append({
            "round": int(self.state.probe_round),
            "top1": top1,
            "p_true": p_true,
            "candidates": list(candidates),
            "confidences": list(confidences),
        })

    def check_given_inputs(self, response: str) -> bool:
        # Accept only responses starting with "TEST:"
        if not isinstance(response, str):
            return False
        response = response.strip()
        raw = response[len(f"{TEST_TAG}"):].strip()
        if self.num_params == 0:
            return raw == ""
        # Split by commas, but allow commas inside strings by using a small parser
        try:
            # Build a tuple text and parse using ast.literal_eval
            tuple_text = f"({raw})" if self.num_params > 1 else raw
            parsed = ast.literal_eval(tuple_text)
            if self.num_params == 1:
                parsed = (parsed,)
            if not isinstance(parsed, tuple):
                parsed = tuple(parsed)
            if len(parsed) != self.num_params:
                self.log_to_self("Input parameter count mismatch.", "end game")
                return False
            # Type-check each parsed value against expected type strings
            for val, expected in zip(parsed, self.param_types):
                if expected in ("int", "Integer"):
                    if not isinstance(val, int) or isinstance(val, bool):
                        return False
                elif expected in ("float", "double"):
                    if not isinstance(val, (int, float)) or isinstance(val, bool):
                        return False
                elif expected in ("str", "string"):
                    if not isinstance(val, str):
                        return False
                elif expected in ("bool", "boolean"):
                    if not isinstance(val, bool):
                        return False
                else:
                    # Any or unknown type: accept but no strict check
                    pass
            return True
        except Exception as e:
            # parsing failed
            print(f"Exception caught: {e}")
            return False

    def extract_given_inputs(self, response: str):
        raw = response.strip()[len(f"{TEST_TAG}"):].strip()
        if self.num_params == 0:
            return []
        # Use ast.literal_eval to safely parse values
        if self.num_params == 1:
            val = ast.literal_eval(raw)
            return [val]
        parsed = ast.literal_eval(f"({raw})")
        return list(parsed)

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

    def load_function(self, module_name: str, function_name: str):
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise ValueError(f"Cannot import module '{module_name}' — {e}")

        try:
            func = getattr(module, function_name)
        except AttributeError:
            raise ValueError(f"Function '{function_name}' not found in '{module_name}'")

        return func

    def run_dynamic_function(self, func_name: str, text: str) -> str:
        func = self.load_function("functions", func_name)
        args = self.extract_given_inputs(text)

        try:
            result = func(*args)
        except Exception as e:
            raise RuntimeError(f"Function execution error: {e}")
        # Return a string representation consistent with return type
        # For strings, return as-is; for bool, True/False; for numbers, str()
        self.state.probe_args = args
        self.state.probe_round += 1
        self.state.probe_output = result

        self.state.observed_pairs.append({
            "round": self.state.probe_round,
            "args": self.state.probe_args,
            "output": self.state.probe_output,
        })

        return self._format_value(result)

    def _parse_response(self, player: Player, response: str) -> Tuple[str, Any]:
        print(f"PROBE: {response}")
        response_str = str(response).strip()

        # If iterative mode is enabled, only TRY is allowed.
        if self.state.iterative_guesses_enabled and not response_str.startswith(TRY_TAG):
            raise ParseError(
                f"Iterative {TRY_TAG} mode is enabled: you MUST use {TRY_TAG} ... with {NEXT_TEST_TAG} ... every turn."
            )

        # ----------------------------
        # TRY mode: guess + mandatory NEXT_TEST (ONE per turn), reasoning allowed after
        # ----------------------------
        if response_str.startswith(TRY_TAG):
            lines = response_str.splitlines()

            # Require TRY on first line
            if not lines[0].strip().startswith(TRY_TAG):
                raise ParseError(f"{TRY_TAG} must be the first line.")

            extracted_code = extract_function_code(response_str)
            if not extracted_code:
                raise ParseError(f"{TRY_TAG} must contain a markdown code block (```python ... ```).")

            # Find NEXT_TEST (must appear exactly once)
            next_input_idx = None
            raw_after_next = None
            for i, ln in enumerate(lines):
                if ln.strip().startswith(NEXT_TEST_TAG):
                    if next_input_idx is not None:
                        raise ParseError(f"Only ONE {NEXT_TEST_TAG} is allowed per turn.")
                    next_input_idx = i
                    raw_after_next = ln.strip()[len(NEXT_TEST_TAG):].strip()

            if next_input_idx is None:
                raise ParseError(f"{TRY_TAG} mode requires {NEXT_TEST_TAG} ... on every turn.")

            # Disallow other action tags anywhere else in the message (prevents batching)
            for ln in lines:
                s = ln.strip()
                if s.startswith(TEST_TAG):
                    raise ParseError(f"Do not include {TEST_TAG} in {TRY_TAG} mode. Use {NEXT_TEST_TAG} only.")
                if s.startswith(SOLVE_TAG):
                    raise ParseError(f"Do not include {SOLVE_TAG} in {TRY_TAG} mode.")

            next_input_line = f"{TEST_TAG} {raw_after_next}"

            if not self.check_given_inputs(next_input_line):
                raise ParseError(f"{NEXT_TEST_TAG} values do not match the required signature/types.")

            # Allow reasoning after NEXT_TEST; we ignore it.
            return "try", {
                "code": extracted_code,
                "next_input_line": next_input_line,
            }

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

        raise ParseError(
            f"Invalid format. Must be '{TEST_TAG} ...' or '{SOLVE_TAG} ```python ...```' or '{TRY_TAG} ... {NEXT_TEST_TAG} ...'"
        )

    def _advance_game(self, player: Player, parsed_response: Tuple[str, Any]):

        action_type, content = parsed_response

        if action_type == "call":
            try:
                input_line = content["input_line"]
                output = self.run_dynamic_function(self.state.function_callable, input_line)

                # Store hypotheses (do NOT consume turns on hypothesis format issues)
                candidates = content.get("hyp_candidates")
                confidences = content.get("hyp_confidences")
                pool = set(self.state.candidate_ids or [])

                if pool:
                    if candidates is None or confidences is None:
                        self.state.hypothesis_format_error_count += 1
                    elif any(c not in pool for c in candidates):
                        self.state.hypothesis_format_error_count += 1
                    else:
                        self._record_hypotheses(candidates, confidences)

                self.set_context_for(self.guesser_player, f"{OUTPUT_TAG} {output}")
                self.log_to_self(
                    "probe results",
                    {"probe_round": self.state.probe_round, "probe_args": self.state.probe_args,
                     "probe_output": self.state.probe_output}
                )

            except Exception as e:
                self.state.runtime_error_count += 1
                self.set_context_for(self.guesser_player, f"Error executing function: {e}")

        elif action_type == "try":
            # Reject {TRY_TAG} if iterative mode not enabled (do this early)
            if not self.state.iterative_guesses_enabled:
                raise ParseError(f"{TRY_TAG} mode is not enabled for this run.")

            code = content["code"]
            next_input_line = content["next_input_line"]

            # Count {SOLVE_TAG} attempts
            self.state.guess_count += 1

            # Load hidden function source for reveal (only used on WIN)
            actual_func = self.load_function("functions", self.state.function_callable)
            actual_code = self._clean_source_for_reveal(actual_func)

            # Evaluate {SOLVE_TAG} against static tests
            is_correct, accuracy, feedback = validate_function_logic(code, self.test_cases)

            # Log accuracy (rounded)
            self.state.test_accuracy = accuracy
            self.log_to_self("test_accuracy", f"Accuracy Score: {accuracy:.3f}")

            # Hypothesis logging (same logic as "call")
            candidates = content.get("hyp_candidates")
            confidences = content.get("hyp_confidences")
            pool = set(self.state.candidate_ids or [])
            if pool:
                if candidates is None or confidences is None:
                    self.state.hypothesis_format_error_count += 1
                elif any(c not in pool for c in candidates):
                    self.state.hypothesis_format_error_count += 1
                else:
                    self._record_hypotheses(candidates, confidences)

            if is_correct:
                # WIN: end immediately, do NOT run {NEXT_TEST_TAG}
                self.state.success = True
                self.log_to_self("outcome", "Game Verdict: WIN")
                reveal_msg = f"That is correct!\nThe hidden function was:\n\n{actual_code}"
                self.log_to_self("reveal_function", reveal_msg)
                self.log_to_self("final_guess_code", f"Your {SOLVE_TAG}\n\n{code}")

                # Efficiency/slack at stop (use probe_round as-is)
                den = max(1, (self.state.max_turns - 1))
                efficiency = 1 - (self.state.probe_round - 1) / den
                self.state.test_efficiency = efficiency
                self.log_to_self("test_efficiency", f"Efficiency Score: {efficiency:.3f}")

                self.state.test_slack = (self.state.max_turns - self.state.probe_round)
                self.log_to_self("test_slack", f"Slack Score: {int(self.state.test_slack)}")
                return

            # WRONG GUESS in {TRY_TAG} mode: game continues, MUST run NEXT_TEST_TAG
            self.state.success = False
            self.state.wrong_guess_count += 1
            self.log_to_self("outcome", f"Incorrect {SOLVE_TAG} ({TRY_TAG} mode continues). {feedback}")
            self.log_to_self("final_guess_code", f"Your {SOLVE_TAG}\n\n{code}")

            # Run the mandatory NEXT_TEST_TAG probe
            try:
                output = self.run_dynamic_function(self.state.function_callable, next_input_line)
                self.set_context_for(self.guesser_player, f"{OUTPUT_TAG} {output}")
                self.log_to_self("probe results", {
                    "probe_round": self.state.probe_round,
                    "probe_args": self.state.probe_args,
                    "probe_output": self.state.probe_output
                })
            except Exception as e:
                self.state.runtime_error_count += 1
                self.set_context_for(self.guesser_player, f"Error executing function: {e}")

            # After the mandatory probe, compute efficiency/slack consistently
            den = max(1, (self.state.max_turns - 1))
            efficiency = 1 - (self.state.probe_round - 1) / den
            self.state.test_efficiency = efficiency
            self.log_to_self("test_efficiency", f"Efficiency Score: {efficiency:.3f}")

            self.state.test_slack = (self.state.max_turns - self.state.probe_round)
            self.log_to_self("test_slack", f"Slack Score: {int(self.state.test_slack)}")

        elif action_type == "solve":
            print(f"Player guessed code:\n{content}")
            self.state.guess_count += 1

            # Load actual function for reveal
            actual_func = self.load_function("functions", self.state.function_callable)
            actual_code = self._clean_source_for_reveal(actual_func)

            # Evaluate {SOLVE_TAG} against static tests
            is_correct, accuracy, feedback = validate_function_logic(content, self.test_cases)

            # --- INTERNAL CONSISTENCY (observed probe pairs) ---
            observed_cases = []
            for pair in (self.state.observed_pairs or []):
                observed_cases.append({"args": pair["args"], "expected": pair["output"]})

            if len(observed_cases) == 0:
                self.state.internal_consistency_score = 1.0
                self.state.internal_consistency_all_observed = True
                self.state.internal_consistency_violations = 0
            else:
                obs_all_correct, obs_accuracy, _ = validate_function_logic(content, observed_cases)
                self.state.internal_consistency_score = float(obs_accuracy)
                self.state.internal_consistency_all_observed = bool(obs_all_correct)
                self.state.internal_consistency_violations = int(round((1.0 - obs_accuracy) * len(observed_cases)))

            self.log_to_self(
                "internal_consistency",
                {
                    "internal_consistency_score": self.state.internal_consistency_score,
                    "internal_consistency_all_observed": self.state.internal_consistency_all_observed,
                    "internal_consistency_violations": self.state.internal_consistency_violations,
                    "observed_pair_count": len(observed_cases),
                }
            )

            self.state.test_accuracy = accuracy
            self.log_to_self("test_accuracy", f"Accuracy Score: {accuracy:.3f}")

            den = max(1, (self.state.max_turns - 1))
            efficiency = 1 - (self.state.probe_round - 1) / den
            self.state.test_efficiency = efficiency
            self.log_to_self("test_efficiency", f"Efficiency Score: {efficiency:.3f}")

            self.state.test_slack = (self.state.max_turns - self.state.probe_round)
            self.log_to_self("test_slack", f"Slack Score: {int(self.state.test_slack)}")

            if is_correct:
                print("LOGIC MATCH! Game Won.")
                self.state.success = True
                self.log_to_self("outcome", "Game Verdict: WIN")
                reveal_msg = f"That is correct!\nThe hidden function was:\n\n{actual_code}"
                self.log_to_self("reveal_function", reveal_msg)
            else:
                print(f"LOGIC MISMATCH. Accuracy: {accuracy * 100:.1f}%")
                self.state.success = False
                self.state.wrong_guess_count += 1

                if self.state.iterative_guesses_enabled:
                    # Iterative mode: do NOT end the game, do NOT reveal hidden function
                    self.log_to_self("outcome", f"Incorrect {SOLVE_TAG} (iterative mode: game continues)")
                    self.set_context_for(
                        self.guesser_player,
                        f"Guess was incorrect. {feedback}\nContinue probing with {TEST_TAG} ... or solve again with {SOLVE_TAG} ..."
                    )
                    # IMPORTANT: do not set failure=True
                else:
                    # Strict mode: wrong {SOLVE_TAG} ends the game
                    self.state.failure = True
                    self.log_to_self("outcome", "Game Verdict: LOSS")
                    reveal_msg = f"Incorrect.\n{feedback}\n\nHidden function:\n\n{actual_code}"
                    self.log_to_self("reveal_function", reveal_msg)

            self.log_to_self("final_guess_code", f"Your {SOLVE_TAG} \n\n{content}")
            self.log_to_self("total_turns_used_and_remaining", {
                "max_allowed_turns": self.state.max_turns,
                "total_turns_used": self.state.probe_round,
                "total_turns_remaining": self.state.max_turns - self.state.probe_round
            })

            self.log_to_self("parse_error_count", f"Parse error count: {self.state.parse_error_count}")
            self.log_to_self("runtime_error_count", f"Runtime error count: {self.state.runtime_error_count}")

    def _on_parse_error(self, error: GameError):
        if self.state.iterative_guesses_enabled:
            self.set_context_for(self.guesser_player,
                                 f"Game Violation: {error}\n"
                                 f"Iterative {TRY_TAG} mode is enabled. Output ONLY:\n"
                                 f"{TRY_TAG} ```python ...```\n{NEXT_TEST_TAG} ...\nHYPOTHESES: ...\nCONFIDENCE: ...\n"
                                 "No extra text.")
        else:
            self.set_context_for(self.guesser_player,
                                 f"Game Violation: {error}\n"
                                 f"Output ONLY:\n{TEST_TAG} ... (plus hypotheses)\nOR\n{SOLVE_TAG} ```python ... ``` (plus hypotheses)\n"
                                 "No extra text.")
        self.state.parse_error_count += 1
        self.success = False
        print(f"Parse Error: {error}. Consuming a turn.")

    def _does_game_proceed(self):
        if self.current_round >= self.max_turns:
            return False
        return not (self.state.aborted or self.state.failure or self.state.success)

    def compute_turn_score(self):
        return 1 if self.success else 0

    def compute_episode_score(self):
        if self.success:
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
        self.log_key("iterative_guesses_enabled", self.state.iterative_guesses_enabled)
        self.log_key("observed_pairs", self.state.observed_pairs)
        pairs = self.state.observed_pairs or []
        probe_count = len(pairs)

        # unique inputs / outputs
        unique_input_keys = set()
        unique_output_keys = set()

        for p in pairs:
            unique_input_keys.add(self._stable_key(p.get("args")))
            unique_output_keys.add(self._stable_key(p.get("output")))

        unique_input_count = len(unique_input_keys)
        unique_output_count = len(unique_output_keys)

        novelty_rate = (unique_output_count / probe_count) if probe_count else 0.0
        input_diversity = (unique_input_count / probe_count) if probe_count else 0.0

        # Optional: output entropy (bits)
        # H = -sum p(o) log2 p(o)
        output_entropy_bits = 0.0
        if probe_count:
            from collections import Counter
            import math
            out_counts = Counter(self._stable_key(p.get("output")) for p in pairs)
            for c in out_counts.values():
                p = c / probe_count
                output_entropy_bits -= p * math.log2(p)

        # Log keys (these will appear in interactions + get passed through by scorer)
        self.log_key("probe_count", probe_count)
        self.log_key("unique_input_count", unique_input_count)
        self.log_key("unique_output_count", unique_output_count)

        # Names your scorer already expects:
        self.log_key("novelty_rate", float(novelty_rate))
        self.log_key("input_diversity", float(input_diversity))

        # Optional (new): useful in analysis, keep numeric so it can go to raw.csv too
        self.log_key("output_entropy_bits", float(output_entropy_bits))

        self.log_key("internal_consistency_score", getattr(self.state, "internal_consistency_score", 0.0))
        self.log_key("internal_consistency_all_observed",
                     getattr(self.state, "internal_consistency_all_observed", False))
        self.log_key("internal_consistency_violations", getattr(self.state, "internal_consistency_violations", 0))

        self._compute_hypothesis_quality_metrics()
        self.log_to_self(
            "hypothesis_quality_summary",
            {
                "format_errors": self.state.hypothesis_format_error_count,
                "top1_hit_at_stop": self.state.hypothesis_top1_hit_at_stop,
                "top3_hit_at_stop": self.state.hypothesis_top3_hit_at_stop,
                "prob_true_at_stop": round(self.state.hypothesis_prob_true_at_stop, 3),
                "avg_neg_log_prob_true": round(self.state.hypothesis_avg_negative_log_prob_true, 3),
                "top1_switches": self.state.hypothesis_top1_switches,
            }
        )

        self.log_key("hypothesis_format_error_count", self.state.hypothesis_format_error_count)
        self.log_key("hypothesis_top1_hit_at_stop", self.state.hypothesis_top1_hit_at_stop)
        self.log_key("hypothesis_top3_hit_at_stop", self.state.hypothesis_top3_hit_at_stop)
        self.log_key("hypothesis_prob_true_at_stop", self.state.hypothesis_prob_true_at_stop)
        self.log_key("hypothesis_avg_negative_log_prob_true", self.state.hypothesis_avg_negative_log_prob_true)
        self.log_key("hypothesis_top1_switches", self.state.hypothesis_top1_switches)

        self.log_to_self(
            "probe_information_summary",
            (
                f"Probes: {probe_count} | "
                f"Unique inputs: {unique_input_count} (diversity={input_diversity:.2f}) | "
                f"Unique outputs: {unique_output_count} (novelty={novelty_rate:.2f}, entropy={output_entropy_bits:.2f} bits)"
            )
        )
        self.log_key("iterative_guesses_enabled", self.state.iterative_guesses_enabled)
        self.log_key("guess_count", self.state.guess_count)
        self.log_key("wrong_guess_count", self.state.wrong_guess_count)

        self.log_to_self(
            "guess_summary",
            {
                "iterative_mode": self.state.iterative_guesses_enabled,
                "guess_count": self.state.guess_count,
                "wrong_guess_count": self.state.wrong_guess_count,
            }
        )


class FunctionDetectiveScorer(GameScorer):
    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def compute_round_score(self, round_idx, round_events: List[Dict]) -> None:
        for event in round_events:
            if event["action"]["type"] == "player_response":
                self.log_round_score(round_idx, 'response_received', 1)

    def compute_episode_scores(self, interactions: Dict):
        # 1) Binary success
        binary_success = 1 if interactions.get(METRIC_SUCCESS, False) else 0

        # 2) Efficiency (already logged by the GameMaster)
        efficiency = float(interactions.get("efficiency", 0.0))

        # 3) Main score: efficiency * success
        score = efficiency * binary_success

        # 4) Aborts override
        if interactions.get(METRIC_ABORTED, False):
            score = np.nan

        # 5) Log benchmark score (0..1)
        self.log_episode_score(BENCH_SCORE, score)

        # Optional: log components for analysis
        self.log_episode_score("binary_success", binary_success)
        self.log_episode_score("efficiency", efficiency)


class FunctionDetectiveGameBenchmark(GameBenchmark):
    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return FunctionDetective(self.game_spec, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return FunctionDetectiveScorer(self.game_name, experiment, game_instance)
