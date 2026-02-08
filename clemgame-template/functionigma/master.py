from typing import Dict, Tuple, List, Any
from dataclasses import dataclass, field
import logging
import numpy as np
import importlib
import inspect
import sys
import os
import json
import ast
from utils import parse_signature_with_types
from utils import extract_function_code, validate_function_logic

from clemcore.backends import Model
from clemcore.clemgame import GameSpec, GameBenchmark, GameMaster, Player, DialogueGameMaster, GameScorer, \
    GameError, ParseError
from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE, BENCH_SCORE

# --- FINAL ROBUST FIX: MONKEY PATCH FOR JSON SERIALIZATION ---
_original_json_default = json.JSONEncoder.default


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
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@dataclass
class GameState:
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


class Guesser(Player):
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


class FunctionigmaGameMaster(DialogueGameMaster):
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
        # parse signature to get param names/types
        self.param_names, self.param_types, self.return_type = parse_signature_with_types(self.signature)
        self.num_params = len(self.param_names)

        # choose given category if present, otherwise -> (fallback to MATH)
        self.category = game_instance.get("category", "MATH")
        candidate_ids = game_instance.get("candidate_ids", [])

        # parameter list string for prompt
        param_list_str = ", ".join(
            f"{n}: {t}" for n, t in zip(self.param_names, self.param_types)) if self.param_names else "none"
        variables = ", ".join(f"<value{i + 1}>" for i in range(self.num_params))

        prompt = self.experiments['initial_prompt_guesser']
        prompt = prompt.replace('$SIGNATURE_OF_CURRENT_FUNCTION$', self.signature)
        prompt = prompt.replace('$NUM_PARAMS$', str(self.num_params))
        prompt = prompt.replace('$PARAM_LIST$', param_list_str)
        prompt = prompt.replace('$CATEGORY$', str(self.category))
        prompt = prompt.replace('$MAX_TURNS$', str(self.max_turns))
        prompt = prompt.replace('$VARIABLES$', variables)
        if candidate_ids:
            prompt += "\n\nCANDIDATE FUNCTION IDS (use only these in hypotheses):\n"
            prompt += "\n".join(f"- {cid}" for cid in candidate_ids)
            prompt += (
                "\n\nOn EVERY turn, output one of the following formats.\n"
                "Format A (probe):\n"
                "INPUT: <args>\n"
                "HYPOTHESES:\n"
                "- candidate: <candidate_id>\n"
                "- candidate: <candidate_id>\n"
                "CONFIDENCE:\n"
                "- <float>\n"
                "- <float>\n"
                "\nConfidences must sum to 1.\n"
                "\nFormat B (final guess):\n"
                "GUESS: ```python\n<code>\n```\n"
            )

        # add player with initial context
        self.guesser_player = Guesser(self.player_models[0])
        self.add_player(self.guesser_player, initial_context=prompt)
        self.state = GameState(
            max_turns=self.max_turns,
            function_signature=self.signature,
            function_callable=self.game_instance['callable'],
        )
        self.state.candidate_ids = candidate_ids

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
        # Accept only responses starting with "INPUT:"
        if not isinstance(response, str):
            return False
        response = response.strip()
        raw = response[len("INPUT:"):].strip()
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
        raw = response.strip()[len("INPUT:"):].strip()
        if self.num_params == 0:
            return []
        # Use ast.literal_eval to safely parse values
        if self.num_params == 1:
            val = ast.literal_eval(raw)
            return [val]
        parsed = ast.literal_eval(f"({raw})")
        return list(parsed)

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
        func = self.load_function("functionigma.functions", func_name)
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

        if isinstance(result, str):
            return result
        return str(result)

    def _parse_response(self, player: Player, response: str) -> Tuple[str, Any]:
        print(f"PROBE: {response}")
        response_str = str(response).strip()

        # --- LOGIC TO HANDLE GUESS ---
        if response_str.startswith("GUESS:"):
            # Use the utility function to get clean code
            extracted_code = extract_function_code(response_str)

            if not extracted_code:
                # Fallback: If regex fails (e.g. user forgot backticks),
                # you might want to take everything after "GUESS:"
                # or raise a ParseError.
                # raising ParseError forces the model to retry with correct formatting.
                raise ParseError("Guess must contain a markdown code block (```python ... ```).")

            # Return tuple: (action, content)
            return "guess", extracted_code

        # --- LOGIC TO HANDLE INPUTS ---
        # --- LOGIC TO HANDLE INPUTS ---
        if response_str.startswith("INPUT:"):
            # Allow multi-line: first line is INPUT, remaining lines are hypothesis block
            lines = response_str.splitlines()
            input_line = lines[0].strip()
            rest = "\n".join(lines[1:]) if len(lines) > 1 else ""

            if self.check_given_inputs(input_line):
                hyp_candidates, hyp_confidences = self._parse_hypothesis_block(rest)
                return "call", {
                    "input_line": input_line,
                    "hyp_candidates": hyp_candidates,
                    "hyp_confidences": hyp_confidences,
                }

        # --- FAILURE ---
        raise ParseError("Invalid format. Must be 'INPUT: x, y' or 'GUESS: ```python ... ```'")

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

                if not pool:
                    # if no candidate pool (older instances), skip hypothesis tracking
                    pass
                elif candidates is None or confidences is None:
                    self.state.hypothesis_format_error_count += 1
                elif any(c not in pool for c in candidates):
                    self.state.hypothesis_format_error_count += 1
                else:
                    self._record_hypotheses(candidates, confidences)

                self.set_context_for(self.guesser_player, f"Function Output: {output}")
                self.log_to_self(
                    "probe results",
                    {"probe_round": self.state.probe_round, "probe_args": self.state.probe_args,
                     "probe_output": self.state.probe_output}
                )

            except Exception as e:
                self.set_context_for(self.guesser_player, f"Error executing function: {e}")


        elif action_type == "guess":
            print(f"Player guessed code:\n{content}")

            # Note: We no longer need to load actual_func to generate tests,
            # but we still load it to get the source code for the "reveal" message.
            actual_func = self.load_function("functionigma.functions", self.state.function_callable)

            try:
                source_lines = inspect.getsource(actual_func).splitlines()
                clean_lines = [line for line in source_lines if not line.strip().startswith("@")]
                actual_code = "\n".join(clean_lines).strip()
            except OSError:
                actual_code = "# Source code unavailable"

            # --- UPDATED VALIDATION ---
            # Pass the static test_cases instead of the function object
            is_correct, accuracy, feedback = validate_function_logic(content, self.test_cases)

            # --- INTERNAL CONSISTENCY (observed probe pairs) ---
            observed_cases = []
            for pair in (self.state.observed_pairs or []):
                # Your observed_pairs currently store "args" and "output" keys
                observed_cases.append({
                    "args": pair["args"],
                    "expected": pair["output"],
                })

            if len(observed_cases) == 0:
                # No probes made → vacuously consistent
                self.state.internal_consistency_score = 1.0
                self.state.internal_consistency_all_observed = True
                self.state.internal_consistency_violations = 0
            else:
                obs_all_correct, obs_accuracy, _ = validate_function_logic(content, observed_cases)
                self.state.internal_consistency_score = float(obs_accuracy)
                self.state.internal_consistency_all_observed = bool(obs_all_correct)
                # approximate violation count from accuracy
                self.state.internal_consistency_violations = int(round((1.0 - obs_accuracy) * len(observed_cases)))

            # Optional: show in transcript for debugging
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
            self.log_to_self("test_accuracy", f"Accuracy Score: {accuracy}")

            den = max(1, (self.state.max_turns - 1))
            efficiency = 1 - (self.state.probe_round - 1) / den

            self.state.test_efficiency = efficiency
            self.log_to_self("test_efficiency", f"Efficiency Score: {self.state.test_efficiency}")

            self.state.test_slack = (self.state.max_turns - self.state.probe_round)
            self.log_to_self("test_slack", f"Slack Score: {self.state.test_slack}")

            if is_correct:
                print("LOGIC MATCH! Game Won.")
                self.state.success = True
                self.log_to_self("outcome", f"Game Verdict: WIN")
                reveal_msg = f"That is correct! \nThe hidden function was:\n\n{actual_code}"
                self.log_to_self("reveal_function", reveal_msg)
            else:
                print(f"LOGIC MISMATCH. Accuracy: {accuracy * 100:.1f}%")
                self.state.success = False
                self.state.failure = True
                self.log_to_self("outcome", f"Game Verdict: LOSS")
                reveal_msg = f"Incorrect\n. {feedback}\n\nHidden function:\n\n{actual_code}"
                self.log_to_self("reveal_function", reveal_msg)

            self.log_to_self("final_guess_code", f"Your guess: \n\n{content}")
            self.log_to_self("total_turns_used_and_remaining", {
                "max_allowed_turns": self.state.max_turns, "total_turns_used": self.state.probe_round,
                "total_turns_remaining": self.state.max_turns - self.state.probe_round})

            self.log_to_self("parse_error_count", f"Parse error count: {self.state.parse_error_count}")
            self.log_to_self("runtime_error_count", f"Runtime error count: {self.state.runtime_error_count}")

            # self.log_to_self("observed_pairs", f"Observed pairs: {self.state.observed_pairs}")

    def _on_parse_error(self, error: GameError):
        self.state.parse_error_count += 1
        self.success = False
        print(f"Parse Error: {error}. Consuming a turn.")
        self.set_context_for(self.guesser_player,
                             f"Game Violation: {error}\n"
                             "Please output ONLY 'INPUT: ...' or 'GUESS: ...' without preceding text.")

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
                "prob_true_at_stop": self.state.hypothesis_prob_true_at_stop,
                "avg_neg_log_prob_true": self.state.hypothesis_avg_negative_log_prob_true,
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


class SomeGameScorer(GameScorer):
    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def compute_round_score(self, round_idx, round_events: List[Dict]) -> None:
        for event in round_events:
            if event["action"]["type"] == "player_response":
                self.log_round_score(round_idx, 'response_received', 1)

    def compute_episode_scores(self, interactions: Dict):
        # --- STRICT SCORING LOGIC ---

        # 1. Determine Success (Binary)
        # METRIC_SUCCESS is only True if validate_function_logic returned True (100% pass)
        is_success = interactions.get(METRIC_SUCCESS, False)

        # 2. Calculate Bench Score
        score = 0
        if is_success:
            score = 100

        # 3. Handle Aborts (Crashes)
        if interactions.get(METRIC_ABORTED, False):
            score = np.nan

        # 4. Log the Main Score
        self.log_episode_score(BENCH_SCORE, score)

        # 5. Secondary Metric: Log partial accuracy just for visibility in CSVs
        # (This does not affect the leaderboard ranking)
        if "accuracy" in interactions:
            self.log_episode_score("accuracy", interactions["accuracy"] * 100)
        passthrough_keys = [
            "efficiency",
            "slack",
            "turns_used",
            "max_turns",
            "parse_error_count",
            "runtime_error_count",
            "probe_count",
            "novelty_rate",
            "input_diversity",
            "internal_consistency_score",
            "internal_consistency_all_observed",
            "internal_consistency_violations",
            "output_entropy_bits",

        ]
        for k in passthrough_keys:
            if k in interactions:
                self.log_episode_score(k, interactions[k])

        hypothesis_keys = [
            "hypothesis_format_error_count",
            "hypothesis_top1_hit_at_stop",
            "hypothesis_top3_hit_at_stop",
            "hypothesis_prob_true_at_stop",
            "hypothesis_avg_negative_log_prob_true",
            "hypothesis_top1_switches",
        ]
        for k in hypothesis_keys:
            if k in interactions:
                self.log_episode_score(k, interactions[k])


class SomeGameBenchmark(GameBenchmark):
    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return FunctionigmaGameMaster(self.game_spec, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return SomeGameScorer(self.game_name, experiment, game_instance)
