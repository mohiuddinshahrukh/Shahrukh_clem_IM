from typing import Dict, Tuple, List, Union
from dataclasses import dataclass
import logging
import numpy as np
import importlib
import inspect
import sys
import os
import json
import ast
from utils import parse_signature_with_types
from utils import extract_function_code, validate_function_logic, generate_random_value

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

from clemcore.backends import Model
from clemcore.clemgame import GameSpec, GameBenchmark, GameMaster, Player, DialogueGameMaster, GameScorer, \
    GameError, ParseError, RuleViolationError
from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE, BENCH_SCORE

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
    test_accuracy: float = 0.0


class guesser(Player):
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


class Functionigma_GameMaster(DialogueGameMaster):
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
        # parse signature to get param names/types
        self.param_names, self.param_types, self.return_type = parse_signature_with_types(self.signature)
        self.num_params = len(self.param_names)

        # choose given category if present, otherwise -> (fallback to MATH)
        self.category = game_instance.get("category", "MATH")

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

        # add player with initial context
        self.guesser_player = guesser(self.player_models[0])
        self.add_player(self.guesser_player, initial_context=prompt)
        self.state = GameState(max_turns=self.max_turns, function_signature=self.signature,
                               function_callable=self.game_instance['callable'])

    ##################

    import ast

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
        if isinstance(result, str):
            return result
        return str(result)

    def _parse_response(self, player: Player, response: str) -> Tuple[str, str]:
        print(f"RESPONSE: {response}")
        response_str = str(response).strip()

        # --- LOGIC TO HANDLE GUESS ---
        if response_str.startswith("GUESS:"):
            print("User is submitting a function guess!")

            # Use the utility function to get clean code
            extracted_code = extract_function_code(response_str)

            if not extracted_code:
                # Fallback: If regex fails (e.g. user forgot backticks),
                # you might want to take everything after "GUESS:"
                # or raise a ParseError.
                # raising ParseError forces the model to retry with correct formatting.
                raise ParseError("Guess must contain a markdown code block (```python ... ```).")

            print("FUNCTION CODE FOUND:\n", extracted_code)

            # Return tuple: (action, content)
            return "guess", extracted_code

        # --- LOGIC TO HANDLE INPUTS ---
        if response_str.startswith("INPUT:"):
            if self.check_given_inputs(response_str):
                return "call", response_str
        else:
            # --- FAILURE ---
            raise ParseError("Invalid format. Must be 'INPUT: x, y' or 'GUESS: ```python ... ```'")

    def _advance_game(self, player: Player, parsed_response: Tuple[str, str]):
        action_type, content = parsed_response

        if action_type == "call":
            try:
                output = self.run_dynamic_function(self.state.function_callable, content)
                self.set_context_for(self.guesser_player, f"Function Output: {output}")
            except Exception as e:
                self.set_context_for(self.guesser_player, f"Error executing function: {e}")

        elif action_type == "guess":
            print(f"Player guessed code:\n{content}")
            actual_func = self.load_function("functionigma.functions", self.state.function_callable)

            try:
                source_lines = inspect.getsource(actual_func).splitlines()
                clean_lines = [line for line in source_lines if not line.strip().startswith("@")]
                actual_code = "\n".join(clean_lines).strip()
            except OSError:
                actual_code = "# Source code unavailable"

            is_correct, accuracy, feedback = validate_function_logic(content, actual_func)

            # CHANGE: Save accuracy to state so we can log it later
            self.state.test_accuracy = accuracy
            self.log_to_self("test_accuracy", accuracy)

            if is_correct:
                print("LOGIC MATCH! Game Won.")
                self.state.success = True
                self.log_to_self("outcome", "win")
                reveal_msg = f"That is correct! \nThe hidden function was:\n\n{actual_code}"
                self.log_to_self("reveal_function", reveal_msg)
            else:
                print(f"LOGIC MISMATCH. Accuracy: {accuracy * 100:.1f}%")
                self.state.success = False
                self.state.failure = True
                self.log_to_self("outcome", "loss")
                reveal_msg = f"That is incorrect. {feedback}\n\nThe hidden function was:\n\n{actual_code}"
                self.log_to_self("reveal_function", reveal_msg)

            self.log_to_self("final_guess_code", content)

    def _on_parse_error(self, error: GameError):
        self.success = False
        print(f"Parse Error: {error}. Consuming a turn.")
        self.set_context_for(self.guesser_player,
                             f"Game Violation: {error}\n"
                             "Please output ONLY 'INPUT: ...' or 'GUESS: ...' without preceding text.")

    def _does_game_proceed(self):
        # 1. Stop if we reached the maximum number of turns
        if self.current_round >= self.max_turns:
            return False

        # 2. Stop if the game is already over (Win/Loss/Abort)
        return not (self.state.aborted or self.state.failure or self.state.success)

    def compute_turn_score(self):
        return 1 if self.success else 0

    def compute_episode_score(self):
        if self.success:
            return 100 / (self.current_round + 1)  # zero-based
        return 0

    def _on_after_game(self):
        # Required: Log all three outcome metrics
        self.log_key(METRIC_ABORTED, self.state.aborted)
        self.log_key(METRIC_LOSE, self.state.failure)
        self.log_key(METRIC_SUCCESS, self.state.success)

        # --- FIX 2: MATCH THE SCORER KEY ---
        # Retrieve from state (defaults to 0.0 if missing)
        accuracy = getattr(self.state, "test_accuracy", 0.0)

        # Log as "accuracy" because SomeGameScorer looks for: if "accuracy" in interactions:
        self.log_key("accuracy", accuracy)


class SomeGameScorer(GameScorer):
    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def compute_round_score(self, round_idx, round_events: List[Dict]) -> None:
        for event in round_events:
            if event["action"]["type"] == "player_response":
                self.log_round_score(round_idx, 'response_received', 1)

    def compute_episode_scores(self, interactions: Dict):
        # 1. Calculate Accuracy (0 to 100) from the log
        #    If no accuracy logged (e.g. crash), default to 0
        acc_score = 0.0
        if "accuracy" in interactions:
            acc_score = interactions["accuracy"] * 100

        # 2. Log it as a custom metric (visible in raw.csv)
        self.log_episode_score("accuracy", acc_score)

        # 3. SET MAIN BENCH SCORE TO ACCURACY
        #    This ensures 'accuracy' appears as the "Quality Score" in results.csv/html
        #    If the game was aborted (crash), we usually log NaN,
        #    but if it just failed tests, we log the partial accuracy.
        if interactions[METRIC_ABORTED]:
            self.log_episode_score(BENCH_SCORE, np.nan)
        else:
            self.log_episode_score(BENCH_SCORE, acc_score)


class SomeGameBenchmark(GameBenchmark):

    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return Functionigma_GameMaster(self.game_spec, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return SomeGameScorer(self.game_name, experiment, game_instance)
