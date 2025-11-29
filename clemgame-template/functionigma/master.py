from typing import Dict, Tuple, List, Union
from dataclasses import dataclass
import logging
import numpy as np
import importlib
import re
import sys
import os

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

        self.guesser_player = guesser(self.player_models[0])
        self.add_player(self.guesser_player, initial_context=self.experiments['initial_prompt_guesser'].replace(
            '$SIGNATURE_OF_CURRENT_FUNCTION$', self.signature))

        print(self.game_instance)
        self.state = GameState(max_turns=self.max_turns, function_signature=self.signature,
                               function_callable=self.game_instance['callable'])

    ##################

    def _advance_game(self, player: Player, parsed_response: str):
        # if self.current_round >= self.state.max_turns:
        #     self.state.aborted = True
        #     raise RuleViolationError(
        #         f"Maximum turns ({self.state.max_turns}) reached"
        #     )
        if not parsed_response:
            raise RuleViolationError

        output = self.run_dynamic_function(self.state.function_callable, parsed_response)
        print("Output:", output)


    def check_given_inputs(self, response: str) -> bool:
        pattern = r"^Input:\s*(-?\d+)\s*,\s*(-?\d+)\s*$"
        match = re.match(pattern, response)
        if not match:
            raise ValueError("Invalid input format.")
        return match is not None

    def extract_given_inputs(self, response):
        match = re.match(r"^Input:\s*(-?\d+)\s*,\s*(-?\d+)\s*$", response)

        return int(match.group(1)), int(match.group(2))

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
        # Import the function dynamically
        func_name = self.state.function_callable
        print("Attempting to load function:", func_name)
        func = self.load_function("functionigma.functions", func_name)
        print("Loaded function:", func)

        # Parse the input
        a, b = self.extract_given_inputs(text)
        print(f'A AND B: \n{a} {b}\n')

        # Run the hidden function
        return str(func(a, b))

    def _parse_response(self, player: Player, response: str) -> str:
        print("\nINSIDE PARSE RESPONSE\n")
        print(f"RESPONSE: {response}")
        response_str = str(response).strip()

        # Step 1: Check if this is a function guess
        if response_str.startswith("GUESS:"):
            print("User is submitting a function guess!")
            # Remove the tag and store the actual code
            function_code = response_str[len("GUESS:"):].strip()
            print("FUNCTION CODE: ", function_code)
            # Log and end game immediately
            self.log_to_self("player_guess_submitted", "end game")
            self.state.success = True  # mark game as ended successfully
            # Stop any further processing
            raise RuleViolationError("Function guess submitted — ending game.")

        # Step 2: Otherwise, treat as normal input query
        if not self.check_given_inputs(response_str):
            raise ParseError("Invalid input format.")
        return response_str

    def _on_parse_error(self, error: GameError):
        self.success = False

    def _does_game_proceed(self):
        return not (self.state.aborted or self.state.failure or self.state.success)

    def compute_turn_score(self):
        return 1 if self.success else 0

    def compute_episode_score(self):
        if self.success:
            return 100 / (self.current_round + 1)  # zero-based
        return 0

    def _on_after_game(self):
        if self.success:
            self.log_key(METRIC_SUCCESS, True)
        else:
            self.log_key(METRIC_SUCCESS, False)


class SomeGameScorer(GameScorer):
    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def compute_round_score(self, round_idx, round_events: List[Dict]) -> None:
        for event in round_events:
            if event["action"]["type"] == "player_response":
                self.log_round_score(round_idx, 'response_received', 1)

    def compute_episode_scores(self, interactions: Dict):
        if interactions[METRIC_SUCCESS]:
            self.log_episode_score(BENCH_SCORE, 100)
        elif interactions[METRIC_LOSE]:
            self.log_episode_score(BENCH_SCORE, 0)
        elif interactions[METRIC_ABORTED]:
            self.log_episode_score(BENCH_SCORE, np.nan)
        else:
            raise ValueError("Missing outcome value (success, failure, abort) in interactions.json")


class SomeGameBenchmark(GameBenchmark):

    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return Functionigma_GameMaster(self.game_spec, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return SomeGameScorer(self.game_name, experiment, game_instance)
