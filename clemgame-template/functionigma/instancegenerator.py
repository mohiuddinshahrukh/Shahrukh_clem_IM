import os
import logging
import sys
import random
import numpy as np
from clemcore.clemgame import GameInstanceGenerator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from functionigma.functions import FUNCTION_REGISTRY
from functionigma.utils import create_static_test_cases

logger = logging.getLogger(__name__)
N_GUESSES = 5


class Functionigma(GameInstanceGenerator):
    def __init__(self):
        super().__init__(os.path.dirname(__file__))

    def on_generate(self, seed=None, **kwargs):
        self.filename = f"instances.json"
        experiments = {}

        def create_experiment(name):
            exp = self.add_experiment(name)
            exp["max_turns"] = N_GUESSES
            exp["initial_prompt_guesser"] = self.load_template(
                "resources/initial_prompts/initial_prompt_guesser"
            )
            return exp

        experiments["easy"] = create_experiment("python_easy")
        experiments["medium"] = create_experiment("python_medium")
        experiments["hard"] = create_experiment("python_hard")

        for i, function_data in enumerate(FUNCTION_REGISTRY):
            diff_level = function_data["difficulty"]

            if diff_level not in experiments:
                logger.warning(f"Unknown difficulty: {diff_level}")
                continue

            target_experiment = experiments[diff_level]
            game_instance = self.add_game_instance(target_experiment, i)

            game_instance["callable"] = function_data["function_name"]
            game_instance["signature"] = function_data["signature"]
            game_instance["category"] = function_data["category"]
            game_instance["difficulty"] = diff_level
            game_instance["docstring"] = function_data["callable"].__doc__

            # deterministic tests per (seed, function)
            base_seed = seed if seed is not None else 0
            case_seed = (base_seed * 10_000) + i
            random.seed(case_seed)
            np.random.seed(case_seed)

            num_tests = {"easy": 20, "medium": 30, "hard": 40}.get(diff_level, 20)

            print(f"Generating static tests for {function_data['function_name']}...")

            static_tests = create_static_test_cases(
                function_data["callable"],
                function_data["category"],
                signature=function_data["signature"],
                difficulty=diff_level,
                num_tests=num_tests,
            )

            game_instance["test_cases"] = static_tests
            print(f"Saved {len(static_tests)} tests.")


if __name__ == '__main__':
    Functionigma().generate()
