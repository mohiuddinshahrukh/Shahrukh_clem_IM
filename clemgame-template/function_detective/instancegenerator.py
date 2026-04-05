import os
import random
import logging
import numpy as np
from clemcore.clemgame import GameInstanceGenerator

try:
    from functions import FUNCTION_REGISTRY
    from utils import create_static_test_cases
except ModuleNotFoundError:
    from function_detective.functions import FUNCTION_REGISTRY
    from function_detective.utils import create_static_test_cases

logger = logging.getLogger(__name__)

DOMAIN_CATEGORIES = [
    "SCALAR_MATH",
    "PAIR_MATH",
    "LIST_SEQUENCE",
    "STRING",
    "LOGIC_FORMAL",
]

MAX_TURNS_BY_DIFFICULTY = {
    "easy": 5,
    "medium": 5,
    "hard": 5,
}

NUM_TESTS_BY_DIFFICULTY = {
    "easy": 20,
    "medium": 30,
    "hard": 40,
}


class FunctionDetectiveInstanceGenerator(GameInstanceGenerator):
    def __init__(self):
        super().__init__(os.path.dirname(__file__))

    def on_generate(self, seed=None, **kwargs):
        self.filename = "instances.json"

        def create_experiment(name: str, max_turns: int):
            exp = self.add_experiment(name)
            exp["max_turns"] = max_turns
            exp["guesser_initial_prompt"] = self.load_template("resources/initial_prompts/initial_guesser")
            return exp

        experiments = {}
        for diff in ["easy", "medium", "hard"]:
            for domain in DOMAIN_CATEGORIES:
                exp_name = f"{diff}_{domain.lower()}"
                experiments[(diff, domain)] = create_experiment(exp_name, MAX_TURNS_BY_DIFFICULTY[diff])

        for i, function_data in enumerate(FUNCTION_REGISTRY):
            diff_level = function_data["difficulty"]
            domain = function_data["category"]  # must be one of the 5 domain labels

            key = (diff_level, domain)
            if key not in experiments:
                logger.warning(f"Unknown bucket (difficulty, domain)={key} for {function_data['function_name']}")
                continue

            target_experiment = experiments[key]

            game_instance = self.add_game_instance(target_experiment, i)
            game_instance["callable"] = function_data["function_name"]
            game_instance["signature"] = function_data["signature"]
            game_instance["category"] = function_data["category"]

            # deterministic tests per (seed, function)
            base_seed = seed if seed is not None else 0
            case_seed = (base_seed * 10_000) + i
            random.seed(case_seed)
            np.random.seed(case_seed)

            num_tests = NUM_TESTS_BY_DIFFICULTY.get(diff_level, 20)

            static_tests = create_static_test_cases(
                function_data["callable"],
                function_data["category"],
                signature=function_data["signature"],
                difficulty=diff_level,
                num_tests=num_tests,
            )

            game_instance["test_cases"] = static_tests
            logger.info("Generated %s static tests for %s", len(static_tests), function_data["function_name"])


if __name__ == '__main__':
    FunctionDetectiveInstanceGenerator().generate()
