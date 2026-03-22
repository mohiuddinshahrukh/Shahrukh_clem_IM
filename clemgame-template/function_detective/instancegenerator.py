import os
import logging
import sys
import random
import numpy as np
from clemcore.clemgame import GameInstanceGenerator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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


class Functionigma(GameInstanceGenerator):
    def __init__(self):
        super().__init__(os.path.dirname(__file__))

    def on_generate(self, seed=None, **kwargs):
        self.filename = "instances.json"
        experiments = {}

        def create_experiment(name: str, max_turns: int):
            exp = self.add_experiment(name)
            exp["max_turns"] = max_turns
            exp["initial_prompt_guesser"] = self.load_template(
                "resources/initial_prompts/initial_prompt_guesser"
            )
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
            game_instance["difficulty"] = diff_level
            game_instance["hint"] = function_data.get("hint", "")
            # Candidate pool for hypotheses: same signature + same category + same difficulty
            # Base pool: same signature + same domain category (ignore difficulty)
            # Base pool: same signature + same domain (ignore difficulty)
            base_pool = [
                fd["function_name"]
                for fd in FUNCTION_REGISTRY
                if fd["signature"] == function_data["signature"]
                   and fd["category"] == domain
            ]

            pool = set(base_pool) | set(function_data.get("confusers", []))
            pool.add(function_data["function_name"])

            MAX_POOL = 12
            pool = sorted(pool)
            if len(pool) > MAX_POOL:
                pool = sorted(random.sample(pool, MAX_POOL))

            game_instance["candidate_ids"] = pool
            print(
                f"Candidate pool size for {function_data['function_name']}: "
                f"{len(game_instance['candidate_ids'])}"
            )

            game_instance["docstring"] = function_data["callable"].__doc__

            # deterministic tests per (seed, function)
            base_seed = seed if seed is not None else 0
            case_seed = (base_seed * 10_000) + i
            random.seed(case_seed)
            np.random.seed(case_seed)

            num_tests = NUM_TESTS_BY_DIFFICULTY.get(diff_level, 20)

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
