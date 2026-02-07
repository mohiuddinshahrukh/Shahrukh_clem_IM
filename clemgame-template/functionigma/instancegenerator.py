import os
import logging
import sys
from clemcore.clemgame import GameInstanceGenerator

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from functionigma.functions import FUNCTION_REGISTRY
# Import the new generator function
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

            if diff_level in experiments:
                target_experiment = experiments[diff_level]
                game_instance = self.add_game_instance(target_experiment, i)

                # Standard Metadata
                game_instance["callable"] = function_data["function_name"]
                game_instance["signature"] = function_data["signature"]
                game_instance["category"] = function_data["category"]
                game_instance["difficulty"] = diff_level
                game_instance["docstring"] = function_data["callable"].__doc__

                # --- NEW: Generate and Save Test Cases ---
                print(f"Generating static tests for {function_data['function_name']}...")

                # Generate 20 test cases
                static_tests = create_static_test_cases(
                    function_data["callable"],
                    function_data["category"],
                    num_tests=20
                )

                # Save to JSON
                game_instance["test_cases"] = static_tests
                print(f"Saved {len(static_tests)} tests.")

            else:
                logger.warning(f"Unknown difficulty: {diff_level}")


if __name__ == '__main__':
    Functionigma().generate()
