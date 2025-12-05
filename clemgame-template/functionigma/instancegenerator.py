import os
import logging
from clemcore.clemgame import GameInstanceGenerator
import sys
from functionigma.functions import FUNCTION_REGISTRY

# Initialize logging
logger = logging.getLogger(__name__)
# Number of guesses
N_GUESSES = 5

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


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

        # 3. Iterate through the Registry and assign to experiments
        for i, function_data in enumerate(FUNCTION_REGISTRY):

            # Get the difficulty string ('easy', 'medium', 'hard')
            diff_level = function_data["difficulty"]

            # Select the correct experiment
            if diff_level in experiments:
                target_experiment = experiments[diff_level]

                # Create the instance within that experiment
                # We use 'i' as a unique ID, or you can maintain separate counters per experiment
                game_instance = self.add_game_instance(target_experiment, i)

                # 4. Populate the Instance Data
                # These keys will be available in the GameMaster via 'self.game_instance'
                game_instance["callable"] = function_data["function_name"]
                game_instance["signature"] = function_data["signature"]
                game_instance["category"] = function_data["category"]
                game_instance["difficulty"] = diff_level

                # Optional: Add a logic description if you want it in the instances file for reference
                # (You might need to add a 'description' field to your registry decorator if you want this)
                game_instance["docstring"] = function_data["callable"].__doc__

                print(f"Added {function_data['function_name']} to experiment 'python_{diff_level}'")

            else:
                logger.warning(
                    f"Function {function_data['function_name']} has unknown difficulty: {diff_level}. Skipping.")


if __name__ == '__main__':
    Functionigma().generate()
