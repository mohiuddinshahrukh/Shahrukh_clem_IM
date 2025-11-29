import os
import logging
from clemcore.clemgame import GameInstanceGenerator

# Initialize logging
logger = logging.getLogger(__name__)
# Number of guesses
N_GUESSES = 7


class Functionigma(GameInstanceGenerator):
    def __init__(self):
        super().__init__(os.path.dirname(__file__))

    def on_generate(self, seed=None, **kwargs):
        self.filename = f"instances_python_easy.json"
        experiment = self.add_experiment(f"python_easy")
        experiment["max_turns"] = N_GUESSES
        experiment["initial_prompt_guesser"] = self.load_template(
            "resources/initial_prompts/initial_prompt_guesser")
        game_instance = self.add_game_instance(experiment, 0)
        game_instance["signature"] = "my_function(x:int, y:int) -> int"
        game_instance["logic_description"] = "return (x + y) % 10"
        game_instance["callable"]= "f_add_mod_10"


if __name__ == '__main__':
    Functionigma().generate()
