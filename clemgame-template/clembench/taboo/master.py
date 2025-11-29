from dataclasses import dataclass
from typing import Dict, List
import logging
import numpy as np

from clemcore.backends import Model
from clemcore.clemgame import GameSpec, GameMaster, GameBenchmark, Player, DialogueGameMaster, GameScorer
from clemcore.clemgame import ParseError, RuleViolationError, GameError
from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE, BENCH_SCORE
from clemcore.utils import string_utils

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

nltk.download('stopwords', quiet=True)
EN_STOPWORDS = stopwords.words('english')

EN_STEMMER = SnowballStemmer("english")

logger = logging.getLogger(__name__)

GUESS_PREFIX = "GUESS:"
CLUE_PREFIX = "CLUE:"


class WordGuesser(Player):

    def __init__(self, model: Model):
        super().__init__(model)
        self._custom_responses = ["Apple", "Banana", "Cherry"]

    def _custom_response(self, messages):
        word = self._custom_responses.pop(0)
        return f'{GUESS_PREFIX} {word}'


class WordDescriber(Player):

    def __init__(self, model: Model):
        super().__init__(model)
        self._custom_responses = ["(1) My first clue is ...", "(2) My second clue is ...", "(3) My third clue is ..."]

    def _custom_response(self, messages):
        clue = self._custom_responses.pop(0)
        return f"{CLUE_PREFIX} {clue}"


@dataclass
class GameState:
    target_word: str
    related_words: List[str]
    max_rounds: int
    describer_initial_prompt: str
    guesser_initial_prompt: str
    success: bool = False
    failure: bool = False
    aborted: bool = False
    clue_error: str = None
    last_clue: str = None
    last_guess: str = None


def check_clue(response: str, state: GameState, stemmer=EN_STEMMER):
    clue_words = string_utils.remove_punctuation(response).lower().split(" ")
    clue_words = [clue_word for clue_word in clue_words if clue_word not in EN_STOPWORDS]
    clue_word_stems = [stemmer.stem(clue_word) for clue_word in clue_words]
    target_word_stem = stemmer.stem(state.target_word)
    related_word_stems = [stemmer.stem(related_word) for related_word in state.related_words]

    for clue_word, clue_word_stem in zip(clue_words, clue_word_stems):  # raise first appearing exception
        if target_word_stem == clue_word_stem:
            reason = f"Target word '{state.target_word}' (stem={target_word_stem}) " \
                     f"is similar to clue word '{clue_word}' (stem={clue_word_stem})"
            raise RuleViolationError(reason, response)
        for related_word, related_word_stem in zip(state.related_words, related_word_stems):
            if related_word_stem == clue_word_stem:
                reason = f"Related word '{related_word}' (stem={related_word_stem}) " \
                         f"is similar to clue word '{clue_word}' (stem={clue_word_stem})"
                raise RuleViolationError(reason, response)


class Taboo(DialogueGameMaster):
    """
    This class implements a taboo game in which player A (the WordDescriber) is describing a
    target word that player B (the WordGuesser) needs to guess. Player A cannot use the target
    word or related words in their explanation. Morphology is checked in check_clue().
    """

    def __init__(self, game_spec: GameSpec, experiment: Dict, player_models: List[Model]):
        super().__init__(game_spec, experiment, player_models)

    def _on_setup(self, **game_instance):
        describer_initial_prompt = self.experiment["describer_initial_prompt"]
        describer_initial_prompt = describer_initial_prompt.replace("$TARGET_WORD$", game_instance["target_word"])
        rel_words = (f"- {game_instance['related_word'][0]}\n"
                     f"- {game_instance['related_word'][1]}\n"
                     f"- {game_instance['related_word'][2]}")
        describer_initial_prompt = describer_initial_prompt.replace("$REL_WORD$", rel_words)
        describer_initial_prompt = describer_initial_prompt.replace("$N$", str(self.experiment["max_turns"]))

        guesser_initial_prompt = self.experiment["guesser_initial_prompt"]
        guesser_initial_prompt = guesser_initial_prompt.replace("$N$", str(self.experiment["max_turns"]))

        self.describer = WordDescriber(self.player_models[0])
        self.guesser = WordGuesser(self.player_models[1])

        self.add_player(self.describer, initial_context=describer_initial_prompt)
        self.add_player(self.guesser, initial_prompt=guesser_initial_prompt)

        # arguments in same order as above
        self.state = GameState(game_instance["target_word"],
                               game_instance["related_word"],
                               self.experiment["max_turns"],
                               describer_initial_prompt,
                               guesser_initial_prompt)

    def _does_game_proceed(self):
        return not (self.state.aborted or self.state.failure or self.state.success)

    def _parse_response(self, player: Player, response: str) -> str:
        prefix = None
        if player == self.guesser:
            prefix = GUESS_PREFIX
        if player == self.describer:
            prefix = CLUE_PREFIX
        assert prefix is not None, f"Communication protocol not specified for player {player}"

        # validate communication protocol (this could also be done for each player individually)
        if not response.startswith(prefix):
            raise ParseError(f"response must start with {prefix}", response)
        self.log_to_self("valid response", "continue")

        # parse response content (here only remove the prefix)
        return response.replace(prefix, "").strip()

    def _on_parse_error(self, error: ParseError):
        self.log_to_self("invalid format", "abort game")
        self.state.aborted = True

    def _advance_game(self, player: Player, parsed_response: str):
        if player == self.describer:
            # validate game rules
            check_clue(parsed_response, self.state)  # throws RuleViolationError
            self.log_to_self("valid clue", parsed_response)
            # transition game state
            self.state.last_clue = parsed_response
            self.set_context_for(self.guesser, f"{CLUE_PREFIX} {parsed_response}")

        if player == self.guesser:
            # validate game rules
            if len(parsed_response.split(" ")) > 1:
                raise RuleViolationError("guess has more than one word", parsed_response)
            self.log_to_self("valid guess", parsed_response)
            # transition game state
            self.state.last_guess = parsed_response
            self.set_context_for(self.describer, f"{GUESS_PREFIX} {self.state.last_guess}")  # ignored if success

            # check game end conditions
            if self.state.last_guess.lower() == self.state.target_word:
                self.log_to_self("correct guess", "end game")
                self.state.success = True
            elif self.current_round == self.state.max_rounds - 1:  # zero-based
                raise RuleViolationError(f"max rounds ({self.state.max_rounds}) reached")

    def _on_game_error(self, error: GameError):
        # note: we could also introduce more concrete subclasses e.g. InvalidClueError and handle them here individually
        self.log_to_self(error.reason, "failed game")
        self.state.clue_error = error.reason
        self.state.failure = True

    def compute_turn_score(self):
        return 1 if self.state.success else 0

    def compute_episode_score(self):
        if self.state.success:
            return 100 / (self.current_round + 1)  # zero-based
        return 0

    def _on_after_game(self):
        self.log_key(METRIC_ABORTED, int(self.state.aborted))
        self.log_key(METRIC_LOSE, int(self.state.failure))
        self.log_key(METRIC_SUCCESS, int(self.state.success))


class TabooScorer(GameScorer):
    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def compute_round_score(self, round_idx, round_events: List[Dict]) -> None:
        guesser_won = False
        for event in round_events:
            if event["action"]["type"] == "correct guess":
                guesser_won = True
        self.log_round_score(round_idx, 'Accuracy', 1 if guesser_won else 0)

    def compute_episode_scores(self, interactions: Dict):
        num_rounds = len(interactions["turns"])
        if interactions[METRIC_SUCCESS]:
            self.log_episode_score(BENCH_SCORE, 100 / num_rounds)
        elif interactions[METRIC_LOSE]:
            self.log_episode_score(BENCH_SCORE, 0)
        elif interactions[METRIC_ABORTED]:
            self.log_episode_score(BENCH_SCORE, np.nan)
        else:
            raise ValueError("Missing outcome value (success, failure, abort) in interactions.json")


class TabooGameBenchmark(GameBenchmark):

    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return Taboo(self.game_spec, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return TabooScorer(self.game_name, experiment, game_instance)
