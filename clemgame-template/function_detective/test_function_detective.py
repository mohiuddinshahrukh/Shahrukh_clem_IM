import unittest
import os
import sys

try:
    from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, BENCH_SCORE
except ModuleNotFoundError:
    METRIC_ABORTED = "Aborted"
    METRIC_SUCCESS = "Success"
    BENCH_SCORE = "BenchScore"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from function_detective.utils import extract_function_code, parse_signature_with_types, is_sandbox_failure

try:
    from function_detective.master import (
        FunctionDetective,
        FunctionGuesser,
        FunctionDetectiveGameState,
        FunctionDetectiveScorer,
    )
    MASTER_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    FunctionDetective = None
    FunctionGuesser = None
    FunctionDetectiveGameState = None
    FunctionDetectiveScorer = None
    MASTER_IMPORT_ERROR = exc


class FunctionDetectiveTestCase(unittest.TestCase):

    def test_public_game_classes_exist(self):
        if MASTER_IMPORT_ERROR is not None:
            self.skipTest(f"master.py dependencies unavailable: {MASTER_IMPORT_ERROR}")
        self.assertTrue(FunctionDetective)
        self.assertTrue(FunctionGuesser)
        self.assertTrue(FunctionDetectiveGameState)

    def test_extract_function_code(self):
        response = """SOLVE: ```python
def solution(x):
    return abs(x)
```"""
        self.assertEqual(extract_function_code(response), "def solution(x):\n    return abs(x)")

    def test_parse_signature_with_types(self):
        names, types, return_type = parse_signature_with_types("(x: int, y: int) -> int")
        self.assertEqual(names, ["x", "y"])
        self.assertEqual(types, ["int", "int"])
        self.assertEqual(return_type, "int")

    def test_quality_score_uses_0_to_100_scale(self):
        if MASTER_IMPORT_ERROR is not None:
            self.skipTest(f"master.py dependencies unavailable: {MASTER_IMPORT_ERROR}")

        scorer = FunctionDetectiveScorer("function_detective", {}, {})
        scorer.compute_episode_scores({
            METRIC_SUCCESS: True,
            METRIC_ABORTED: False,
            "efficiency": 0.75,
        })

        episode_scores = scorer.scores["episode scores"]
        self.assertEqual(episode_scores[BENCH_SCORE], 75.0)
        self.assertEqual(episode_scores["quality_score"], 75.0)
        self.assertEqual(episode_scores["efficiency"], 75.0)

    def test_detects_sandbox_failure_feedback(self):
        self.assertTrue(
            is_sandbox_failure(
                "Sandbox Failure.\nStderr: 'failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine'"
            )
        )
        self.assertTrue(
            is_sandbox_failure("Docker is not running or not reachable. Start Docker Desktop and try again.")
        )
        self.assertFalse(is_sandbox_failure("Failed Cases:\nInput: [1, 2] | Expected: 3 | Got: 4"))


if __name__ == '__main__':
    unittest.main()
