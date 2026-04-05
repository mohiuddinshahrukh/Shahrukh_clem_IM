import unittest
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from function_detective.utils import extract_function_code, parse_signature_with_types

try:
    from function_detective.master import FunctionDetective, FunctionGuesser, FunctionDetectiveGameState
    MASTER_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    FunctionDetective = None
    FunctionGuesser = None
    FunctionDetectiveGameState = None
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


if __name__ == '__main__':
    unittest.main()
