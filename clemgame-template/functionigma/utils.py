import re
import random
import inspect
import string
from typing import Tuple

from typing import Any, Callable, List, Dict, get_origin, get_args, Optional


def generate_random_value(annotation: Any, category: str = None) -> Any:
    """
    Generates a random value based on the type hint and the problem category.
    Handles nested types like List[int].
    """

    # --- 1. Handle Lists (Recursion) ---
    # Check if the annotation is a generic List (e.g., List[int])
    if get_origin(annotation) is list:
        # Get the inner type (e.g., int from List[int])
        args = get_args(annotation)
        inner_type = args[0] if args else int

        # Generate a list of random length (0 to 10 items)
        return [generate_random_value(inner_type, category) for _ in range(random.randint(0, 10))]

    # --- 2. Handle Integers ---
    if annotation is int:
        edge_cases = [0, 1, -1, 10, -10]

        # Category-specific tweaks
        if category == "math":
            # Add larger numbers for math problems to test overflow/logic
            edge_cases.extend([100, -100, 99, 1000])
        elif category == "bitwise":
            # Add powers of 2 for bitwise logic
            edge_cases.extend([2, 4, 8, 16, 32, 255])

        return random.choice(edge_cases + [random.randint(-50, 50)])

    # --- 3. Handle Floats ---
    elif annotation is float:
        return random.choice([0.0, 1.0, -1.0, 0.5, random.uniform(-100.0, 100.0)])

    # --- 4. Handle Strings ---
    elif annotation is str:
        special_chars = "!@#$%^&*"
        letters = string.ascii_lowercase

        # Category-specific tweaks
        if category == "string":
            # Test palindromes, spaces, repetition
            candidates = [
                "", " ", "a", "ab", "aba",  # Palindrome
                "   ",  # Whitespace
                "hello world",
                "12345"  # Digits as string
            ]
            return random.choice(candidates)

        # Default random string
        random_word = ''.join(random.choice(letters) for _ in range(random.randint(1, 8)))
        return random.choice(["", "a", " ", random_word])

    # --- 5. Default Fallback ---
    # If no type hint provided, assume Int
    return random.randint(-50, 50)


def validate_function_logic(guessed_code: str, actual_callable: Callable, category: str = "general",
                            num_tests: int = 50) -> Tuple[bool, float, str]:
    """
    Runs untrusted guessed code inside an ephemeral Docker container.
    Returns: (success: bool, accuracy: float, feedback: str)
    """
    import subprocess
    import json
    import random

    random.seed(42)  # deterministic test cases

    # --- STEP 1: Inspect target function ---
    try:
        sig = inspect.signature(actual_callable)
        params = sig.parameters
    except ValueError:
        return False, 0.0, "Error: Could not inspect function signature."

    # --- STEP 2: Generate test cases ---
    test_cases = []
    for _ in range(num_tests):
        args = []
        for _, param in params.items():
            val = generate_random_value(param.annotation, category)
            args.append(val)
        test_cases.append(args)

    # --- STEP 3: Compute ground truth outputs locally ---
    expected_outputs = []
    for args in test_cases:
        try:
            expected_outputs.append(actual_callable(*args))
        except Exception:
            expected_outputs.append("__SKIP__")

    # --- STEP 4: Prepare container payload ---
    payload = {
        "guessed_code": guessed_code,
        "test_cases": test_cases
    }
    payload_json = json.dumps(payload)

    # --- STEP 5: Run ephemeral Docker container ---
    try:
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "--network", "none",
                "-i",
                "functionigma-sandbox"
            ],
            input=payload_json,
            text=True,
            capture_output=True,
            timeout=5
        )
    except subprocess.TimeoutExpired:
        return False, 0.0, "Sandbox execution timed out."
    except Exception as e:
        return False, 0.0, f"Docker execution failed: {e}"

    # --- STEP 6: Parse sandbox JSON response ---
    try:
        clean_output = result.stdout.strip()
        response = json.loads(clean_output)
    except json.JSONDecodeError:
        # Fallback: try to find the last valid JSON line if logs polluted stdout
        lines = result.stdout.strip().split('\n')
        try:
            response = json.loads(lines[-1])
        except:
            return False, 0.0, f"Invalid JSON from sandbox.\nStdout: {result.stdout}\nStderr: {result.stderr}"

    if response.get("status") != "ok":
        error_msg = response.get('error', 'Unknown error')
        return False, 0.0, f"Runtime Error in guess: {error_msg}"

    sandbox_results = response.get("results", [])
    if len(sandbox_results) != len(test_cases):
        return False, 0.0, "Mismatch in number of results returned by sandbox."

    # --- STEP 7: Compare and Calculate Accuracy ---
    passed_count = 0
    total_valid_tests = 0
    failures = []

    for i, (expected, actual) in enumerate(zip(expected_outputs, sandbox_results)):
        if expected == "__SKIP__":
            continue

        total_valid_tests += 1

        # Check equality
        # Note: Be careful with floats here in production (use math.isclose)
        if expected == actual:
            passed_count += 1
        else:
            # Capture the failure details
            failures.append(f"Input: {test_cases[i]}\n   Expected: {expected}\n   Got:      {actual}")

    # Avoid division by zero
    accuracy = passed_count / total_valid_tests if total_valid_tests > 0 else 0.0

    # Prepare Feedback Message
    if passed_count == total_valid_tests:
        return True, 1.0, "All tests passed!"
    else:
        # Show up to 3 failures so we don't spam the transcript
        feedback_msg = f"Passed {passed_count}/{total_valid_tests} tests.\n\nFailed Cases:"
        for f in failures[:3]:
            feedback_msg += f"\n- {f}"

        if len(failures) > 3:
            feedback_msg += f"\n...and {len(failures) - 3} more."

        return False, accuracy, feedback_msg


def extract_function_code(text: str) -> Optional[str]:
    """
    Extracts Python code from a response string.
    """
    pattern = r"```(?:python)?\s*(.*?)```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

    if match:
        return match.group(1).strip()
    return None


# utils.py (or in master.py)
import ast
from typing import List, Tuple


def parse_signature_with_types(signature: str) -> Tuple[List[str], List[str], str]:
    """
    Parse a signature string like "def f(x: int, y: str) -> int"
    Returns (param_names, param_types, return_type_str)
    param_types and return_type_str are strings like "int", "str"
    """
    # Normalize if user gave just "my_function(x:int, y:int) -> int"
    src = signature.strip()
    if src.startswith("def "):
        tree = ast.parse(src)
        func = tree.body[0]
        params = [arg.arg for arg in func.args.args]
        types = []
        for arg in func.args.args:
            if arg.annotation is not None:
                types.append(ast.unparse(arg.annotation))
            else:
                types.append("Any")
        ret = ast.unparse(func.returns) if func.returns is not None else "Any"
        return params, types, ret
    else:
        # Build a tiny wrapper so ast can parse it
        fake = f"def _f({src.split('(')[1]}"
        try:
            tree = ast.parse(fake)
            func = tree.body[0]
            params = [arg.arg for arg in func.args.args]
            types = []
            for arg in func.args.args:
                if arg.annotation is not None:
                    types.append(ast.unparse(arg.annotation))
                else:
                    types.append("Any")
            # try parse return
            if "->" in src:
                ret = src.split("->")[-1].strip()
            else:
                ret = "Any"
            return params, types, ret
        except Exception:
            # fallback: attempt regex parse
            import re
            m = re.match(r".*\((.*)\)\s*->\s*(\w+)", src)
            if m:
                params_raw = m.group(1).strip()
                parts = [p.strip() for p in params_raw.split(",")] if params_raw else []
                names = []
                types_out = []
                for p in parts:
                    if ":" in p:
                        n, t = p.split(":", 1)
                        names.append(n.strip())
                        types_out.append(t.strip())
                    else:
                        names.append(p.split()[0])
                        types_out.append("Any")
                return names, types_out, m.group(2).strip()
            return [], [], "Any"
