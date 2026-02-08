import re
import string
import inspect
import json
import random
import ast
from typing import Any, Callable, List, Dict, get_origin, get_args, Optional, Tuple


def generate_random_value(annotation: Any, category: str = None) -> Any:
    # ... (Keep your existing generate_random_value implementation exactly as is) ...
    # --- 1. Handle Lists (Recursion) ---
    if get_origin(annotation) is list:
        args = get_args(annotation)
        inner_type = args[0] if args else int
        return [generate_random_value(inner_type, category) for _ in range(random.randint(0, 10))]

    # --- 2. Handle Integers ---
    if annotation is int:
        edge_cases = [0, 1, -1, 10, -10]
        if category == "math":
            edge_cases.extend([100, -100, 99, 1000])
        elif category == "bitwise":
            edge_cases.extend([2, 4, 8, 16, 32, 255])
        return random.choice(edge_cases + [random.randint(-50, 50)])

    # --- 3. Handle Floats ---
    elif annotation is float:
        return random.choice([0.0, 1.0, -1.0, 0.5, random.uniform(-100.0, 100.0)])

    # --- 4. Handle Strings ---
    elif annotation is str:
        if category == "string":
            candidates = ["", " ", "a", "ab", "aba", "   ", "hello world", "12345"]
            return random.choice(candidates)
        letters = string.ascii_lowercase
        random_word = ''.join(random.choice(letters) for _ in range(random.randint(1, 8)))
        return random.choice(["", "a", " ", random_word])

    return random.randint(-50, 50)


# --- helpers: signature/type-aware edge cases ---

def _json_safe(value: Any) -> Any:
    """
    Make sure expected outputs are JSON-serializable the same way the sandbox returns them.
    This mirrors your validate_function_logic normalization approach. :contentReference[oaicite:2]{index=2}
    """
    try:
        return json.loads(json.dumps(value))
    except TypeError:
        return str(value)


def _parse_signature_types_from_callable(actual_callable: Callable) -> List[Any]:
    """
    Return a list of python typing annotations for params (best-effort).
    Falls back to int if missing.
    """
    sig = inspect.signature(actual_callable)
    ann = []
    for p in sig.parameters.values():
        ann.append(p.annotation if p.annotation is not inspect._empty else int)
    return ann


def _edge_values_for_annotation(annotation: Any, category: Optional[str] = None) -> List[Any]:
    """
    Generate a small set of edge candidates per parameter type.
    """
    origin = get_origin(annotation)

    if origin is list:
        inner = get_args(annotation)[0] if get_args(annotation) else int
        # edge lists: empty, singletons, duplicates, negatives, sorted/reversed
        return [
            [],
            [generate_random_value(inner, category)],
            [0],
            [1, 1],
            [2, 1],
            [1, 2, 3],
            [3, 2, 1],
            [-1, 0, 1],
            [5, -5, 5],
        ]

    if annotation is int:
        base = [0, 1, -1, 2, -2, 10, -10, 99, -99, 1000, -1000]
        if category and category.lower() == "math":
            base += [100, -100, 50, -50]
        return base

    if annotation is float:
        return [0.0, 1.0, -1.0, 0.5, -0.5, 10.25, -10.25]

    if annotation is str:
        # include whitespace and punctuation to force generalization
        return ["", " ", "a", "ab", "aba", "A", "aA", "hello", "hello world", "123", "a-b", "   "]

    if annotation is bool:
        return [True, False]

    # fallback
    return [generate_random_value(annotation, category)]


def _special_cases_for_known_functions(function_name: str) -> List[List[Any]]:
    """
    Optional: discriminative cases for specific functions (by name).
    Return a list of args-lists, e.g. [[arg1, arg2], ...]
    Keeps generator robust as you add harder functions.
    """
    if function_name == "first_missing_positive":
        return [
            [[]],
            [[1]],
            [[2]],
            [[1, 2, 0]],
            [[3, 4, -1, 1]],
            [[1, 1, 2, 2]],
        ]
    if function_name == "balanced_parentheses":
        return [
            [""],
            ["()"],
            ["(]"],
            ["([{}])"],
            ["([)]"],
            ["abc(def)"],
            [")("],
        ]
    if function_name == "edit_distance":
        return [
            ["", ""],
            ["", "a"],
            ["a", ""],
            ["a", "a"],
            ["kitten", "sitting"],
            ["flaw", "lawn"],
        ]
    return []


def create_static_test_cases(
        actual_callable: Callable,
        category: str,
        num_tests: int = 50,
        signature: Optional[str] = None,
        difficulty: Optional[str] = None,
) -> List[Dict]:
    """
    Generates a list of static test cases with inputs and expected outputs.
    Returns: [{'args': [1, 2], 'expected': 3}, ...]
    - Difficulty-aware: more edge cases for hard.
    - Deterministic seeding should be done by the caller (instancegenerator), not here.
    """
    # IMPORTANT: do NOT random.seed(42) here; instancegenerator should control seeding.

    # Figure out annotations from the callable (best effort).
    try:
        param_annotations = _parse_signature_types_from_callable(actual_callable)
        sig = inspect.signature(actual_callable)
        param_count = len(sig.parameters)
    except ValueError:
        return []

    function_name = getattr(actual_callable, "__name__", "")

    # Difficulty mix: portion of cases drawn from edge buckets
    diff = (difficulty or "").lower()
    edge_ratio = {"easy": 0.30, "medium": 0.45, "hard": 0.60}.get(diff, 0.40)

    # Build per-parameter edge pools
    edge_pools: List[List[Any]] = [
        _edge_values_for_annotation(ann, category) for ann in param_annotations
    ]

    test_cases: List[Dict] = []
    seen_args = set()

    def _add_case(args: List[Any]) -> None:
        nonlocal test_cases
        key = json.dumps(_json_safe(args), sort_keys=True)
        if key in seen_args:
            return
        try:
            expected = actual_callable(*args)
        except Exception:
            return
        test_cases.append({"args": args, "expected": _json_safe(expected)})
        seen_args.add(key)

    # 1) Add special discriminative cases (if any)
    for args in _special_cases_for_known_functions(function_name):
        if len(args) == param_count:
            _add_case(args)

    # 2) Add edge-bucket cases until we hit target edge budget
    edge_budget = int(round(num_tests * edge_ratio))
    while len(test_cases) < min(num_tests, edge_budget):
        args = []
        for pool in edge_pools:
            args.append(random.choice(pool))
        _add_case(args)

    # 3) Fill remaining with random samples
    safety = 0
    while len(test_cases) < num_tests and safety < num_tests * 50:
        safety += 1
        args = []
        for ann in param_annotations:
            args.append(generate_random_value(ann, category))
        _add_case(args)

    return test_cases


def validate_function_logic(guessed_code: str, test_cases: List[Dict]) -> Tuple[bool, float, str]:
    """
    Runs guessed code against PRE-GENERATED static test cases inside Docker.
    """
    import subprocess
    import json

    # 1. Extract just the arguments to send to the container
    input_args_list = [case["args"] for case in test_cases]
    expected_outputs = [case["expected"] for case in test_cases]

    # 2. Prepare Payload
    payload = {
        "guessed_code": guessed_code,
        "test_cases": input_args_list
    }
    payload_json = json.dumps(payload)

    # 3. Run Docker
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

    # 4. Parse Results
    try:
        clean_output = result.stdout.strip()
        response = json.loads(clean_output)
    except json.JSONDecodeError:
        # Include STDERR so you can see why it crashed!
        return False, 0.0, f"Sandbox Failure.\nStdout: '{result.stdout}'\nStderr: '{result.stderr}'"

    if response.get("status") != "ok":
        return False, 0.0, f"Runtime Error: {response.get('error')}"

    sandbox_results = response.get("results", [])

    if len(sandbox_results) != len(test_cases):
        return False, 0.0, "Mismatch in result count."

    # 5. Compare
    passed_count = 0
    failures = []

    for i, (expected_raw, actual) in enumerate(zip(expected_outputs, sandbox_results)):
        # --- FIX: Normalize Expected Output to match JSON format ---
        # This turns tuples (1,2) into lists [1,2] so they match the sandbox output
        try:
            expected = json.loads(json.dumps(expected_raw))
        except TypeError:
            expected = str(expected_raw)

        if expected == actual:
            passed_count += 1
        else:
            failures.append(f"Input: {input_args_list[i]} | Expected: {expected} | Got: {actual}")

    accuracy = passed_count / len(test_cases) if test_cases else 0.0

    if passed_count == len(test_cases):
        return True, 1.0, "All tests passed!"
    else:
        feedback = "Failed Cases:\n" + "\n".join(failures[:3])
        return False, accuracy, feedback


def extract_function_code(text: str) -> Optional[str]:
    """
    Extracts Python code from a response string.
    """
    pattern = r"```(?:python)?\s*(.*?)```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

    if match:
        return match.group(1).strip()
    return None




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
