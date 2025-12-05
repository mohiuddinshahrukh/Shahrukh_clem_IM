import re
import random
import inspect
import math
import string
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
                            num_tests: int = 50) -> bool:
    """
    Dynamically inspects the actual function to generate relevant test inputs,
    then compiles and tests the guessed code against it.
    """

    # --- STEP 1: INSPECT THE TARGET FUNCTION ---
    try:
        sig = inspect.signature(actual_callable)
        params = sig.parameters
    except ValueError:
        print("Error: Could not inspect signature of the actual function.")
        return False

    # --- STEP 2: COMPILE THE GUESS ---
    # We expand the global scope to include common built-ins needed for "Medium" problems
    local_scope = {}
    global_scope = {
        "math": math,
        "abs": abs, "min": min, "max": max,
        "len": len, "sum": sum, "sorted": sorted,
        "range": range, "list": list, "int": int, "str": str, "enumerate": enumerate
    }

    try:
        exec(guessed_code, global_scope, local_scope)
    except Exception as e:
        print(f"Syntax Error in guess: {e}")
        return False

    # Find the callable in the guess
    guessed_func = None
    # We prefer a function that has the same name if possible, otherwise take the last one
    target_name = actual_callable.__name__

    if target_name in local_scope and callable(local_scope[target_name]):
        guessed_func = local_scope[target_name]
    else:
        # Fallback: take the last defined callable
        for obj in local_scope.values():
            if callable(obj):
                guessed_func = obj

    if not guessed_func:
        print("No callable function found in parsed code.")
        return False

    # --- STEP 3: RUN DYNAMIC TESTS ---
    for i in range(num_tests):
        try:
            # A. Generate args dynamically based on signature AND Category
            test_args = []
            for name, param in params.items():
                val = generate_random_value(param.annotation, category=category)
                test_args.append(val)
        except Exception as e:
            print(f"Error generating test data: {e}")
            return False

        # B. Run Comparison
        try:
            # Execute Ground Truth
            expected = actual_callable(*test_args)

            # Execute Guess (wrapped to catch crashes)
            try:
                actual = guessed_func(*test_args)
            except Exception as e:
                print(f"Guessed function CRASHED on inputs {test_args}: {e}")
                return False

            # Check Equality
            # Note: For floats, you might need math.isclose(), but for now exact match is fine
            if expected != actual:
                print(f"MISMATCH on inputs {test_args}. Expected: {expected}, Got: {actual}")
                return False

        except Exception as e:
            # If the ACTUAL function crashes (e.g. valid input caused division by zero in ground truth),
            # we skip this specific random case.
            continue

    return True


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
