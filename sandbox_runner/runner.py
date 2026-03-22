import json
import sys
import traceback


# 1. HELPER: Redirect print to stderr to keep stdout clean for JSON
def safe_print(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def main():
    try:
        try:
            payload = json.load(sys.stdin)
        except json.JSONDecodeError:
            payload = {}

        code = payload.get("guessed_code", "")
        tests = payload.get("test_cases", [])

        # ... (Logging to stderr is fine) ...

        # 2. HELPER: Fake Generic for Type Hints (List[int])
        class FakeGeneric:
            def __class_getitem__(cls, item):
                return cls

        local_scope = {}
        safe_globals = {
            "__builtins__": {
                # Base Types
                "int": int, "float": float, "bool": bool, "str": str,
                "list": list, "dict": dict, "tuple": tuple, "set": set,

                # FIX: Add typing compatibility to prevent NameError
                "List": list, "Dict": dict, "Tuple": tuple,
                "Optional": FakeGeneric, "Union": FakeGeneric, "Any": FakeGeneric,

                # Math & Utils
                "abs": abs, "min": min, "max": max, "sum": sum,
                "range": range, "len": len, "enumerate": enumerate, "sorted": sorted,
                "reversed": reversed, "zip": zip, "map": map, "filter": filter,
                "chr": chr, "ord": ord, "all": all, "any": any,

                # FIX: Use safe_print
                "print": safe_print
            }
        }

        exec(code, safe_globals, local_scope)

        # Find function (ignoring our safe_print helper)
        func = next((v for v in local_scope.values() if callable(v) and v is not safe_print), None)
        if not func:
            # Fallback for lambdas
            func = next((v for k, v in local_scope.items() if callable(v)), None)

        if not func:
            raise ValueError("No callable found in guessed code")

        results = []
        for args in tests:
            if not isinstance(args, list):
                args = [args]
            try:
                result = func(*args)
                results.append(result)
            except Exception as e:
                # Log to stderr, return special error marker
                print(f"Test failed: {e}", file=sys.stderr)
                results.append("__ERROR__")

        print(json.dumps({"status": "ok", "results": results}))

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"status": "error", "error": str(e)}))


if __name__ == "__main__":
    main()
