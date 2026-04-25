from enum import Enum
from typing import List, Callable, Dict, Any
import re


class Category(Enum):
    SCALAR_MATH = "SCALAR_MATH"
    PAIR_MATH = "PAIR_MATH"
    LIST_SEQUENCE = "LIST_SEQUENCE"
    STRING = "STRING"
    LOGIC_FORMAL = "LOGIC_FORMAL"


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


FUNCTION_REGISTRY: List[Dict[str, Any]] = []


def register(category: Category, difficulty: Difficulty, confusers=None, family_id: str = "", hint: str = ""):
    confusers = confusers or []

    def decorator(function: Callable):
        FUNCTION_REGISTRY.append({
            "function_name": function.__name__,
            "callable": function,
            "category": category.value,
            "difficulty": difficulty.value,
            "signature": _get_signature_str(function),
            "confusers": confusers,
            "family_id": family_id,
            "hint": hint,
        })
        return function

    return decorator


def _get_signature_str(function):
    import inspect
    return str(inspect.signature(function))


# ============================================================
# SCALAR_MATH
# ============================================================

@register(
    category=Category.SCALAR_MATH,
    difficulty=Difficulty.EASY,
    confusers=["square", "piecewise_linear_break0", "piecewise_linear_break1"],
    family_id="scalar_basic",
    hint="We are looking for a simple one-variable numeric rule (single input → single output). Consider sign-sensitive behavior."
)
def abs_value(x: int) -> int:
    """Return absolute value of x."""
    return abs(x)


# @register(
#     category=Category.SCALAR_MATH,
#     difficulty=Difficulty.EASY,
#     confusers=["abs_value", "clamp_0_10", "clamp_0_9"],
#     family_id="scalar_basic",
#     hint="We are looking for a simple one-variable numeric rule. Consider growth faster than linear (outputs increase quickly for larger |x|)."
# )
# def square(x: int) -> int:
#     """Return x squared."""
#     return x * x


# @register(
#     category=Category.SCALAR_MATH,
#     difficulty=Difficulty.MEDIUM,
#     confusers=["clamp_0_9", "piecewise_linear_break0", "piecewise_linear_break1"],
#     family_id="clamp_range",
#     hint="We are looking for a piecewise rule with saturation (outputs stop changing beyond certain bounds). Probe outside the normal range."
# )
# def clamp_0_10(x: int) -> int:
#     """Clamp x into [0, 10]."""
#     if x < 0:
#         return 0
#     if x > 10:
#         return 10
#     return x


@register(
    category=Category.SCALAR_MATH,
    difficulty=Difficulty.MEDIUM,
    confusers=["clamp_0_10", "piecewise_linear_break0", "piecewise_linear_break1"],
    family_id="clamp_range"
)
def clamp_0_9(x: int) -> int:
    """Clamp x into [0, 9]."""
    if x < 0:
        return 0
    if x > 9:
        return 9
    return x


# @register(
#     category=Category.SCALAR_MATH,
#     difficulty=Difficulty.HARD,
#     confusers=["piecewise_linear_break1", "clamp_0_10", "clamp_0_9"],
#     family_id="piecewise_linear"
# )
# def piecewise_linear_break0(x: int) -> int:
#     """2*x if x < 0 else x + 3."""
#     return 2 * x if x < 0 else x + 3


@register(
    category=Category.SCALAR_MATH,
    difficulty=Difficulty.HARD,
    confusers=["piecewise_linear_break0", "clamp_0_10", "clamp_0_9"],
    family_id="piecewise_linear"
)
def piecewise_linear_break1(x: int) -> int:
    """2*x if x < 1 else x + 3."""
    return 2 * x if x < 1 else x + 3


# ============================================================
# PAIR_MATH
# ============================================================

@register(
    category=Category.PAIR_MATH,
    difficulty=Difficulty.EASY,
    confusers=["max_of_two", "signed_difference", "abs_difference"],
    family_id="pair_order"
)
def min_of_two(x: int, y: int) -> int:
    """Return min(x, y)."""
    return x if x < y else y


# @register(
#     category=Category.PAIR_MATH,
#     difficulty=Difficulty.EASY,
#     confusers=["min_of_two", "signed_difference", "abs_difference"],
#     family_id="pair_order"
# )
# def max_of_two(x: int, y: int) -> int:
#     """Return max(x, y)."""
#     return x if x > y else y


# @register(
#     category=Category.PAIR_MATH,
#     difficulty=Difficulty.MEDIUM,
#     confusers=["abs_difference", "min_of_two", "max_of_two"],
#     family_id="pair_difference"
# )
# def signed_difference(x: int, y: int) -> int:
#     """Return x - y."""
#     return x - y


@register(
    category=Category.PAIR_MATH,
    difficulty=Difficulty.MEDIUM,
    confusers=["signed_difference", "clamp_absdiff_0_10", "min_of_two", "max_of_two"],
    family_id="pair_difference"
)
def abs_difference(x: int, y: int) -> int:
    """Return abs(x - y)."""
    return abs(x - y)


# @register(
#     category=Category.PAIR_MATH,
#     difficulty=Difficulty.HARD,
#     confusers=["abs_difference", "clamp_x_0_10_ignore_y", "signed_difference"],
#     family_id="pair_composition"
# )
# def clamp_absdiff_0_10(x: int, y: int) -> int:
#     """Return clamp(abs(x-y), 0, 10)."""
#     d = abs(x - y)
#     if d > 10:
#         return 10
#     return d  # d is always >= 0


@register(
    category=Category.PAIR_MATH,
    difficulty=Difficulty.HARD,
    confusers=["clamp_absdiff_0_10", "min_of_two", "max_of_two"],
    family_id="pair_composition"
)
def clamp_x_0_10_ignore_y(x: int, y: int) -> int:
    """Return clamp(x, 0, 10). Ignores y."""
    if x < 0:
        return 0
    if x > 10:
        return 10
    return x


# ============================================================
# STRING
# ============================================================

@register(
    category=Category.STRING,
    difficulty=Difficulty.HARD,
    confusers=["is_palindrome_alnum"],
    family_id="palindrome"
)
def is_palindrome_exact(s: str) -> bool:
    """Return True if s is exactly equal to its reverse (case/punct matter)."""
    return s == s[::-1]


# @register(
#     category=Category.STRING,
#     difficulty=Difficulty.HARD,
#     confusers=["is_palindrome_exact"],
#     family_id="palindrome"
# )
# def is_palindrome_alnum(s: str) -> bool:
#     """Return True if alnum-only, lowercased s is a palindrome."""
#     filtered = "".join(ch.lower() for ch in s if ch.isalnum())
#     return filtered == filtered[::-1]


# ============================================================
# LOGIC_FORMAL
# ============================================================

# @register(
#     category=Category.LOGIC_FORMAL,
#     difficulty=Difficulty.MEDIUM,
#     confusers=["has_equal_counts_all_brackets", "balanced_parentheses_only", "balanced_brackets_all"],
#     family_id="bracket_counting"
# )
# def has_equal_paren_counts(s: str) -> bool:
#     """True iff count('(') == count(')') (ignores order and other bracket types)."""
#     return s.count("(") == s.count(")")


@register(
    category=Category.LOGIC_FORMAL,
    difficulty=Difficulty.MEDIUM,
    confusers=["has_equal_paren_counts", "balanced_brackets_all", "balanced_parentheses_only"],
    family_id="bracket_counting"
)
def has_equal_counts_all_brackets(s: str) -> bool:
    """
    True iff counts match independently for (), [], {}.
    Ignores nesting/order, only counts.
    """
    return (
            s.count("(") == s.count(")")
            and s.count("[") == s.count("]")
            and s.count("{") == s.count("}")
    )


@register(
    category=Category.LOGIC_FORMAL,
    difficulty=Difficulty.HARD,
    confusers=["balanced_parentheses_only", "has_equal_counts_all_brackets"],
    family_id="bracket_nesting"
)
def balanced_brackets_all(s: str) -> bool:
    """
    Properly nested balance for (), [], {}.
    Ignores non-bracket characters.
    """
    stack: List[str] = []
    opening = {"(": ")", "[": "]", "{": "}"}
    closing = {")", "]", "}"}

    for ch in s:
        if ch in opening:
            stack.append(opening[ch])
        elif ch in closing:
            if not stack or stack.pop() != ch:
                return False
    return len(stack) == 0


# @register(
#     category=Category.LOGIC_FORMAL,
#     difficulty=Difficulty.HARD,
#     confusers=["balanced_brackets_all", "has_equal_paren_counts"],
#     family_id="bracket_nesting"
# )
# def balanced_parentheses_only(s: str) -> bool:
#     """
#     Properly nested balance for () only.
#     Ignores any other characters (including []{}).
#     """
#     stack = 0
#     for ch in s:
#         if ch == "(":
#             stack += 1
#         elif ch == ")":
#             stack -= 1
#             if stack < 0:
#                 return False
#     return stack == 0
