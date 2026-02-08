from enum import Enum
from typing import List, Callable, Dict, Any


# Typing for categories.
class Category(Enum):
    MATH = "math"
    STRING = "string"


class Difficulty(Enum):
    EASY = 'easy'
    MEDIUM = 'medium'
    HARD = 'hard'


# making a registry to store metadata about functions.
FUNCTION_REGISTRY: List[Dict[str, Any]] = []


def register(category: Category, difficulty: Difficulty):
    # Here we can define our decorator function.
    def decorator(function: Callable):
        FUNCTION_REGISTRY.append({
            'function_name': function.__name__,
            'callable': function,
            'category': category.value,
            'difficulty': difficulty.value,
            'signature': _get_signature_str(function)
        })
        return function

    return decorator


def _get_signature_str(function):
    import inspect
    sig = inspect.signature(function)
    return str(sig)


# -----------------------
# EASY (quick to infer)
# -----------------------


@register(category=Category.MATH, difficulty=Difficulty.EASY)
def f_add(x: int, y: int) -> int:
    return x + y


@register(category=Category.STRING, difficulty=Difficulty.EASY)
def rev_str(x: str) -> str:
    return x[::-1]


@register(category=Category.MATH, difficulty=Difficulty.EASY)
def bubble_sort(xs: List[int]) -> List[int]:
    n = len(xs)
    arr = xs.copy()
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


@register(category=Category.MATH, difficulty=Difficulty.EASY)
def abs_diff(x: int, y: int) -> int:
    """Absolute difference."""
    return abs(x - y)


@register(category=Category.MATH, difficulty=Difficulty.EASY)
def clamp(x: int, lo: int, hi: int) -> int:
    """Clamp x into [lo, hi]. Assumes lo <= hi."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


@register(category=Category.STRING, difficulty=Difficulty.EASY)
def count_vowels(s: str) -> int:
    """Count vowels (aeiou) case-insensitive."""
    vowels = set("aeiou")
    return sum(1 for ch in s.lower() if ch in vowels)


@register(category=Category.STRING, difficulty=Difficulty.EASY)
def is_palindrome(s: str) -> bool:
    """Palindrome check, case-sensitive."""
    return s == s[::-1]


# -----------------------
# MEDIUM (needs more probes)
# -----------------------

@register(category=Category.MATH, difficulty=Difficulty.MEDIUM)
def gcd_euclid(a: int, b: int) -> int:
    """Greatest common divisor (non-negative)."""
    a, b = abs(a), abs(b)
    while b != 0:
        a, b = b, a % b
    return a


@register(category=Category.MATH, difficulty=Difficulty.MEDIUM)
def unique_sorted(xs: List[int]) -> List[int]:
    """Return sorted unique elements."""
    return sorted(set(xs))


@register(category=Category.MATH, difficulty=Difficulty.MEDIUM)
def rotate_left(xs: List[int], k: int) -> List[int]:
    """Rotate list left by k positions."""
    if not xs:
        return []
    n = len(xs)
    k = k % n
    return xs[k:] + xs[:k]


@register(category=Category.STRING, difficulty=Difficulty.MEDIUM)
def caesar_shift(s: str, k: int) -> str:
    """
    Shift letters by k (Caesar cipher).
    Only shifts A-Z and a-z; other chars unchanged.
    """
    out = []
    k = k % 26
    for ch in s:
        if "a" <= ch <= "z":
            out.append(chr((ord(ch) - ord("a") + k) % 26 + ord("a")))
        elif "A" <= ch <= "Z":
            out.append(chr((ord(ch) - ord("A") + k) % 26 + ord("A")))
        else:
            out.append(ch)
    return "".join(out)


# -----------------------
# HARD (edge cases / deeper structure)
# -----------------------

@register(category=Category.MATH, difficulty=Difficulty.HARD)
def first_missing_positive(xs: List[int]) -> int:
    """
    Return the smallest missing positive integer.
    Example: [3,4,-1,1] -> 2 ; [1,2,0] -> 3
    """
    present = set(x for x in xs if x > 0)
    m = 1
    while m in present:
        m += 1
    return m


@register(category=Category.STRING, difficulty=Difficulty.HARD)
def balanced_parentheses(s: str) -> bool:
    """
    Check if (), [], {} are balanced and properly nested.
    Ignores non-bracket characters.
    """
    stack = []
    opening = {"(": ")", "[": "]", "{": "}"}
    closing = {")", "]", "}"}
    for ch in s:
        if ch in opening:
            stack.append(opening[ch])
        elif ch in closing:
            if not stack or stack.pop() != ch:
                return False
    return len(stack) == 0


@register(category=Category.STRING, difficulty=Difficulty.HARD)
def edit_distance(a: str, b: str) -> int:
    """
    Levenshtein edit distance (insert/delete/replace), O(len(a)*len(b)).
    """
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev_diag = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,  # delete
                dp[j - 1] + 1,  # insert
                prev_diag + cost  # replace
            )
            prev_diag = temp
    return dp[m]
