from enum import Enum
from typing import List, Callable, Dict, Any


# Typing for categories.
class Category(Enum):
    MATH = "math"
    STRING = "string"
    LIST = 'list'
    BITWISE = 'bitwise'


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


@register(category=Category.MATH, difficulty=Difficulty.EASY)
def f_add(x: int, y: int) -> int:
    return x + y


@register(category=Category.STRING, difficulty=Difficulty.EASY)
def rev_str(x: str) -> str:
    return x[::-1]
