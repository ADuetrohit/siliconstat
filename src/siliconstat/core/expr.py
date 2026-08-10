"""Safe arithmetic expression evaluation.

Used for ``.param`` definitions and ``EXPR`` measurements.  Netlists are
untrusted input (they can be uploaded through the REST API), so this module
deliberately does **not** use :func:`eval`.  It walks a parsed AST and rejects
every node type that is not a pure arithmetic operation, which makes attribute
access, subscripting, comprehensions, lambdas, imports and calls to anything
outside the whitelist structurally impossible.

See ``docs/architecture.md`` (Security) for the threat model.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Callable, Mapping

from .exceptions import SiliconStatError

__all__ = ["safe_eval", "ExpressionError", "ALLOWED_FUNCTIONS", "expression_names"]


class ExpressionError(SiliconStatError):
    """Raised when an expression is malformed or uses a forbidden construct."""


def _db20(x: float) -> float:
    return 20.0 * math.log10(abs(x)) if x else float("-inf")


def _db10(x: float) -> float:
    return 10.0 * math.log10(abs(x)) if x else float("-inf")


ALLOWED_FUNCTIONS: dict[str, Callable[..., float]] = {
    "abs": abs,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "atan": math.atan,
    "asin": math.asin,
    "acos": math.acos,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "floor": math.floor,
    "ceil": math.ceil,
    "min": min,
    "max": max,
    "pow": math.pow,
    "hypot": math.hypot,
    "db": _db20,
    "db20": _db20,
    "db10": _db10,
    "sign": lambda x: math.copysign(1.0, x) if x else 0.0,
}

ALLOWED_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "inf": math.inf,
}

_BIN_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

_UNARY_OPS: dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_MAX_EXPRESSION_LENGTH = 2000


def _evaluate(node: ast.AST, variables: Mapping[str, float]) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, variables)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ExpressionError(f"only numeric literals are allowed (got {node.value!r})")
        return float(node.value)
    if isinstance(node, ast.Name):
        key = node.id
        if key in variables:
            return float(variables[key])
        lowered = key.lower()
        if lowered in variables:
            return float(variables[lowered])
        if lowered in ALLOWED_CONSTANTS:
            return ALLOWED_CONSTANTS[lowered]
        known = ", ".join(sorted(variables)) or "(none)"
        raise ExpressionError(f"unknown symbol {key!r} in expression; available: {known}")
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ExpressionError(
                f"operator {type(node.op).__name__} is not allowed in expressions")
        left = _evaluate(node.left, variables)
        right = _evaluate(node.right, variables)
        try:
            return float(op(left, right))
        except ZeroDivisionError as exc:
            raise ExpressionError(f"division by zero while evaluating expression") from exc
        except OverflowError as exc:
            raise ExpressionError("numeric overflow while evaluating expression") from exc
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ExpressionError(
                f"unary operator {type(node.op).__name__} is not allowed")
        return float(op(_evaluate(node.operand, variables)))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ExpressionError("only direct calls to whitelisted functions are allowed")
        fname = node.func.id.lower()
        fn = ALLOWED_FUNCTIONS.get(fname)
        if fn is None:
            raise ExpressionError(
                f"function {node.func.id!r} is not allowed; permitted functions: "
                f"{', '.join(sorted(ALLOWED_FUNCTIONS))}")
        if node.keywords:
            raise ExpressionError("keyword arguments are not allowed in expressions")
        args = [_evaluate(a, variables) for a in node.args]
        try:
            return float(fn(*args))
        except (ValueError, TypeError, OverflowError) as exc:
            raise ExpressionError(f"error calling {fname}(): {exc}") from exc
    raise ExpressionError(
        f"syntax element {type(node).__name__} is not allowed in expressions")


def safe_eval(expression: str, variables: Mapping[str, float] | None = None) -> float:
    """Evaluate a purely arithmetic *expression* against *variables*."""
    text = (expression or "").strip()
    if not text:
        raise ExpressionError("empty expression")
    if len(text) > _MAX_EXPRESSION_LENGTH:
        raise ExpressionError(
            f"expression is too long ({len(text)} > {_MAX_EXPRESSION_LENGTH} characters)")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"invalid expression {text!r}: {exc.msg}") from exc
    value = _evaluate(tree, variables or {})
    if isinstance(value, complex):  # pragma: no cover - defensive
        raise ExpressionError("expressions must evaluate to a real number")
    return float(value)


def expression_names(expression: str) -> set[str]:
    """Return the free symbol names referenced by *expression*."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"invalid expression {expression!r}: {exc.msg}") from exc
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id.lower() in ALLOWED_CONSTANTS:
                continue
            names.add(node.id)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.discard(node.func.id)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.discard(node.func.id)
    return names
