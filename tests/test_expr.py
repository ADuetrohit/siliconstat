"""The safe expression evaluator.

Netlists are untrusted input -- they can be uploaded through the REST API --
so this module is a security boundary and is tested like one.  The point of
these tests is not that dangerous expressions produce wrong answers, but that
they are *structurally impossible* to express.
"""

from __future__ import annotations

import math

import pytest

from siliconstat.core.expr import (
    ALLOWED_FUNCTIONS,
    ExpressionError,
    expression_names,
    safe_eval,
)


# ---------------------------------------------------------------------------
# it computes correctly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("expression,expected", [
    ("1+1", 2.0),
    ("2*3 - 4/2", 4.0),
    ("2**10", 1024.0),
    ("-5", -5.0),
    ("+5", 5.0),
    ("7 % 4", 3.0),
    ("7 // 2", 3.0),
    ("(1+2)*(3+4)", 21.0),
    ("1e-3 * 1e3", 1.0),
])
def test_arithmetic(expression, expected):
    assert safe_eval(expression) == pytest.approx(expected)


def test_variables():
    variables = {"iout": 1.0082e-5, "iref": 1.0e-5}
    assert safe_eval("100*(iout-iref)/iref", variables) == pytest.approx(0.82, rel=1e-9)


def test_variable_lookup_is_case_insensitive_as_a_fallback():
    assert safe_eval("VOUT", {"vout": 1.25}) == pytest.approx(1.25)


def test_constants():
    assert safe_eval("pi") == pytest.approx(math.pi)
    assert safe_eval("e") == pytest.approx(math.e)


@pytest.mark.parametrize("name", sorted(ALLOWED_FUNCTIONS))
def test_every_whitelisted_function_is_callable(name):
    """Each entry in the whitelist must actually work with sensible arguments."""
    arity_two = {"min", "max", "pow", "hypot"}
    expression = f"{name}(0.5, 0.25)" if name in arity_two else f"{name}(0.5)"
    result = safe_eval(expression)
    assert isinstance(result, float)


def test_db_helper():
    assert safe_eval("db(10)") == pytest.approx(20.0)
    assert safe_eval("db10(10)") == pytest.approx(10.0)


def test_nested_calls():
    assert safe_eval("sqrt(abs(-16)) + max(1, 2)") == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# it refuses everything else
# ---------------------------------------------------------------------------

DANGEROUS = [
    pytest.param("__import__('os')", id="import"),
    pytest.param("__import__('os').system('echo hi')", id="import-system"),
    pytest.param("open('secret.txt')", id="open"),
    pytest.param("eval('1+1')", id="eval"),
    pytest.param("exec('x=1')", id="exec"),
    pytest.param("globals()", id="globals"),
    pytest.param("(1).__class__", id="attribute"),
    pytest.param("x.__class__.__bases__", id="attribute-chain"),
    pytest.param("[1,2,3][0]", id="subscript"),
    pytest.param("d['key']", id="subscript-name"),
    pytest.param("lambda: 1", id="lambda"),
    pytest.param("[i for i in (1,2)]", id="listcomp"),
    pytest.param("{i: i for i in (1,2)}", id="dictcomp"),
    pytest.param("(i for i in (1,2))", id="genexp"),
    pytest.param("'string'", id="string-literal"),
    pytest.param("f'{1}'", id="fstring"),
    pytest.param("True", id="bool-literal"),
    pytest.param("None", id="none-literal"),
    pytest.param("[1, 2]", id="list-literal"),
    pytest.param("{1: 2}", id="dict-literal"),
    pytest.param("(1, 2)", id="tuple-literal"),
    pytest.param("1 if 2 else 3", id="ternary"),
    pytest.param("1 and 2", id="boolop"),
    pytest.param("not 1", id="not"),
    pytest.param("1 < 2", id="compare"),
    pytest.param("1 & 2", id="bitand"),
    pytest.param("1 << 2", id="shift"),
    pytest.param("print(1)", id="print"),
    pytest.param("getattr(1, 'real')", id="getattr"),
    pytest.param("setattr", id="bare-name-not-defined"),
]


@pytest.mark.parametrize("expression", DANGEROUS)
def test_dangerous_constructs_are_rejected(expression):
    with pytest.raises(ExpressionError):
        safe_eval(expression, {"x": 1.0, "d": 2.0})


def test_walrus_is_rejected():
    with pytest.raises(ExpressionError):
        safe_eval("(y := 4)")


def test_unknown_symbol_reports_what_is_available():
    with pytest.raises(ExpressionError) as excinfo:
        safe_eval("gain * 2", {"vout": 1.0, "vin": 0.5})
    message = str(excinfo.value)
    assert "gain" in message
    assert "vout" in message and "vin" in message


def test_unknown_function_lists_the_whitelist():
    with pytest.raises(ExpressionError) as excinfo:
        safe_eval("cbrt(8)")
    assert "sqrt" in str(excinfo.value)


def test_keyword_arguments_are_rejected():
    with pytest.raises(ExpressionError):
        safe_eval("max(1, key=2)")


def test_empty_expression_is_rejected():
    with pytest.raises(ExpressionError):
        safe_eval("")
    with pytest.raises(ExpressionError):
        safe_eval("   ")


def test_syntax_error_is_reported_cleanly():
    with pytest.raises(ExpressionError) as excinfo:
        safe_eval("1 +")
    assert "invalid expression" in str(excinfo.value)


def test_absurdly_long_expression_is_rejected():
    with pytest.raises(ExpressionError) as excinfo:
        safe_eval("1+" * 2000 + "1")
    assert "too long" in str(excinfo.value)


def test_division_by_zero_becomes_a_domain_error():
    with pytest.raises(ExpressionError) as excinfo:
        safe_eval("1/0")
    assert "zero" in str(excinfo.value).lower()


def test_math_domain_error_is_wrapped():
    with pytest.raises(ExpressionError):
        safe_eval("sqrt(-1)")


def test_deeply_nested_expression_still_evaluates():
    """Nesting is allowed; only the *kinds* of node are restricted."""
    assert safe_eval("((((1+2))))") == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# name extraction
# ---------------------------------------------------------------------------

def test_expression_names_excludes_functions_and_constants():
    names = expression_names("sqrt(iout) + iref * pi")
    assert names == {"iout", "iref"}


def test_expression_names_on_invalid_input():
    with pytest.raises(ExpressionError):
        expression_names("1 +")
