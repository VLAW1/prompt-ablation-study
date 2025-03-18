"""
Evaluates the responses in math reasoning domain.

Based on the grader from the "Let's Verify Step by Step" paper's
    repository, PRM800K (the source of the MATH500 dataset):
https://github.com/openai/prm800k/

grader.py link:
https://github.com/openai/prm800k/blob/main/prm800k/grading/grader.py

Which states:
Answer checker API that uses sympy to simplify expressions and check for equality.

Call grade_answer(given_answer: str, ground_truth: str).
"""

import re
import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser

from src.domains.math_reasoning import response_normalization


# sympy might hang -- we don't care about trying to be lenient in these cases
BAD_SUBSTRINGS = ['^{', '^(']
BAD_REGEXES = ['\^[0-9]+\^', '\^[0-9][0-9]+']
TUPLE_CHARS = '()[]'


def _sympy_parse(expr: str) -> sympy.Expr:
    """
    Parses an expression with sympy.

    Parameters
    ----------
    expr : str
        The expression to parse.

    Returns
    -------
    sympy.Expr
        The parsed expression.
    """
    py_expr = expr.replace('^', '**')
    sympy_expr = sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )
    return sympy_expr


def _parse_latex(expr: str) -> str:
    """
    Attempts to parse latex to an expression sympy can read.

    Parameters
    ----------
    expr : str
        The latex expression to parse.

    Returns
    -------
    str
        The parsed expression.
    """
    expr = expr.replace('\\tfrac', '\\frac')
    expr = expr.replace('\\dfrac', '\\frac')
    expr = expr.replace('\\frac', ' \\frac')  # Play nice with mixed numbers.
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    # Replace the specific characters that this parser uses.
    expr = expr.replace('√', 'sqrt')
    expr = expr.replace('π', 'pi')
    expr = expr.replace('∞', 'inf')
    expr = expr.replace('∪', 'U')
    expr = expr.replace('·', '*')
    expr = expr.replace('×', '*')

    return expr.strip()


def _is_float(num: str) -> bool:
    """
    Check if a string is a float.

    Parameters
    ----------
    num : str
        The string to check.

    Returns
    -------
    bool
        True if the string is a float, False otherwise.
    """
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    """
    Check if a float is an integer.

    Parameters
    ----------
    x : float
        The float to check.

    Returns
    -------
    bool
        True if the float is an integer, False otherwise.
    """
    try:
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _is_frac(expr: str) -> bool:
    """
    Check if a string is a fraction.

    Parameters
    ----------
    expr : str
        The string to check.

    Returns
    -------
    bool
        True if the string is a fraction, False otherwise.
    """
    return bool(re.search(r'^-?[0-9]+.?/0*[1-9][0-9]*.?$', expr))


def _str_is_int(x: str) -> bool:
    """
    Check if a string is an integer.

    Parameters
    ----------
    x : str
        The string to check.

    Returns
    -------
    bool
        True if the string is an integer, False otherwise.
    """
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _str_to_int(x: str) -> int:
    """
    Convert a string to an integer.

    Parameters
    ----------
    x : str
        The string to convert to an integer.

    Returns
    -------
    int
        The integer value of the string.
    """
    x = x.replace(',', '')
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str) -> str:
    """
    Inject implicit mixed numbers into the expression.

    Parameters
    ----------
    step : str
        The expression to inject implicit mixed numbers into.

    Returns
    -------
    str
        The expression with implicit mixed numbers injected.
    """
    p1 = re.compile('([0-9]) +([0-9])')
    step = p1.sub('\\1+\\2', step)  ## implicit mults
    return step


def _strip_properly_formatted_commas(expr: str) -> str:
    """
    Strip properly formatted commas from the expression.

    Parameters
    ----------
    expr : str
        The expression to strip properly formatted commas from.

    Returns
    -------
    str
        The expression with commas stripped.
    """
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile('(\d)(,)(\d\d\d)($|\D)')
    while True:
        next_expr = p1.sub('\\1\\3\\4', expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str | None) -> str | None:
    """
    Normalize answer expressions.

    Parameters
    ----------
    expr : str | None
        The expression to normalize.
    Returns
    -------
    str | None
        The normalized expression or None if the expression is None.
    """
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    m = re.search('^\\\\text\{(?P<text>.+?)\}$', expr)
    if m is not None:
        expr = m.group('text')

    expr = expr.replace('\\%', '%')
    expr = expr.replace('\\$', '$')
    expr = expr.replace('$', '')
    expr = expr.replace('%', '')
    expr = expr.replace(' or ', ' , ')
    expr = expr.replace(' and ', ' , ')

    expr = expr.replace('million', '*10^6')
    expr = expr.replace('billion', '*10^9')
    expr = expr.replace('trillion', '*10^12')

    for unit in [
        'degree',
        'cm',
        'centimeter',
        'meter',
        'mile',
        'second',
        'minute',
        'hour',
        'day',
        'week',
        'month',
        'year',
        'foot',
        'feet',
        'inch',
        'yard',
    ]:
        expr = re.sub(f'{unit}(es)?(s)? *(\^[0-9]+)?', '', expr)
    expr = re.sub('\^ *\\\\circ', '', expr)

    if len(expr) > 0 and expr[0] == '{' and expr[-1] == '}':
        expr = expr[1:-1]

    expr = re.sub(',\\\\! *', '', expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if '\\' in expr:
        try:
            expr = _parse_latex(expr)
        except:
            pass

    # edge case with mixed numbers and negative signs
    expr = re.sub('- *', '-', expr)

    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(' ', '')

    # if we somehow still have latex braces here, just drop them
    expr = expr.replace('{', '')
    expr = expr.replace('}', '')

    # don't be case sensitive for text answers
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str) -> int:
    """
    Count the number of unknown letters in an expression.

    Parameters
    ----------
    expr : str
        The expression to count unknown letters in.

    Returns
    -------
    int
        The number of unknown letters in the expression.
    """
    expr = expr.replace('sqrt', '')
    expr = expr.replace('frac', '')
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str) -> bool:
    """
    Determine if an expression should be evaluated.

    Parameters
    ----------
    expr : str
        The expression to determine if should be evaluated.

    Returns
    -------
    bool
        True if the expression should be evaluated, False otherwise.
    """
    # we don't want to try parsing unknown text or functions of more than two variables
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

    return True


def are_equal_under_sympy(
    ground_truth_normalized: str, given_normalized: str
) -> bool:
    """
    Use sympy to determine if two expressions are equal.

    Parameters
    ----------
    ground_truth_normalized : str
        The ground truth expression.
    given_normalized : str
        The given expression.

    Returns
    -------
    bool
        True if the expressions are equal, False otherwise.
    """
    are_equal = False
    try:
        expr = f'({ground_truth_normalized})-({given_normalized})'
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except:
        pass
    return are_equal


def split_tuple(expr: str) -> list[str]:
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers

    Parameters
    ----------
    expr : str
        The expression to split.

    Returns
    -------
    list[str]
        The split elements.
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(',')]
    else:
        elems = [expr]
    return elems


def grade_answer(given_answer: str | None, ground_truth: str) -> bool:
    """
    Grade an answer against a ground truth.
    The answer will be considered correct if:
        (a) it normalizes to the same string as the ground truth answer
        OR
        (b) sympy can simplify the difference between the expressions to 0

    Parameters
    ----------
    given_answer : str | None
        The given answer.
    ground_truth : str
        The ground truth answer.

    Returns
    -------
    bool
        True if the answer is correct, False otherwise.
    """
    if given_answer is None or given_answer == '':
        return False

    ground_truth_normalized_mathd = response_normalization.normalize_answer(
        ground_truth
    )
    given_answer_normalized_mathd = response_normalization.normalize_answer(
        given_answer
    )

    # be at least as lenient as mathd
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True

    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
        return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0]
        or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(
            ground_truth_elems, given_elems
        ):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                # if the ground truth answer is an integer, we require the given answer to be a strict match (no sympy.simplify)
                is_correct = False
            else:
                is_correct = are_equal_under_sympy(
                    ground_truth_elem, given_elem
                )
            if not is_correct:
                break

    return is_correct
