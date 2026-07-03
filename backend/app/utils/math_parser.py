"""Mathematical expression parser utility using SymPy for safe evaluation."""
import math as _math
import re as _re
import numpy as np
from typing import Callable, List
from sympy import (
    symbols, sympify, lambdify, pi, E, sqrt, exp, log, ln, sin, cos, tan,
    asin, acos, atan, sinh, cosh, tanh, Abs, Function as _sympy_Function,
)


class _real_cbrt_sym(_sympy_Function):
    """SymPy symbolic wrapper for real cube root: cbrt(-8) = -2."""
    pass


class _real_root_sym(_sympy_Function):
    """SymPy symbolic wrapper for real nth root: real_root(-8, 3) = -2."""
    pass


def _real_root(x: float, n: int) -> float:
    """Compute the real nth root preserving sign for odd n.
    
    Unlike Python's ** which returns complex for negative bases with fractional
    exponents, this always returns the real root: cbrt(-8) = -2.0.
    """
    if n <= 0:
        raise ValueError(f"Root order must be positive, got {n}")
    if n % 2 == 1:
        # Odd root: preserve sign
        return _math.copysign(abs(x) ** (1.0 / n), x)
    else:
        # Even root: requires non-negative input
        if x < 0:
            raise ValueError(f"Cannot compute even root of negative number: {x}^(1/{n})")
        return x ** (1.0 / n)


def _real_cbrt(x: float) -> float:
    """Compute the real cube root preserving sign."""
    return _math.copysign(abs(x) ** (1.0 / 3), x)


def _convert_power(expr: str) -> str:
    """Convert ^ to ** for exponentiation."""
    result = []
    for ch in expr:
        if ch == '^':
            result.append('**')
        else:
            result.append(ch)
    return ''.join(result)


def _preprocess(expr: str) -> str:
    """Preprocess expression string to be SymPy-compatible."""
    s = expr.strip()
    # Remove surrounding math delimiters
    s = _re.sub(r'^\$\$|\$\$$', '', s)
    s = _re.sub(r'^\\\[|\\\]$', '', s)
    
    # Convert ^ to **
    s = _convert_power(s)
    
    # Handle e** -> exp(  (must be done before implicit multiplication)
    # e**x -> exp(x), e**(...) -> exp(...)
    s = _re.sub(r'\be\*\*(\w+)', r'exp(\1)', s)
    s = _re.sub(r'\be\*\*\(([^)]+)\)', r'exp(\1)', s)
    
    # Convert fractional-power patterns to real-root functions
    # (expression)**(1/3) -> _real_cbrt(expression)
    # (expression)**(1/5), (1/7), etc. -> _real_root(expression, odd)
    s = _re.sub(r'\(([^()]+)\)\*\*\(1/3\)', r'_real_cbrt(\1)', s)
    s = _re.sub(r'\(([^()]+)\)\*\*\(1/5\)', r'_real_root(\1,5)', s)
    s = _re.sub(r'\(([^()]+)\)\*\*\(1/7\)', r'_real_root(\1,7)', s)
    s = _re.sub(r'\(([^()]+)\)\*\*\(1/9\)', r'_real_root(\1,9)', s)
    # Also handle single variable: x**(1/3) -> _real_cbrt(x)
    s = _re.sub(r'\b([a-zA-Z])\*\*\(1/3\)', r'_real_cbrt(\1)', s)
    s = _re.sub(r'\b([a-zA-Z])\*\*\(1/5\)', r'_real_root(\1,5)', s)
    s = _re.sub(r'\b([a-zA-Z])\*\*\(1/7\)', r'_real_root(\1,7)', s)
    s = _re.sub(r'\b([a-zA-Z])\*\*\(1/9\)', r'_real_root(\1,9)', s)
    
    # STEP 0: Temporarily mask known function names so they don't get
    #         split by adjacent-letter implicit multiplication rules.
    #         e.g., "sin" must NOT become "s*i*n".
    _FUNC_NAMES = [
        'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh',
        'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
        'log10', 'log', 'ln', 'exp', 'sqrt', 'abs', 'pi',
    ]
    _masks = {}
    _counter = [0]
    
    def _mask_func(match: _re.Match) -> str:
        token = f'__FUNC{_counter[0]}__'
        _masks[token] = match.group(0)
        _counter[0] += 1
        return token
    
    # Sort by length descending so longer names match first (arcsin before sin)
    for fn in sorted(_FUNC_NAMES, key=len, reverse=True):
        s = _re.sub(r'\b' + fn + r'\b', _mask_func, s)
    
    # Implicit multiplication rules:
    # 1. digit + letter: 2x -> 2*x (only after masked names have been replaced)
    s = _re.sub(r'(\d)([a-zA-Z])', r'\1*\2', s)
    # 2. digit + ( : 2( -> 2*(
    s = _re.sub(r'(\d)(\()', r'\1*\2', s)
    # 3. ) + word char: )x -> )*x  (use \w for masked function tokens like __FUNC0__)
    s = _re.sub(r'(\))(\w)', r'\1*\2', s)
    # 4. ) + ( : )( -> )*(
    s = _re.sub(r'(\))(\()', r'\1*\2', s)
    # 5. ) + digit: )2 -> )*2
    s = _re.sub(r'(\))(\d)', r'\1*\2', s)
    # 6. Single letter variable followed by ( : x(x+1) -> x*(x+1)
    #    Use \b (word boundary) to ensure letter is standalone, NOT part of function name
    #    After masking, function names are __FUNC0__ etc, so \b works safely
    s = _re.sub(r'\b([a-zA-Z])(\()', r'\1*\2', s)
    # 7. Adjacent single-letter variables: xy -> x*y, y^2zx -> y^{2}*z*x
    #    After masking, only single-letter variables remain un-masked
    s = _re.sub(r'\b([a-zA-Z])([a-zA-Z])', r'\1*\2', s)
    
    # Unmask: restore all function names
    for token, original in _masks.items():
        s = s.replace(token, original)
    
    return s


def parse_function(expr: str) -> Callable[[float], float]:
    """Parse a mathematical expression into a callable function f(x) using SymPy.
    
    Supports: +, -, *, /, **, ^, sin, cos, tan, asin, acos, atan,
              sinh, cosh, tanh, exp, log, ln, sqrt, abs, pi, e
    """
    try:
        processed = _preprocess(expr)
        x = symbols('x')
        
        local_dict = {
            'e': E, 'E': E, 'pi': pi,
            'sin': sin, 'cos': cos, 'tan': tan,
            'asin': asin, 'acos': acos, 'atan': atan,
            'sinh': sinh, 'cosh': cosh, 'tanh': tanh,
            'sqrt': sqrt, 'exp': exp, 'log': log, 'ln': ln,
            'abs': Abs,
            # Use SymPy Function subclasses so they remain unevaluated during parsing
            '_real_cbrt': _real_cbrt_sym,
            '_real_root': _real_root_sym,
        }
        
        sympy_expr = sympify(processed, locals=local_dict)
        # Provide numeric implementations during lambdify
        # lambdify uses class name as function name: _real_cbrt_sym, _real_root_sym
        _custom_impl = {
            '_real_cbrt_sym': _real_cbrt,
            '_real_root_sym': _real_root,
        }
        f = lambdify(x, sympy_expr, modules=[_custom_impl, 'numpy'])
        return f
    except (SyntaxError, TypeError, ValueError, AttributeError) as e:
        raise ValueError(f"Invalid mathematical expression: '{expr}'. Error: {e}")


def parse_multivariable_function(expr: str, variables: List[str]) -> Callable[..., float]:
    """Parse a multivariable expression into a callable function using SymPy."""
    try:
        processed = _preprocess(expr)
        sym_vars = symbols(variables)
        
        local_dict = {
            'e': E, 'E': E, 'pi': pi,
            'sin': sin, 'cos': cos, 'tan': tan,
            'asin': asin, 'acos': acos, 'atan': atan,
            'sinh': sinh, 'cosh': cosh, 'tanh': tanh,
            'sqrt': sqrt, 'exp': exp, 'log': log, 'ln': ln,
            'abs': Abs,
            # Use SymPy Function subclasses so they remain unevaluated during parsing
            '_real_cbrt': _real_cbrt_sym,
            '_real_root': _real_root_sym,
        }
        
        # Register actual SymPy symbols (NOT None, which breaks pow operations).
        # sympy.symbols(['x','y']) returns a list, symbols('x') returns Symbol.
        if isinstance(sym_vars, (list, tuple)):
            for i, var_name in enumerate(variables):
                if i < len(sym_vars):
                    local_dict[var_name] = sym_vars[i]
        else:
            # Single symbol
            local_dict[variables[0]] = sym_vars
        
        sympy_expr = sympify(processed, locals=local_dict)
        # Provide numeric implementations during lambdify
        # lambdify uses class name as function name: _real_cbrt_sym, _real_root_sym
        _custom_impl = {
            '_real_cbrt_sym': _real_cbrt,
            '_real_root_sym': _real_root,
        }
        f = lambdify(sym_vars, sympy_expr, modules=[_custom_impl, 'numpy'])
        return f
    except (SyntaxError, TypeError, ValueError, AttributeError) as e:
        raise ValueError(f"Invalid expression: '{expr}'. Error: {e}")


def parse_function_latex(latex_expr: str) -> Callable[[float], float]:
    """Parse a LaTeX math expression into a callable function f(x)."""
    py_expr = latex_to_python(latex_expr)
    return parse_function(py_expr)


def parse_multivariable_latex(latex_expr: str, variables: List[str]) -> Callable[..., float]:
    """Parse a LaTeX math expression into a multivariable callable function."""
    py_expr = latex_to_python(latex_expr)
    return parse_multivariable_function(py_expr, variables)


def compute_numerical_jacobian(
    functions: List[str],
    variables: List[str],
    point: List[float],
    h: float = 1e-8
) -> List[List[float]]:
    """Compute the Jacobian matrix numerically using central differences."""
    n = len(functions)
    m = len(variables)
    jacobian = [[0.0] * m for _ in range(n)]
    
    for i, func_expr in enumerate(functions):
        try:
            f = parse_multivariable_function(func_expr, variables)
        except ValueError:
            continue
            
        for j in range(m):
            point_plus = list(point)
            point_minus = list(point)
            point_plus[j] += h
            point_minus[j] -= h
            try:
                f_plus = f(*point_plus)
                f_minus = f(*point_minus)
                jacobian[i][j] = (f_plus - f_minus) / (2 * h)
            except Exception:
                jacobian[i][j] = 0.0
    
    return jacobian


def latex_to_python(latex_expr: str) -> str:
    """Convert LaTeX math expression to Python eval-compatible expression."""
    expr = latex_expr.strip()
    
    # Remove \\left and \\right
    expr = _re.sub(r'\\left\b', '', expr)
    expr = _re.sub(r'\\right\b', '', expr)
    
    # Remove display math delimiters
    expr = _re.sub(r'^\$\$|\$\$$', '', expr)
    expr = _re.sub(r'^\\\[|\\\]$', '', expr)
    
    # Convert LaTeX functions to plain names
    func_map = {
        r'\\sin\b': 'sin', r'\\cos\b': 'cos', r'\\tan\b': 'tan',
        r'\\cot\b': 'cot', r'\\arcsin\b': 'arcsin', r'\\arccos\b': 'arccos',
        r'\\arctan\b': 'arctan', r'\\sinh\b': 'sinh', r'\\cosh\b': 'cosh',
        r'\\tanh\b': 'tanh', r'\\ln\b': 'ln', r'\\log\b': 'log',
        r'\\sqrt\b': 'sqrt', r'\\exp\b': 'exp',
    }
    for latex_func, py_func in func_map.items():
        expr = _re.sub(latex_func, py_func, expr)
    
    # Convert \\pi -> pi
    expr = _re.sub(r'\\pi\b', 'pi', expr)
    
    # Convert e^{...} -> e**(...)
    expr = _re.sub(r'\be\^{([^}]+)}', r'e**(\1)', expr)
    
    # Convert ^{...} -> **(...)
    expr = _re.sub(r'\^\{([^}]+)\}', r'**(\1)', expr)
    
    # Convert ^digit -> **digit
    expr = _re.sub(r'\^(\d+)', r'**\1', expr)
    
    # Convert \\cdot -> *
    expr = _re.sub(r'\\cdot\b', '*', expr)
    
    # Convert \\times -> *
    expr = _re.sub(r'\\times\b', '*', expr)
    
    # Convert \\frac{a}{b} -> (a)/(b)
    while '\\frac' in expr:
        expr = _re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', expr, count=1)
    
    # Handle remaining braces for function args
    expr = expr.replace('{', '(').replace('}', ')')
    
    # Handle matrix newlines
    expr = _re.sub(r'\\\\', ', ', expr)
    expr = _re.sub(r'&', ', ', expr)
    
    return expr