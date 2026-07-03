"""Test the _preprocess fix for adjacent-letter implicit multiplication."""
import sys
sys.path.insert(0, '.')
from app.utils.math_parser import _preprocess, parse_multivariable_function

tests = [
    # (input, expected_preprocessed, description)
    ('x**2-xy-4', 'x**2-x*y-4', 'xy -> x*y'),
    ('y^2+y*x^2', 'y**2+y*x**2', 'explicit * preserved'),
    ('x*y', 'x*y', 'explicit * preserved 2'),
    ('(x+1)*(y-1)', '(x+1)*(y-1)', 'parentheses with *'),
    ('2*x^2*y', '2*x**2*y', '2*x^2*y'),
    ('xy', 'x*y', 'simple xy'),
    ('ab+cd', 'a*b+c*d', 'ab + cd'),
    ('sin(x)*cos(y)', 'sin(x)*cos(y)', 'sin(x)*cos(y) NOT s*i*n'),
    ('sin(x)cos(y)', 'sin(x)*cos(y)', 'sin(x)cos(y) -> sin(x)*cos(y)'),
]

print("=" * 60)
print("Testing _preprocess() implicit multiplication fix")
print("=" * 60)

all_pass = True
for expr, expected, desc in tests:
    result = _preprocess(expr)
    passed = result == expected
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False
    print(f"  {status}: {desc}")
    print(f"    input:    {expr}")
    print(f"    expected: {expected}")
    print(f"    got:      {result}")
    print()

# Test full parse_multivariable_function with 'x^2-xy-4'
print("-" * 60)
print("Testing parse_multivariable_function('x**2-xy-4', ['x','y'])")
print("-" * 60)
try:
    f = parse_multivariable_function('x**2-xy-4', ['x', 'y'])
    result = f(2, 3)
    print(f"  PASS: f(2,3) = {result} (expected: 2^2 - 2*3 - 4 = 4-6-4 = -6)")
    if abs(result + 6) > 1e-10:
        print(f"  WARNING: expected -6, got {result}")
        all_pass = False
except Exception as e:
    print(f"  FAIL: {e}")
    all_pass = False

# Test full parse_multivariable_function with 'y^2+y*x^2'
print()
print("Testing parse_multivariable_function('y^2+y*x^2', ['x','y'])")
print("-" * 60)
try:
    f = parse_multivariable_function('y^2+y*x^2', ['x', 'y'])
    result = f(2, 3)
    print(f"  PASS: f(2,3) = {result} (expected: 3^2 + 3*2^2 = 9+12 = 21)")
    if abs(result - 21) > 1e-10:
        print(f"  WARNING: expected 21, got {result}")
        all_pass = False
except Exception as e:
    print(f"  FAIL: {e}")
    all_pass = False

print()
print("=" * 60)
if all_pass:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")
print("=" * 60)