/**
 * Chuyển đổi biểu thức toán học raw input sang LaTeX hiển thị.
 * 
 * Xử lý các pattern:
 *   x^2        -> x^{2}
 *   x^(2)      -> x^{2}
 *   (x+1)^3    -> (x+1)^{3}
 *   e^x        -> e^{x}
 *   e^(3x)     -> e^{3x}
 *   sin(x)     -> \sin(x)
 *   cos, tan, ln, log, sqrt, abs...
 *   exp(...)   -> e^{...}
 *   2x         -> 2x (giữ nguyên, dùng \cdot nếu cần)
 *   a*b        -> a \cdot b
 */

export function toLatex(expr: string): string {
  if (!expr || !expr.trim()) return '';
  
  let s = expr.trim();

  // STEP 0: Normalize ** -> ^ (Python-style power back to ^ for LaTeX processing)
  s = s.replace(/\*\*/g, '^');

  // STEP 1: Convert function names to LaTeX (\sin, \cos, etc.)
  // Must be done BEFORE exponent handling because function names contain letters
  const funcMap: Record<string, string> = {
    'sin': '\\sin',
    'cos': '\\cos',
    'tan': '\\tan',
    'cot': '\\cot',
    'arcsin': '\\arcsin',
    'arccos': '\\arccos',
    'arctan': '\\arctan',
    'sinh': '\\sinh',
    'cosh': '\\cosh',
    'tanh': '\\tanh',
    'ln': '\\ln',
    'log10': '\\log_{10}',
    'log': '\\log',
    'sqrt': '\\sqrt',
    'pi': '\\pi',
    'abs': '\\left|',
  };
  
  // Replace function names (word boundaries to avoid partial matches)
  for (const [fn, latexFn] of Object.entries(funcMap)) {
    // Only replace when followed by (, space, operator, or end
    s = s.replace(new RegExp('\\b' + fn + '(?=\\s*\\()', 'g'), latexFn);
  }

  // STEP 2: Handle exp(...) -> e^{...}
  s = s.replace(/\\exp\s*\(([^)]+)\)/g, 'e^{$1}');

  // STEP 3: Handle e^ as a special case before general exponent
  // e^x -> e^{x}, e^(expr) -> e^{expr}
  s = s.replace(/e\^(\w+)/g, 'e^{$1}');
  s = s.replace(/e\^\(([^)]+)\)/g, 'e^{$1}');

  // STEP 4: Handle general exponentiation x^y -> x^{y}
  // Pattern: ^\w+  or  ^(...)  
  // First handle ^(expression) -> ^{expression}
  s = s.replace(/\^\(([^)]+)\)/g, '^{$1}');
  // Then handle ^word -> ^{word}
  s = s.replace(/\^(\w+)/g, '^{$1}');
  // Handle ^number -> ^{number}
  s = s.replace(/\^(\d+)/g, '^{$1}');

  // STEP 5: Convert * to \cdot
  s = s.replace(/\*/g, ' \\cdot ');

  // STEP 6: abs(...) -> |...| (abs was already converted to \left| above)
  // Add matching \right| for each \left|
  // Count \left| occurrences and add \right|
  let absCount = 0;
  let result = '';
  for (let i = 0; i < s.length; i++) {
    if (s.substring(i).startsWith('\\left|')) {
      absCount++;
      result += '\\left|';
      i += 6; // skip past \left|
    } else if (s[i] === ')' && absCount > 0) {
      // Check if this is the closing of an abs block
      // Simple heuristic: if we have an open \left| without matching \right|
      result += '\\right|)';
      absCount--;
    } else {
      result += s[i];
    }
  }
  s = result;

  // STEP 7: Remove extra backslashes that might have been introduced
  s = s.replace(/\\\\\\/g, '\\');

  return s;
}

/**
 * Chuyển đổi raw input expression sang Python-compatible string.
 * Xử lý: ^ -> **, implicit multiplication, function normalization
 * 
 * VD: 'sin(2x)-e^(3x)' -> 'sin(2*x)-exp(3*x)'
 */
export function rawToPython(expr: string): string {
  let s = expr.trim();

  // Step 1: Convert ^ to **
  s = s.replace(/\^/g, '**');

  // Step 2: Handle implicit multiplication
  // 2x -> 2*x
  s = s.replace(/(\d)([a-zA-Z])/g, '$1*$2');
  // x(x+1) -> x*(x+1), )(-> )*(
  s = s.replace(/([a-zA-Z])\(/g, '$1*(');
  // )( -> )*(
  s = s.replace(/\)\(/g, ')*(');
  // )digit -> )*digit  
  s = s.replace(/\)(\d)/g, ')*$1');

  // Step 3: e^... or e**... -> exp(...)
  s = s.replace(/e\*\*(.+)/g, 'exp($1)');

  // Step 4: Normalize log10 -> log (Python math uses log for base-10 in some contexts,
  // but SymPy uses log for natural log and log(x, 10) for base-10)
  // Keep as is for SymPy compatibility

  return s;
}

/**
 * Chuyển LaTeX sang Python eval-compatible expression.
 * VD: 'x^2 + y^2 - 4' -> 'x**2 + y**2 - 4'
 */
export function latexToPython(latex: string): string {
  let expr = latex.trim();

  // Remove \left and \right
  expr = expr.replace(/\\left\b/g, '');
  expr = expr.replace(/\\right\b/g, '');

  // Remove display math delimiters
  expr = expr.replace(/^\$\$|\$\$$/g, '');
  expr = expr.replace(/^\\\[|\\\]$/g, '');

  // Convert \sin -> sin, \cos -> cos, etc.
  const latexFuncs: Record<string, string> = {
    '\\\\sin': 'sin', '\\\\cos': 'cos', '\\\\tan': 'tan',
    '\\\\cot': 'cot', '\\\\arcsin': 'arcsin', '\\\\arccos': 'arccos',
    '\\\\arctan': 'arctan', '\\\\sinh': 'sinh', '\\\\cosh': 'cosh',
    '\\\\tanh': 'tanh', '\\\\ln': 'ln', '\\\\log': 'log10',
    '\\\\sqrt': 'sqrt', '\\\\exp': 'exp',
  };
  for (const [latexFn, pyFn] of Object.entries(latexFuncs)) {
    expr = expr.replace(new RegExp(latexFn + '\\b', 'g'), pyFn);
  }

  // Convert \pi -> pi
  expr = expr.replace(/\\pi\b/g, 'pi');

  // Convert e^{...} or e^... -> exp(...)
  expr = expr.replace(/e\^{([^}]+)}/g, 'exp($1)');
  expr = expr.replace(/e\^(\w+|\([^)]+\))/g, 'exp($1)');

  // Convert ^{...} -> **(...)
  expr = expr.replace(/\^{([^}]+)}/g, '**($1)');

  // Convert ^digit -> **digit
  expr = expr.replace(/\^(\d+)/g, '**$1');

  // Convert ^(expr) -> **(expr) 
  expr = expr.replace(/\^\(([^)]+)\)/g, '**($1)');

  // Convert \cdot -> *
  expr = expr.replace(/\\cdot\b/g, '*');

  // Convert \times -> *
  expr = expr.replace(/\\times\b/g, '*');

  // Convert \frac{a}{b} -> (a)/(b)
  while (/\\frac/.test(expr)) {
    expr = expr.replace(/\\frac\{([^}]+)\}\{([^}]+)\}/, '($1)/($2)');
  }

  // Convert \left|...\right| -> abs(...)
  expr = expr.replace(/\\left\|/g, 'abs(');
  expr = expr.replace(/\\right\|/g, ')');

  // Convert \\ -> , (matrix row separator)
  expr = expr.replace(/\\\\/g, ', ');
  expr = expr.replace(/&/g, ', ');

  // Handle plain ^ (not part of ^{...}) - fallback conversion
  expr = expr.replace(/\^/g, '**');

  return expr;
}

/**
 * Kiểm tra xem chuỗi LaTeX có hợp lệ cơ bản không
 */
export function isValidLatex(latex: string): boolean {
  let count = 0;
  for (const ch of latex) {
    if (ch === '{') count++;
    else if (ch === '}') count--;
    if (count < 0) return false;
  }
  return count === 0;
}