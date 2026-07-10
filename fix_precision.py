"""Replace precision=4 -> precision=7 and round(v, 6) -> round(v, 7) in all Python files under backend/app."""
import glob, os, sys

BASE = os.path.join(os.path.dirname(__file__), 'backend', 'app')

# Also include schemas and routers
dirs_to_fix = [
    os.path.join(BASE, 'algorithms'),
    os.path.join(BASE, 'routers'),
    os.path.join(BASE, 'schemas'),
]

# Also include test files
dirs_to_fix.append(os.path.join(os.path.dirname(__file__), 'backend'))

files = []
for d in dirs_to_fix:
    for f in glob.glob(os.path.join(d, '*.py')):
        files.append(f)

# Also handle frontend TSX files
frontend_src = os.path.join(os.path.dirname(__file__), 'frontend', 'src')
for root, dirs, fnames in os.walk(frontend_src):
    for fname in fnames:
        if fname.endswith('.tsx') or fname.endswith('.ts'):
            files.append(os.path.join(root, fname))

print(f"Found {len(files)} files to process")

for fp in files:
    with open(fp, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    
    # Backend: precision=4 -> precision=7 (in _matrix_to_latex calls)
    content = content.replace('precision=4', 'precision=7')
    
    # Backend: round(..., 6) -> round(..., 7) 
    # But careful: we need to match round(expr, 6) patterns
    # Use regex-like approach: replace ', 6)' with ', 7)' 
    # Only in contexts where it's clearly a precision parameter
    
    prefix = "round("
    import re
    def replace_round_6(m):
        full = m.group(0)
        if full.endswith(', 6)'):
            return full[:-4] + ', 7)'
        return full
    
    content = re.sub(r'round\([^)]+,\s*6\)', replace_round_6, content)
    
    # Also handle toFixed(6) in TS/TSX files
    content = re.sub(r'toFixed\(6\)', 'toFixed(7)', content)
    
    # Handle safeToFixed default precision: precision: number = 6 or precision: number=6
    content = re.sub(r'(precision:\s*number\s*=\s*)6(\b)', r'\g<1>7\g<2>', content)
    
    # Handle renderMatrix(precision?: number = 6) type patterns  
    # Actually safeToFixed and renderMatrix in SolutionSteps.tsx
    
    if content != original:
        with open(fp, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  MODIFIED: {fp}")
    else:
        print(f"  no change: {fp}")

print("\nDone!")