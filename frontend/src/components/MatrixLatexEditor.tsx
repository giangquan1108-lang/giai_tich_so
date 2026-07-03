import { useState, useRef, useEffect, useCallback } from 'react';
import {
  Box, TextField, Typography, IconButton, Tooltip, Select, MenuItem,
  FormControl, InputLabel, Paper, Chip, Divider, Alert,
} from '@mui/material';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import ClearIcon from '@mui/icons-material/Clear';
import GridOnIcon from '@mui/icons-material/GridOn';

interface MatrixLatexEditorProps {
  matrix: number[][];
  /** B is now an m×p matrix (not a vector). Pass [] for no B display (e.g. inverse tab). */
  bMatrix: number[][];
  onMatrixChange: (matrix: number[][]) => void;
  onBMatrixChange: (bMatrix: number[][]) => void;
  rows: number;   // m — rows of A (also rows of B)
  aCols: number;  // n — columns of A
  bCols: number;  // p — columns of B
  onRowsChange?: (rows: number) => void;
  onAColsChange?: (cols: number) => void;
  onBColsChange?: (cols: number) => void;
  label?: string;
  /** If true, show a warning when rows !== aCols (square matrix required) */
  requireSquare?: boolean;
  squareWarningMessage?: string;
  /** If true, hide the B matrix entirely (e.g. inverse tab) */
  hideB?: boolean;
}

export default function MatrixLatexEditor({
  matrix, bMatrix, onMatrixChange, onBMatrixChange,
  rows: m, aCols, bCols,
  onRowsChange, onAColsChange, onBColsChange,
  label = 'Ma trận',
  requireSquare = false,
  squareWarningMessage = '',
  hideB = false,
}: MatrixLatexEditorProps) {
  const previewRef = useRef<HTMLDivElement>(null);

  // Refs for keyboard navigation — 2D arrays
  const aRefs = useRef<(HTMLInputElement | null)[][]>([]);
  const bRefs = useRef<(HTMLInputElement | null)[][]>([]);

  // Ensure refs arrays are sized correctly
  if (!aRefs.current || aRefs.current.length !== m) {
    aRefs.current = Array.from({ length: m }, () => Array(aCols).fill(null));
  }
  for (let i = 0; i < m; i++) {
    if (!aRefs.current[i] || aRefs.current[i].length !== aCols) {
      aRefs.current[i] = Array(aCols).fill(null);
    }
  }
  if (!bRefs.current || bRefs.current.length !== m) {
    bRefs.current = Array.from({ length: m }, () => Array(bCols).fill(null));
  }
  for (let i = 0; i < m; i++) {
    if (!bRefs.current[i] || bRefs.current[i].length !== bCols) {
      bRefs.current[i] = Array(bCols).fill(null);
    }
  }

  const focusACell = useCallback((r: number, c: number) => {
    const el = aRefs.current[r]?.[c];
    if (el) {
      el.focus();
      el.select();
    }
  }, []);

  const focusBCell = useCallback((r: number, c: number) => {
    const el = bRefs.current[r]?.[c];
    if (el) {
      el.focus();
      el.select();
    }
  }, []);

  // Keyboard handler for matrix A
  const handleAKeyDown = (r: number, c: number, e: React.KeyboardEvent) => {
    let handled = true;
    switch (e.key) {
      case 'ArrowLeft':
        if (c > 0) focusACell(r, c - 1);
        else if (r > 0) focusACell(r - 1, aCols - 1);
        break;
      case 'ArrowRight':
        if (c < aCols - 1) focusACell(r, c + 1);
        else if (r < m - 1) focusACell(r + 1, 0);
        break;
      case 'ArrowUp':
        if (r > 0) focusACell(r - 1, c);
        else if (c > 0) focusACell(0, c - 1);
        break;
      case 'ArrowDown':
        if (r < m - 1) focusACell(r + 1, c);
        else if (c < aCols - 1) focusACell(m - 1, c + 1);
        break;
      case 'Enter':
        if (r < m - 1) focusACell(r + 1, c);
        else if (c < aCols - 1) focusACell(0, c + 1);
        break;
      case 'Tab':
        if (e.shiftKey) {
          // Shift+Tab: move left, or to last B cell
          if (c > 0) focusACell(r, c - 1);
          else if (r > 0) focusACell(r - 1, aCols - 1);
          else if (!hideB && bCols > 0) focusBCell(m - 1, bCols - 1);
        } else {
          // Tab: move right, or to first B cell
          if (c < aCols - 1) focusACell(r, c + 1);
          else if (r < m - 1) focusACell(r + 1, 0);
          else if (!hideB && bCols > 0) focusBCell(0, 0);
        }
        break;
      default:
        handled = false;
    }
    if (handled) {
      e.preventDefault();
      e.stopPropagation();
    }
  };

  // Keyboard handler for matrix B
  const handleBKeyDown = (r: number, c: number, e: React.KeyboardEvent) => {
    let handled = true;
    switch (e.key) {
      case 'ArrowLeft':
        if (c > 0) focusBCell(r, c - 1);
        else if (r > 0) focusBCell(r - 1, bCols - 1);
        break;
      case 'ArrowRight':
        if (c < bCols - 1) focusBCell(r, c + 1);
        else if (r < m - 1) focusBCell(r + 1, 0);
        break;
      case 'ArrowUp':
        if (r > 0) focusBCell(r - 1, c);
        else if (c > 0) focusBCell(0, c - 1);
        break;
      case 'ArrowDown':
        if (r < m - 1) focusBCell(r + 1, c);
        else if (c < bCols - 1) focusBCell(m - 1, c + 1);
        break;
      case 'Enter':
        if (r < m - 1) focusBCell(r + 1, c);
        else if (c < bCols - 1) focusBCell(0, c + 1);
        break;
      case 'Tab':
        if (e.shiftKey) {
          // Shift+Tab: move left, or to last A cell
          if (c > 0) focusBCell(r, c - 1);
          else if (r > 0) focusBCell(r - 1, bCols - 1);
          else focusACell(m - 1, aCols - 1);
        } else {
          // Tab: move right, stays within B
          if (c < bCols - 1) focusBCell(r, c + 1);
          else if (r < m - 1) focusBCell(r + 1, 0);
        }
        break;
      default:
        handled = false;
    }
    if (handled) {
      e.preventDefault();
      e.stopPropagation();
    }
  };

  // String state for A
  const [aStrings, setAStrings] = useState<string[][]>(() =>
    Array.from({ length: m }, (_, i) =>
      Array.from({ length: aCols }, (_, j) =>
        i < matrix.length && j < (matrix[i]?.length ?? 0) ? String(matrix[i][j]) : '0'
      )
    )
  );
  // String state for B
  const [bStrings, setBStrings] = useState<string[][]>(() =>
    Array.from({ length: m }, (_, i) =>
      Array.from({ length: bCols }, (_, j) =>
        i < bMatrix.length && j < (bMatrix[i]?.length ?? 0) ? String(bMatrix[i][j]) : '0'
      )
    )
  );

  // Re-init string state when dimensions change
  useEffect(() => {
    setAStrings(
      Array.from({ length: m }, (_, i) =>
        Array.from({ length: aCols }, (_, j) =>
          i < matrix.length && j < (matrix[i]?.length ?? 0) ? String(matrix[i][j]) : '0'
        )
      )
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [m, aCols]);

  useEffect(() => {
    setBStrings(
      Array.from({ length: m }, (_, i) =>
        Array.from({ length: bCols }, (_, j) =>
          i < bMatrix.length && j < (bMatrix[i]?.length ?? 0) ? String(bMatrix[i][j]) : '0'
        )
      )
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [m, bCols]);

  const isSquare = m === aCols;

  const matrixLatex = matrix.length > 0 && matrix[0]?.length > 0 && bMatrix.length > 0 && bMatrix[0]?.length > 0
    ? `A_{${m}\\times ${aCols}} = \\begin{pmatrix} ${matrix.map(r => r.join(' & ')).join(' \\\\ ')} \\end{pmatrix}, \\quad B_{${m}\\times ${bCols}} = \\begin{pmatrix} ${bMatrix.map(r => r.join(' & ')).join(' \\\\ ')} \\end{pmatrix}`
    : matrix.length > 0 && matrix[0]?.length > 0
    ? `A_{${m}\\times ${aCols}} = \\begin{pmatrix} ${matrix.map(r => r.join(' & ')).join(' \\\\ ')} \\end{pmatrix}`
    : '';

  useEffect(() => {
    if (!previewRef.current || !matrixLatex) {
      if (previewRef.current) previewRef.current.innerHTML = '';
      return;
    }
    previewRef.current.innerHTML = `\\[${matrixLatex}\\]`;
    if ((window as any).MathJax?.typesetPromise) {
      (window as any).MathJax.typesetPromise([previewRef.current]).catch(() => {});
    }
  }, [matrixLatex]);

  const parseNumber = (value: string): number => {
    const trimmed = value.trim();
    if (trimmed === '' || trimmed === '-' || trimmed === '.') return 0;
    const parsed = parseFloat(trimmed);
    return isNaN(parsed) ? 0 : parsed;
  };

  const handleAChange = (r: number, c: number, value: string) => {
    setAStrings(prev => {
      const ns = prev.map(row => [...row]);
      while (ns.length < m) ns.push(Array(aCols).fill('0'));
      for (let i = 0; i < m; i++) {
        while (ns[i].length < aCols) ns[i].push('0');
      }
      ns[r][c] = value;
      return ns;
    });

    const newMatrix = matrix.map(row => [...row]);
    while (newMatrix.length < m) newMatrix.push(Array(aCols).fill(0));
    for (let i = 0; i < m; i++) {
      while (newMatrix[i].length < aCols) newMatrix[i].push(0);
    }
    newMatrix[r][c] = parseNumber(value);
    onMatrixChange(newMatrix);
  };

  const handleBChange = (r: number, c: number, value: string) => {
    setBStrings(prev => {
      const ns = prev.map(row => [...row]);
      while (ns.length < m) ns.push(Array(bCols).fill('0'));
      for (let i = 0; i < m; i++) {
        while (ns[i].length < bCols) ns[i].push('0');
      }
      ns[r][c] = value;
      return ns;
    });

    const newB = bMatrix.map(row => [...row]);
    while (newB.length < m) newB.push(Array(bCols).fill(0));
    for (let i = 0; i < m; i++) {
      while (newB[i].length < bCols) newB[i].push(0);
    }
    newB[r][c] = parseNumber(value);
    onBMatrixChange(newB);
  };

  const handleClear = () => {
    onMatrixChange(Array.from({ length: m }, () => Array(aCols).fill(0)));
    onBMatrixChange(Array.from({ length: m }, () => Array(bCols).fill(0)));
    setAStrings(Array.from({ length: m }, () => Array(aCols).fill('0')));
    setBStrings(Array.from({ length: m }, () => Array(bCols).fill('0')));
  };

  const CELL_WIDTH = aCols <= 4 && bCols <= 2 ? 80 : m <= 4 ? 72 : 64;
  const dimLabel = `A:${m}×${aCols} B:${m}×${bCols}`;

  return (
    <Paper variant="outlined" sx={{ p: 2, mb: 2 }}>
      {/* Header row */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <GridOnIcon color="primary" fontSize="small" />
          <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>{label}</Typography>
          <Chip label={dimLabel} size="small" color={isSquare ? 'primary' : 'warning'} variant="outlined" />
        </Box>
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
          {onRowsChange && (
            <FormControl size="small" sx={{ minWidth: 80 }}>
              <InputLabel>Rows(A)</InputLabel>
              <Select value={m} label="Rows(A)" onChange={(e) => onRowsChange(Number(e.target.value))}>
                {[2, 3, 4, 5, 6, 7, 8].map(v => <MenuItem key={v} value={v}>{v}</MenuItem>)}
              </Select>
            </FormControl>
          )}
          {onAColsChange && (
            <FormControl size="small" sx={{ minWidth: 80 }}>
              <InputLabel>Cols(A)</InputLabel>
              <Select value={aCols} label="Cols(A)" onChange={(e) => onAColsChange(Number(e.target.value))}>
                {[2, 3, 4, 5, 6, 7, 8].map(v => <MenuItem key={v} value={v}>{v}</MenuItem>)}
              </Select>
            </FormControl>
          )}
          {!hideB && onBColsChange && (
            <FormControl size="small" sx={{ minWidth: 80 }}>
              <InputLabel>Cols(B)</InputLabel>
              <Select value={bCols} label="Cols(B)" onChange={(e) => onBColsChange(Number(e.target.value))}>
                {[1, 2, 3, 4, 5, 6, 7, 8].map(v => <MenuItem key={v} value={v}>{v}</MenuItem>)}
              </Select>
            </FormControl>
          )}
          <Tooltip title="Đặt lại về 0">
            <IconButton size="small" onClick={handleClear}><ClearIcon fontSize="small" /></IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Square matrix warning */}
      {requireSquare && !isSquare && squareWarningMessage && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          {squareWarningMessage}
        </Alert>
      )}

      <Divider sx={{ mb: 2 }} />

      {/* Equation layout: A × X = B */}
      <Box sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'center', gap: 1, overflowX: 'auto' }}>
        {/* === Matrix A === */}
        <Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.5 }}>
            <Typography sx={{ fontWeight: 600, color: 'primary.main', fontSize: '0.9rem' }}>A</Typography>
            <Chip label={`${m}×${aCols}`} size="small" variant="outlined" sx={{ fontSize: '0.65rem', height: 20 }} />
          </Box>
          {/* Column indices */}
          <Box sx={{ display: 'flex', gap: 0.5, mb: 0.5 }}>
            <Box sx={{ width: 26 }} />
            {Array.from({ length: aCols }, (_, j) => (
              <Typography key={j} sx={{ width: CELL_WIDTH, textAlign: 'center', color: 'text.secondary', fontSize: '0.7rem' }}>
                col{j + 1}
              </Typography>
            ))}
          </Box>
          {Array.from({ length: m }, (_, i) => (
            <Box key={i} sx={{ display: 'flex', gap: 0.5, mb: 0.5, alignItems: 'center' }}>
              <Typography sx={{ width: 26, textAlign: 'center', color: 'text.secondary', fontSize: '0.75rem' }}>
                {i + 1}
              </Typography>
              {Array.from({ length: aCols }, (_, j) => (
                <TextField
                  key={j} size="small" type="text" inputMode="decimal"
                  value={aStrings[i]?.[j] ?? '0'}
                  onChange={(e) => handleAChange(i, j, e.target.value)}
                  onKeyDown={(e) => handleAKeyDown(i, j, e)}
                  inputRef={(el: HTMLInputElement | null) => { aRefs.current[i][j] = el; }}
                  sx={{ width: CELL_WIDTH }}
                  slotProps={{
                    input: { style: { textAlign: 'center', fontSize: '0.88rem', fontWeight: 500 } },
                  }}
                />
              ))}
            </Box>
          ))}
        </Box>

        {/* × symbol */}
        <Box sx={{ display: 'flex', alignItems: 'center', height: m * 44 + 50, pt: 2 }}>
          <Typography variant="h5" sx={{ fontWeight: 700, color: 'text.secondary', mx: 0.5 }}>×</Typography>
        </Box>

        {/* === X (unknowns) === */}
        <Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.5 }}>
            <Typography sx={{ fontWeight: 600, color: 'success.main', fontSize: '0.9rem' }}>X</Typography>
            <Chip label={`${aCols}×${bCols}`} size="small" variant="outlined" sx={{ fontSize: '0.65rem', height: 20, color: 'success.main', borderColor: 'success.main' }} />
          </Box>
          <Box sx={{ display: 'flex', gap: 0.5, mb: 0.5 }}>
            <Box sx={{ width: 26 }} />
            {Array.from({ length: bCols }, (_, j) => (
              <Typography key={j} sx={{ width: CELL_WIDTH, textAlign: 'center', color: 'success.main', fontSize: '0.7rem' }}>
                col{j + 1}
              </Typography>
            ))}
          </Box>
          {Array.from({ length: aCols }, (_, i) => (
            <Box key={i} sx={{ display: 'flex', gap: 0.5, mb: 0.5, alignItems: 'center' }}>
              <Typography sx={{ width: 26, textAlign: 'center', color: 'text.secondary', fontSize: '0.75rem' }}>
                {i + 1}
              </Typography>
              {Array.from({ length: bCols }, (_, j) => (
                <Paper
                  key={j}
                  variant="outlined"
                  sx={{
                    width: CELL_WIDTH, height: 40, display: 'flex', alignItems: 'center', justifyContent: 'center',
                    bgcolor: 'action.hover', borderColor: 'success.light',
                  }}
                >
                  <Typography sx={{ fontSize: '0.88rem', fontStyle: 'italic', color: 'success.main', fontWeight: 500 }}>
                    x{i + 1}{j + 1}
                  </Typography>
                </Paper>
              ))}
            </Box>
          ))}
        </Box>

        {/* = symbol */}
        <Box sx={{ display: 'flex', alignItems: 'center', height: m * 44 + 50, pt: 2 }}>
          <Typography variant="h5" sx={{ fontWeight: 700, color: 'text.secondary', mx: 0.5 }}>=</Typography>
        </Box>

        {/* === Matrix B === */}
        {!hideB && (
          <Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.5 }}>
              <Typography sx={{ fontWeight: 600, color: 'secondary.main', fontSize: '0.9rem' }}>B</Typography>
              <Chip label={`${m}×${bCols}`} size="small" variant="outlined" sx={{ fontSize: '0.65rem', height: 20, color: 'secondary.main', borderColor: 'secondary.main' }} />
            </Box>
            <Box sx={{ display: 'flex', gap: 0.5, mb: 0.5 }}>
              <Box sx={{ width: 26 }} />
              {Array.from({ length: bCols }, (_, j) => (
                <Typography key={j} sx={{ width: CELL_WIDTH, textAlign: 'center', color: 'text.secondary', fontSize: '0.7rem' }}>
                  col{j + 1}
                </Typography>
              ))}
            </Box>
            {Array.from({ length: m }, (_, i) => (
              <Box key={i} sx={{ display: 'flex', gap: 0.5, mb: 0.5, alignItems: 'center' }}>
                <Typography sx={{ width: 26, textAlign: 'center', color: 'text.secondary', fontSize: '0.75rem' }}>
                  {i + 1}
                </Typography>
              {Array.from({ length: bCols }, (_, j) => (
                  <TextField
                    key={j} size="small" type="text" inputMode="decimal"
                    value={bStrings[i]?.[j] ?? '0'}
                    onChange={(e) => handleBChange(i, j, e.target.value)}
                    onKeyDown={(e) => handleBKeyDown(i, j, e)}
                    inputRef={(el: HTMLInputElement | null) => { bRefs.current[i][j] = el; }}
                    sx={{ width: CELL_WIDTH }}
                    slotProps={{
                      input: {
                        style: { textAlign: 'center', fontSize: '0.88rem', fontWeight: 500 },
                        sx: { bgcolor: 'action.hover' },
                      },
                    }}
                  />
                ))}
              </Box>
            ))}
          </Box>
        )}
      </Box>

      <Divider sx={{ my: 2 }} />

      {/* LaTeX Preview */}
      {matrixLatex && (
        <Box sx={{
          mt: 1, py: 1.5, textAlign: 'center',
          bgcolor: (t) => t.palette.mode === 'dark' ? 'grey.900' : 'grey.50',
          borderRadius: 1, fontSize: '1.1rem', overflow: 'auto',
          border: '1px dashed', borderColor: 'divider',
        }}>
          <Box ref={previewRef} sx={{ minHeight: 30 }} />
        </Box>
      )}

      <Box sx={{ mt: 1, display: 'flex', justifyContent: 'flex-end' }}>
        <Typography variant="caption" color="text.secondary" sx={{ mr: 1, alignSelf: 'center' }}>
          A({m}×{aCols}) · X({aCols}×{bCols}) = B({m}×{bCols})
        </Typography>
        <Tooltip title="Sao chép mã LaTeX">
          <IconButton size="small" onClick={() => navigator.clipboard.writeText(matrixLatex)}>
            <ContentCopyIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      </Box>
    </Paper>
  );
}