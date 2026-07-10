import { useState, useEffect } from 'react';
import {
  Container, Typography, TextField, Select, MenuItem, FormControl,
  InputLabel, Button, Grid, Paper, Box, CircularProgress, Alert, Chip, Stack,
} from '@mui/material';
import { linearSystemAPI, type LinearSystemResponse, type MatrixProperties } from '../services/api';
import MatrixLatexEditor from '../components/MatrixLatexEditor';
import FormulaRenderer from '../components/FormulaRenderer';
import IterationTable from '../components/IterationTable';
import ResultCard from '../components/ResultCard';
import SolutionSteps from '../components/SolutionSteps';

const allMethods = [
  { value: 'gaussian', label: 'Gaussian Elimination', group: 'direct', requiresSquare: false },
  { value: 'gauss_jordan', label: 'Gauss-Jordan Elimination', group: 'direct', requiresSquare: false },
  { value: 'lu', label: 'LU Decomposition', group: 'direct', requiresSquare: true },
  { value: 'cholesky', label: 'Cholesky Decomposition', group: 'direct', requiresSquare: true },
  { value: 'jacobi', label: 'Jacobi', group: 'iterative', requiresSquare: true },
  { value: 'gauss_seidel', label: 'Gauss-Seidel', group: 'iterative', requiresSquare: true },
  { value: 'simple_iteration', label: 'Simple Iteration (Lặp đơn)', group: 'iterative_simple', requiresSquare: true },
  { value: 'seidel', label: 'Seidel Iteration', group: 'iterative_simple', requiresSquare: true },
];

const isIterative = (method: string) => {
  const g = allMethods.find(m => m.value === method)?.group;
  return g === 'iterative' || g === 'iterative_simple';
};
const isDirect = (method: string) => allMethods.find(m => m.value === method)?.group === 'direct';
const isIterativeSimple = (method: string) => allMethods.find(m => m.value === method)?.group === 'iterative_simple';

function makeDefaultAMatrix(m: number, n: number): number[][] {
  if (m === 3 && n === 3) return [[10, -1, 2], [-1, 11, -1], [2, -1, 10]];
  return Array.from({ length: m }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 10 : 0)));
}

function makeDefaultBMatrix(m: number, p: number): number[][] {
  if (m === 3 && p === 1) return [[6], [25], [-11]];
  return Array.from({ length: m }, () => Array(p).fill(0));
}

export default function LinearSystem() {
  const [rows, setRows] = useState(3);
  const [aCols, setACols] = useState(3);
  const [bCols, setBCols] = useState(1);
  const [matrix, setMatrix] = useState<number[][]>(makeDefaultAMatrix(3, 3));
  const [bMatrix, setBMatrix] = useState<number[][]>(makeDefaultBMatrix(3, 1));
  const [method, setMethod] = useState('gaussian');
  const [epsilon, setEpsilon] = useState('0.000001');
  const [maxIter, setMaxIter] = useState('100');
  const [initialGuessStr, setInitialGuessStr] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<LinearSystemResponse | null>(null);
  const [matrixProps, setMatrixProps] = useState<MatrixProperties | null>(null);

  const isSquare = rows === aCols;
  const selectedMethodInfo = allMethods.find(m => m.value === method);
  const methodBlocked = selectedMethodInfo?.requiresSquare && !isSquare;
  const showIterativeParams = isIterative(method);
  const iterativeBBlocked = isIterative(method) && bCols > 1;

  useEffect(() => {
    if (matrix.length > 0 && matrix[0]?.length > 0) {
      const timer = setTimeout(() => {
        linearSystemAPI.properties(matrix).then(res => setMatrixProps(res.data)).catch(() => {});
      }, 500);
      return () => clearTimeout(timer);
    }
  }, [matrix]);

  const handleRowsChange = (newRows: number) => {
    if (newRows < 1 || newRows > 10) return;
    setRows(newRows);
    setMatrix(prev => Array.from({ length: newRows }, (_, i) =>
      Array.from({ length: aCols }, (_, j) => (i < prev.length && j < (prev[i]?.length ?? 0)) ? prev[i][j] : 0)));
    setBMatrix(prev => Array.from({ length: newRows }, (_, i) =>
      Array.from({ length: bCols }, (_, j) => (i < prev.length && j < (prev[i]?.length ?? 0)) ? prev[i][j] : 0)));
  };

  const handleAColsChange = (newCols: number) => {
    if (newCols < 1 || newCols > 10) return;
    setACols(newCols);
    setMatrix(prev => Array.from({ length: rows }, (_, i) =>
      Array.from({ length: newCols }, (_, j) => (i < prev.length && j < (prev[i]?.length ?? 0)) ? prev[i][j] : 0)));
  };

  const handleBColsChange = (newCols: number) => {
    if (newCols < 1 || newCols > 10) return;
    setBCols(newCols);
    setBMatrix(prev => Array.from({ length: rows }, (_, i) =>
      Array.from({ length: newCols }, (_, j) => (i < prev.length && j < (prev[i]?.length ?? 0)) ? prev[i][j] : 0)));
  };

  const handleSolve = async () => {
    if (methodBlocked || iterativeBBlocked) return;
    setLoading(true);
    try {
      const eps = parseFloat(epsilon) || 1e-6;
      const maxIt = parseInt(maxIter) || 100;
      const initial = initialGuessStr
        ? initialGuessStr.split(',').map(s => parseFloat(s.trim())).filter(v => !isNaN(v))
        : undefined;
      const response = await linearSystemAPI.solve({
        A: matrix, B: bMatrix, method,
        epsilon: eps, max_iterations: maxIt,
        initial_guess: initial,
      });
      setResult(response.data);
    } catch {
      setResult({ success: false, message: 'Lỗi kết nối server.', solution: [], steps: [], formula: '', iterations: [], iterations_count: 0, final_error: 0 });
    }
    setLoading(false);
  };

  const renderSolutionMatrix = (sol: number[][]) => {
    if (!sol || sol.length === 0) return null;
    const nCols = sol[0]?.length ?? 0;
    if (nCols === 1) {
      return <Typography variant="h5" sx={{ fontWeight: 600, color: 'primary.main' }}>
        X = ({sol.map(row => row[0].toFixed(7)).join(', ')})
      </Typography>;
    }
    return <Box>
      <Typography variant="h5" sx={{ fontWeight: 600, color: 'primary.main', mb: 1 }}>X ({sol.length}×{nCols}) =</Typography>
      <Box sx={{ fontFamily: 'monospace', fontSize: '0.95rem', overflowX: 'auto' }}>
        {sol.map((row, ri) => <Box key={ri}>| {row.map((v: number) => v.toFixed(7).padStart(12)).join(' ')} |</Box>)}
      </Box>
    </Box>;
  };

  return (
    <Container maxWidth="lg">
      <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>📐 Hệ phương trình đại số tuyến tính</Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>8 phương pháp giải AX = B</Typography>

      <Paper sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={3}>
          <Grid size={{ xs: 12, md: 4 }}>
            <FormControl fullWidth>
              <InputLabel>Phương pháp</InputLabel>
              <Select value={method} label="Phương pháp" onChange={(e) => setMethod(e.target.value)}>
                {allMethods.map(m => <MenuItem key={m.value} value={m.value}>{m.label}{m.requiresSquare && !isSquare ? ' (yêu cầu vuông)' : ''}</MenuItem>)}
              </Select>
            </FormControl>
          </Grid>

          {showIterativeParams && (<>
            <Grid size={{ xs: 6, md: 2 }}><TextField fullWidth type="text" inputMode="decimal" label="Epsilon" value={epsilon} onChange={(e) => setEpsilon(e.target.value)} size="small" /></Grid>
            <Grid size={{ xs: 6, md: 2 }}><TextField fullWidth type="text" inputMode="numeric" label="Max Iter" value={maxIter} onChange={(e) => setMaxIter(e.target.value)} size="small" /></Grid>
            <Grid size={{ xs: 12, md: 4 }}><TextField fullWidth type="text" label="Initial guess (cách nhau bởi dấu ,)" value={initialGuessStr} onChange={(e) => setInitialGuessStr(e.target.value)} helperText={`VD: ${Array(aCols).fill('0').join(', ')}`} size="small" /></Grid>
          </>)}

          {methodBlocked && <Grid size={{ xs: 12 }}><Alert severity="warning">{selectedMethodInfo?.label} yêu cầu ma trận vuông (n×n). Hiện tại A là {rows}×{aCols}.</Alert></Grid>}
          {showIterativeParams && !isSquare && <Grid size={{ xs: 12 }}><Alert severity="info">Phương pháp lặp chỉ hỗ trợ ma trận vuông.</Alert></Grid>}
          {iterativeBBlocked && <Grid size={{ xs: 12 }}><Alert severity="warning">Phương pháp lặp chỉ hỗ trợ B có 1 cột. Hiện tại B có {bCols} cột.</Alert></Grid>}

          <Grid size={{ xs: 12 }}>
            <MatrixLatexEditor matrix={matrix} bMatrix={bMatrix} onMatrixChange={setMatrix} onBMatrixChange={setBMatrix}
              rows={rows} aCols={aCols} bCols={bCols} onRowsChange={handleRowsChange} onAColsChange={handleAColsChange}
              onBColsChange={handleBColsChange} label="A × X = B" requireSquare={selectedMethodInfo?.requiresSquare ?? false}
              squareWarningMessage={selectedMethodInfo?.requiresSquare ? `${selectedMethodInfo?.label} yêu cầu ma trận vuông.` : ''} />
          </Grid>

          {matrixProps && <Grid size={{ xs: 12 }}>
            <Paper variant="outlined" sx={{ p: 2, bgcolor: 'action.hover' }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>🔍 Phân tích ma trận tự động</Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 1 }}>
                <Chip size="small" label={matrixProps.is_square ? "Vuông" : "Không vuông"} color={matrixProps.is_square ? "success" : "warning"} variant="outlined" />
                {matrixProps.is_square && (<>
                  <Chip size="small" label="Đối xứng" color={matrixProps.is_symmetric ? "success" : "default"} variant="outlined" />
                  <Chip size="small" label="Xác định dương" color={matrixProps.is_positive_definite ? "success" : "default"} variant="outlined" />
                  <Chip size="small" label="Ba đường chéo" color={matrixProps.is_tridiagonal ? "success" : "default"} variant="outlined" />
                  <Chip size="small" label="Chéo trội" color={matrixProps.is_diagonally_dominant_strict ? "success" : "default"} variant="outlined" />
                </>)}
              </Box>
              {matrixProps.recommendations.length > 0 && <Box>{matrixProps.recommendations.map((rec, idx) => <Typography key={idx} variant="body2" sx={{ color: 'info.main', fontWeight: 500 }}>💡 {rec}</Typography>)}</Box>}
            </Paper>
          </Grid>}

          <Grid size={{ xs: 12 }}>
            <Button variant="contained" size="large" onClick={handleSolve}
              disabled={loading || methodBlocked || iterativeBBlocked || (showIterativeParams && !isSquare)}
              startIcon={loading ? <CircularProgress size={20} /> : null}>Giải hệ phương trình</Button>
          </Grid>
        </Grid>
      </Paper>

      {result && (<>
        {result.solution_type === 'unique' && result.solution && result.solution.length > 0 && <ResultCard success={result.success} message={result.message}>
          <Box sx={{ mt: 1 }}>{renderSolutionMatrix(result.solution)}{result.execution_time && <Typography variant="body2" color="text.secondary">⏱ {result.execution_time.toFixed(7)}s</Typography>}</Box>
        </ResultCard>}
        {result.solution_type === 'inconsistent' && <ResultCard success={false} message={result.message}>
          <Paper variant="outlined" sx={{ p: 2, mt: 2, bgcolor: 'error.light', color: 'error.contrastText' }}>
            <Typography variant="h6" sx={{ fontWeight: 700 }}>⚠️ Hệ vô nghiệm</Typography>
            <Typography sx={{ mt: 1 }}>rank(A) = {result.rank_A} {'<'} rank([A|B]) = {result.rank_augmented}</Typography>
          </Paper>
        </ResultCard>}
        {result.solution_type === 'infinite' && <ResultCard success={true} message={result.message}>
          <Stack spacing={2} sx={{ mt: 1 }}>
            <Paper variant="outlined" sx={{ p: 2 }}><Typography variant="subtitle1" sx={{ fontWeight: 600 }}>📊 Phân tích hệ</Typography><Typography sx={{ mt: 1 }}>rank(A) = {result.rank_A} = rank([A|B]) = {result.rank_augmented} {'<'} số ẩn n = {result.analysis ? String((result.analysis as Record<string, unknown>).n) : '?'}</Typography></Paper>
            {result.free_variables && result.free_variables.length > 0 && <Paper variant="outlined" sx={{ p: 2 }}><Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>🔓 Biến tự do</Typography><Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>{result.free_variables.map((fv, idx) => <Chip key={idx} label={`${fv} = c${idx + 1}`} color="secondary" variant="outlined" />)}</Box></Paper>}
            {result.particular_solution && result.particular_solution.length > 0 && <Paper variant="outlined" sx={{ p: 2 }}><Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>📌 Nghiệm riêng x<sub>p</sub></Typography><Box sx={{ fontFamily: 'monospace', fontSize: '0.9rem', overflowX: 'auto' }}>{(result.particular_solution[0] as number[]).map((_, ci) => <Box key={ci} sx={{ mb: 1 }}><Typography variant="body2" sx={{ fontWeight: 600 }}>Cột {ci + 1}: [{(result.particular_solution as number[][]).map(row => row[ci].toFixed(7)).join(', ')}]<sup>T</sup></Typography></Box>)}</Box></Paper>}
            {result.basis_vectors && result.basis_vectors.length > 0 && <Paper variant="outlined" sx={{ p: 2 }}><Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>🧱 Vector cơ sở</Typography>{result.basis_vectors.map((bv, idx) => <Box key={idx} sx={{ mb: 1.5 }}><Typography variant="body2" sx={{ fontWeight: 600 }}>v<sub>{idx + 1}</sub> = [{bv.map(v => v.toFixed(7)).join(', ')}]<sup>T</sup></Typography><Box sx={{ fontFamily: 'monospace', fontSize: '0.9rem', borderLeft: '2px solid', borderRight: '2px solid', borderColor: 'primary.main', display: 'inline-block', px: 1, py: 0.5, lineHeight: 1.5 }}>{bv.map((v, ri) => <Box key={ri}>{v.toFixed(7)}</Box>)}</Box></Box>)}</Paper>}
            {result.general_solution_latex && <Paper variant="outlined" sx={{ p: 2 }}><Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>🔢 Nghiệm tổng quát</Typography><FormulaRenderer latex={result.general_solution_latex} /></Paper>}
          </Stack>
        </ResultCard>}
        {isIterative(method) && result.iterations && result.iterations.length > 0 && result.solution && result.solution.length > 0 && <ResultCard success={result.success} message={result.message}>
          <Box sx={{ mt: 1 }}>{renderSolutionMatrix(result.solution)}<Typography>Số vòng lặp: {result.iterations_count} | Sai số: {result.final_error.toExponential(4)}</Typography>{result.diagonally_dominant === false && <Alert severity="warning" sx={{ mt: 1 }}>Ma trận không chéo trội — hội tụ có thể chậm hoặc không hội tụ.</Alert>}</Box>
        </ResultCard>}
        {result.formula && <Box sx={{ mt: 3 }}><Typography variant="h6">Công thức:</Typography><FormulaRenderer latex={result.formula} /></Box>}
        {isIterative(method) && result.iterations && result.iterations.length > 0 && <IterationTable data={result.iterations} title="Bảng quá trình lặp" />}
        {result.steps && result.steps.length > 0 && <SolutionSteps steps={result.steps as any} method={method} />}
      </>)}
    </Container>
  );
}