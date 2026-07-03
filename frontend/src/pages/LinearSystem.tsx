import { useState, useEffect } from 'react';
import {
  Container, Typography, TextField, Select, MenuItem, FormControl,
  InputLabel, Button, Grid, Paper, Box, CircularProgress, Alert, Chip,
  Stack, Tabs, Tab,
} from '@mui/material';
import { linearSystemAPI, type LinearSystemResponse, type MatrixProperties, type MatrixInverseResponse } from '../services/api';
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
  { value: 'thomas', label: 'Thomas Algorithm (TDMA)', group: 'direct', requiresSquare: true },
  { value: 'jacobi', label: 'Jacobi', group: 'iterative', requiresSquare: true },
  { value: 'gauss_seidel', label: 'Gauss-Seidel', group: 'iterative', requiresSquare: true },
  { value: 'sor', label: 'SOR (Successive Over-Relaxation)', group: 'iterative', requiresSquare: true },
];

const isIterative = (method: string) => allMethods.find(m => m.value === method)?.group === 'iterative';
const isDirect = (method: string) => allMethods.find(m => m.value === method)?.group === 'direct';

function makeDefaultAMatrix(m: number, n: number): number[][] {
  if (m === 3 && n === 3) {
    return [
      [10, -1, 2],
      [-1, 11, -1],
      [2, -1, 10],
    ];
  }
  return Array.from({ length: m }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 10 : 0))
  );
}

function makeDefaultBMatrix(m: number, p: number): number[][] {
  if (m === 3 && p === 1) return [[6], [25], [-11]];
  return Array.from({ length: m }, () => Array(p).fill(0));
}

export default function LinearSystem() {
  // ---- Tab state ----
  const [tabIndex, setTabIndex] = useState(0);

  // ---- Solve AX = B state ----
  const [rows, setRows] = useState(3);         // m = rows(A) = rows(B)
  const [aCols, setACols] = useState(3);       // n = cols(A)
  const [bCols, setBCols] = useState(1);       // p = cols(B)
  const [matrix, setMatrix] = useState<number[][]>(makeDefaultAMatrix(3, 3));
  const [bMatrix, setBMatrix] = useState<number[][]>(makeDefaultBMatrix(3, 1));
  const [method, setMethod] = useState('gaussian');
  const [epsilon, setEpsilon] = useState('0.000001');
  const [maxIter, setMaxIter] = useState('100');
  const [omega, setOmega] = useState('1.0');
  const [initialGuessStr, setInitialGuessStr] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<LinearSystemResponse | null>(null);

  // ---- Inverse state ----
  const [invSize, setInvSize] = useState(3);
  const [invMatrix, setInvMatrix] = useState<number[][]>([
    [1, 2, 0],
    [3, 4, 1],
    [0, 1, 2],
  ]);
  const [invMethod, setInvMethod] = useState('gauss_jordan');
  const [invLoading, setInvLoading] = useState(false);
  const [invResult, setInvResult] = useState<MatrixInverseResponse | null>(null);

  // ---- Shared ----
  const [matrixProps, setMatrixProps] = useState<MatrixProperties | null>(null);

  const isSquare = rows === aCols;
  const selectedMethodInfo = allMethods.find(m => m.value === method);
  const methodBlocked = selectedMethodInfo?.requiresSquare && !isSquare;
  const showIterativeParams = isIterative(method);
  // Iterative methods only support B(m×1)
  const iterativeBBlocked = isIterative(method) && bCols > 1;

  // Auto-analyze matrix properties (use whichever matrix is active)
  useEffect(() => {
    const activeMatrix = tabIndex === 0 ? matrix : invMatrix;
    if (activeMatrix.length > 0 && activeMatrix[0]?.length > 0) {
      const timer = setTimeout(() => {
        linearSystemAPI.properties(activeMatrix).then(res => setMatrixProps(res.data)).catch(() => {});
      }, 500);
      return () => clearTimeout(timer);
    }
  }, [matrix, invMatrix, tabIndex]);

  // ---- AX=B handlers ----
  const handleRowsChange = (newRows: number) => {
    if (newRows < 1 || newRows > 10) return;
    setRows(newRows);
    setMatrix(prev => {
      const newMatrix = Array.from({ length: newRows }, (_, i) =>
        Array.from({ length: aCols }, (_, j) =>
          (i < prev.length && j < (prev[i]?.length ?? 0)) ? prev[i][j] : 0
        )
      );
      return newMatrix;
    });
    setBMatrix(prev => {
      const newB = Array.from({ length: newRows }, (_, i) =>
        Array.from({ length: bCols }, (_, j) =>
          (i < prev.length && j < (prev[i]?.length ?? 0)) ? prev[i][j] : 0
        )
      );
      return newB;
    });
  };

  const handleAColsChange = (newCols: number) => {
    if (newCols < 1 || newCols > 10) return;
    setACols(newCols);
    setMatrix(prev => {
      const newMatrix = Array.from({ length: rows }, (_, i) =>
        Array.from({ length: newCols }, (_, j) =>
          (i < prev.length && j < (prev[i]?.length ?? 0)) ? prev[i][j] : 0
        )
      );
      return newMatrix;
    });
  };

  const handleBColsChange = (newCols: number) => {
    if (newCols < 1 || newCols > 10) return;
    setBCols(newCols);
    setBMatrix(prev => {
      const newB = Array.from({ length: rows }, (_, i) =>
        Array.from({ length: newCols }, (_, j) =>
          (i < prev.length && j < (prev[i]?.length ?? 0)) ? prev[i][j] : 0
        )
      );
      return newB;
    });
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
        omega: method === 'sor' ? parseFloat(omega) || 1.0 : undefined,
      });
      setResult(response.data);
    } catch {
      setResult({
        success: false, message: 'Lỗi kết nối server.',
        solution: [], steps: [], formula: '',
        iterations: [], iterations_count: 0, final_error: 0,
      });
    }
    setLoading(false);
  };

  // ---- Inverse handlers ----
  const handleInvSizeChange = (newSize: number) => {
    if (newSize < 1 || newSize > 10) return;
    setInvSize(newSize);
    setInvMatrix(prev => {
      const newMatrix = Array.from({ length: newSize }, (_, i) =>
        Array.from({ length: newSize }, (_, j) =>
          (i < prev.length && j < (prev[i]?.length ?? 0)) ? prev[i][j] : 0
        )
      );
      return newMatrix;
    });
  };

  const handleInverse = async () => {
    setInvLoading(true);
    try {
      const response = await linearSystemAPI.inverse(invMatrix, invMethod);
      setInvResult(response.data);
    } catch {
      setInvResult({ success: false, message: 'Lỗi kết nối server.' });
    }
    setInvLoading(false);
  };

  // ---- Render solution as n×p matrix ----
  const renderSolutionMatrix = (sol: number[][]) => {
    if (!sol || sol.length === 0) return null;
    const nRows = sol.length;
    const nCols = sol[0]?.length ?? 0;
    if (nCols === 1) {
      // Single column — show as vector
      return (
        <Typography variant="h5" sx={{ fontWeight: 600, color: 'primary.main' }}>
          X = ({sol.map(row => row[0].toFixed(6)).join(', ')})
        </Typography>
      );
    }
    return (
      <Box>
        <Typography variant="h5" sx={{ fontWeight: 600, color: 'primary.main', mb: 1 }}>
          X ({nRows}×{nCols}) =
        </Typography>
        <Box sx={{ fontFamily: 'monospace', fontSize: '0.95rem', overflowX: 'auto' }}>
          {sol.map((row, ri) => (
            <Box key={ri}>
              | {row.map((v: number) => v.toFixed(6).padStart(12)).join(' ')} |
            </Box>
          ))}
        </Box>
      </Box>
    );
  };

  return (
    <Container maxWidth="lg">
      <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
        📐 Hệ phương trình đại số tuyến tính
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        8 phương pháp giải AX = B + Tìm ma trận nghịch đảo
      </Typography>

      {/* ---- Tabs ---- */}
      <Tabs value={tabIndex} onChange={(_, v) => setTabIndex(v)} sx={{ mb: 3 }}>
        <Tab label="Giải hệ AX = B" />
        <Tab label="Ma trận nghịch đảo A⁻¹" />
      </Tabs>

      {/* ===== TAB 0: Giải hệ AX = B ===== */}
      {tabIndex === 0 && (
        <>
          <Paper sx={{ p: 3, mb: 3 }}>
            <Grid container spacing={3}>
              <Grid size={{ xs: 12, md: 4 }}>
                <FormControl fullWidth>
                  <InputLabel>Phương pháp</InputLabel>
                  <Select value={method} label="Phương pháp" onChange={(e) => setMethod(e.target.value)}>
                    {allMethods.map(m => (
                      <MenuItem key={m.value} value={m.value}>
                        {m.label}
                        {m.requiresSquare && !isSquare ? ' (yêu cầu vuông)' : ''}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>

              {showIterativeParams && (
                <>
                  <Grid size={{ xs: 6, md: 2 }}>
                    <TextField fullWidth type="text" inputMode="decimal" label="Epsilon" value={epsilon}
                      onChange={(e) => setEpsilon(e.target.value)} size="small" />
                  </Grid>
                  <Grid size={{ xs: 6, md: 2 }}>
                    <TextField fullWidth type="text" inputMode="numeric" label="Max Iter" value={maxIter}
                      onChange={(e) => setMaxIter(e.target.value)} size="small" />
                  </Grid>
                  {method === 'sor' && (
                    <Grid size={{ xs: 6, md: 2 }}>
                      <TextField fullWidth type="text" inputMode="decimal" label="Omega (ω)" value={omega}
                        onChange={(e) => setOmega(e.target.value)} size="small" />
                    </Grid>
                  )}
                  <Grid size={{ xs: 12, md: method === 'sor' ? 2 : 4 }}>
                    <TextField fullWidth type="text" inputMode="text" label="Initial guess (cách nhau bởi dấu ,)"
                      value={initialGuessStr}
                      onChange={(e) => setInitialGuessStr(e.target.value)}
                      helperText={`VD: ${Array.from({ length: aCols }, () => '0').join(', ')}`}
                      size="small" />
                  </Grid>
                </>
              )}

              {methodBlocked && (
                <Grid size={{ xs: 12 }}>
                  <Alert severity="warning">
                    {selectedMethodInfo?.label} yêu cầu ma trận vuông (n×n). 
                    Hiện tại A là {rows}×{aCols}. Vui lòng chọn phương pháp khác hoặc điều chỉnh kích thước.
                  </Alert>
                </Grid>
              )}

              {showIterativeParams && !isSquare && (
                <Grid size={{ xs: 12 }}>
                  <Alert severity="info">Phương pháp lặp chỉ hỗ trợ ma trận vuông. Vui lòng đặt số hàng = số cột.</Alert>
                </Grid>
              )}

              {iterativeBBlocked && (
                <Grid size={{ xs: 12 }}>
                  <Alert severity="warning">
                    Phương pháp lặp hiện chỉ hỗ trợ vector vế phải (B có 1 cột). 
                    Hiện tại B có {bCols} cột. Vui lòng đặt Cols(B) = 1 hoặc chọn phương pháp trực tiếp.
                  </Alert>
                </Grid>
              )}

              <Grid size={{ xs: 12 }}>
                <MatrixLatexEditor
                  matrix={matrix}
                  bMatrix={bMatrix}
                  onMatrixChange={setMatrix}
                  onBMatrixChange={setBMatrix}
                  rows={rows}
                  aCols={aCols}
                  bCols={bCols}
                  onRowsChange={handleRowsChange}
                  onAColsChange={handleAColsChange}
                  onBColsChange={handleBColsChange}
                  label="A × X = B"
                  requireSquare={selectedMethodInfo?.requiresSquare ?? false}
                  squareWarningMessage={
                    selectedMethodInfo?.requiresSquare
                      ? `${selectedMethodInfo?.label} yêu cầu ma trận vuông. A hiện tại là ${rows}×{aCols}.`
                      : ''
                  }
                />
              </Grid>

              {matrixProps && (
                <Grid size={{ xs: 12 }}>
                  <Paper variant="outlined" sx={{ p: 2, bgcolor: 'action.hover' }}>
                    <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>🔍 Phân tích ma trận tự động</Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 1 }}>
                      <Chip size="small" label={matrixProps.is_square ? "Vuông" : "Không vuông"}
                        color={matrixProps.is_square ? "success" : "warning"} variant="outlined" />
                      {matrixProps.is_square && (
                        <>
                          <Chip size="small" label="Đối xứng" color={matrixProps.is_symmetric ? "success" : "default"} variant="outlined" />
                          <Chip size="small" label="Xác định dương" color={matrixProps.is_positive_definite ? "success" : "default"} variant="outlined" />
                          <Chip size="small" label="Ba đường chéo" color={matrixProps.is_tridiagonal ? "success" : "default"} variant="outlined" />
                          <Chip size="small" label="Chéo trội" color={matrixProps.is_diagonally_dominant_strict ? "success" : "default"} variant="outlined" />
                        </>
                      )}
                    </Box>
                    {matrixProps.recommendations.length > 0 && (
                      <Box>
                        {matrixProps.recommendations.map((rec, idx) => (
                          <Typography key={idx} variant="body2" sx={{ color: 'info.main', fontWeight: 500 }}>💡 {rec}</Typography>
                        ))}
                      </Box>
                    )}
                  </Paper>
                </Grid>
              )}

              <Grid size={{ xs: 12 }}>
                <Button
                  variant="contained" size="large"
                  onClick={handleSolve}
                  disabled={loading || methodBlocked || iterativeBBlocked || (showIterativeParams && !isSquare)}
                  startIcon={loading ? <CircularProgress size={20} /> : null}>
                  Giải hệ phương trình
                </Button>
              </Grid>
            </Grid>
          </Paper>

          {/* AX=B Results */}
          {result && (
            <>
              {result.solution_type === 'unique' && result.solution && result.solution.length > 0 && (
                <ResultCard success={result.success} message={result.message}>
                  <Box sx={{ mt: 1 }}>
                    {renderSolutionMatrix(result.solution)}
                    {result.execution_time && (
                      <Typography variant="body2" color="text.secondary">⏱ {result.execution_time.toFixed(6)}s</Typography>
                    )}
                  </Box>
                </ResultCard>
              )}

              {result.solution_type === 'inconsistent' && (
                <ResultCard success={false} message={result.message}>
                  <Paper variant="outlined" sx={{ p: 2, mt: 2, bgcolor: 'error.light', color: 'error.contrastText' }}>
                    <Typography variant="h6" sx={{ fontWeight: 700 }}>⚠️ Hệ vô nghiệm</Typography>
                    <Typography sx={{ mt: 1 }}>rank(A) = {result.rank_A} {'<'} rank([A|B]) = {result.rank_augmented}</Typography>
                  </Paper>
                </ResultCard>
              )}

              {result.solution_type === 'infinite' && (
                <ResultCard success={true} message={result.message}>
                  <Stack spacing={2} sx={{ mt: 1 }}>
                    <Paper variant="outlined" sx={{ p: 2 }}>
                      <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>📊 Phân tích hệ</Typography>
                      <Typography sx={{ mt: 1 }}>
                        rank(A) = {result.rank_A} = rank([A|B]) = {result.rank_augmented} {'<'} số ẩn n = {result.analysis ? String((result.analysis as Record<string, unknown>).n) : '?'}
                      </Typography>
                    </Paper>
                    {result.free_variables && result.free_variables.length > 0 && (
                      <Paper variant="outlined" sx={{ p: 2 }}>
                        <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>🔓 Biến tự do</Typography>
                        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                          {result.free_variables.map((fv, idx) => (
                            <Chip key={idx} label={`${fv} = c${idx + 1}`} color="secondary" variant="outlined" />
                          ))}
                        </Box>
                      </Paper>
                    )}
                    {result.particular_solution && result.particular_solution.length > 0 && (
                      <Paper variant="outlined" sx={{ p: 2 }}>
                        <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>📌 Nghiệm riêng x<sub>p</sub></Typography>
                        {Array.isArray(result.particular_solution[0]) ? (
                          // n×p matrix — show each column
                          <Box sx={{ fontFamily: 'monospace', fontSize: '0.9rem', overflowX: 'auto' }}>
                            {(result.particular_solution[0] as number[]).map((_, ci) => (
                              <Box key={ci} sx={{ mb: 1 }}>
                                <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                  Cột {ci + 1}: [{(result.particular_solution as number[][]).map(row => row[ci].toFixed(6)).join(', ')}]<sup>T</sup>
                                </Typography>
                              </Box>
                            ))}
                          </Box>
                        ) : (
                          // Legacy flat vector
                          <Box sx={{ fontFamily: 'monospace', fontSize: '1.1rem' }}>
                            x<sub>p</sub> = [{(result.particular_solution as unknown as number[]).map(v => v.toFixed(6)).join(', ')}]<sup>T</sup>
                          </Box>
                        )}
                      </Paper>
                    )}
                    {result.basis_vectors && result.basis_vectors.length > 0 && (
                      <Paper variant="outlined" sx={{ p: 2 }}>
                        <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>🧱 Vector cơ sở</Typography>
                        {result.basis_vectors.map((bv, idx) => (
                          <Box key={idx} sx={{ mb: 1.5 }}>
                            <Typography variant="body2" sx={{ fontWeight: 600 }}>v<sub>{idx + 1}</sub> = [{bv.map(v => v.toFixed(6)).join(', ')}]<sup>T</sup></Typography>
                            <Box sx={{ fontFamily: 'monospace', fontSize: '0.9rem', borderLeft: '2px solid', borderRight: '2px solid', borderColor: 'primary.main', display: 'inline-block', px: 1, py: 0.5, lineHeight: 1.5 }}>
                              {bv.map((v, ri) => <Box key={ri}>{v.toFixed(6)}</Box>)}
                            </Box>
                          </Box>
                        ))}
                      </Paper>
                    )}
                    {result.general_solution_latex && (
                      <Paper variant="outlined" sx={{ p: 2 }}>
                        <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>🔢 Nghiệm tổng quát</Typography>
                        <FormulaRenderer latex={result.general_solution_latex} />
                      </Paper>
                    )}
                  </Stack>
                </ResultCard>
              )}

              {isIterative(method) && result.iterations && result.iterations.length > 0 && result.solution && result.solution.length > 0 && (
                <ResultCard success={result.success} message={result.message}>
                  <Box sx={{ mt: 1 }}>
                    {renderSolutionMatrix(result.solution)}
                    <Typography>Số vòng lặp: {result.iterations_count} | Sai số: {result.final_error.toExponential(4)}</Typography>
                    {result.diagonally_dominant === false && (
                      <Alert severity="warning" sx={{ mt: 1 }}>Ma trận không chéo trội — hội tụ có thể chậm hoặc không hội tụ.</Alert>
                    )}
                  </Box>
                </ResultCard>
              )}

              {result.formula && (
                <Box sx={{ mt: 3 }}>
                  <Typography variant="h6">Công thức:</Typography>
                  <FormulaRenderer latex={result.formula} />
                </Box>
              )}

              {isIterative(method) && result.iterations && result.iterations.length > 0 && (
                <IterationTable data={result.iterations} title="Bảng quá trình lặp" />
              )}

              {isDirect(method) && result.steps && result.steps.length > 0 && (
                <SolutionSteps steps={result.steps as any} method={method} />
              )}
            </>
          )}
        </>
      )}

      {/* ===== TAB 1: Ma trận nghịch đảo ===== */}
      {tabIndex === 1 && (
        <>
          <Paper sx={{ p: 3, mb: 3 }}>
            <Grid container spacing={3}>
              <Grid size={{ xs: 12, md: 5 }}>
                <FormControl fullWidth>
                  <InputLabel>Phương pháp</InputLabel>
                  <Select value={invMethod} label="Phương pháp" onChange={(e) => setInvMethod(e.target.value)}>
                    <MenuItem value="gauss_jordan">Gauss-Jordan Elimination</MenuItem>
                    <MenuItem value="adjoint">Adjoint (Ma trận phụ hợp)</MenuItem>
                    <MenuItem value="lu">LU Decomposition</MenuItem>
                    <MenuItem value="cholesky">Cholesky Decomposition</MenuItem>
                  </Select>
                </FormControl>
              </Grid>

              {/* Cholesky conditions warning */}
              {invMethod === 'cholesky' && matrixProps && !matrixProps.is_symmetric && (
                <Grid size={{ xs: 12 }}>
                  <Alert severity="warning">
                    Phương pháp Cholesky yêu cầu ma trận đối xứng xác định dương.
                    Ma trận hiện tại không đối xứng — vui lòng chọn phương pháp khác.
                  </Alert>
                </Grid>
              )}
              {invMethod === 'cholesky' && matrixProps?.is_symmetric && !matrixProps.is_positive_definite && (
                <Grid size={{ xs: 12 }}>
                  <Alert severity="warning">
                    Phương pháp Cholesky yêu cầu ma trận xác định dương.
                    Ma trận hiện tại đối xứng nhưng không xác định dương.
                  </Alert>
                </Grid>
              )}

              <Grid size={{ xs: 12 }}>
                <MatrixLatexEditor
                  matrix={invMatrix}
                  bMatrix={[]}
                  onMatrixChange={setInvMatrix}
                  onBMatrixChange={() => {}}
                  rows={invSize}
                  aCols={invSize}
                  bCols={0}
                  onRowsChange={handleInvSizeChange}
                  onAColsChange={handleInvSizeChange}
                  label="A (n×n)"
                  requireSquare={true}
                  squareWarningMessage="Ma trận phải vuông để có nghịch đảo."
                  hideB={true}
                />
              </Grid>

              {/* Matrix analysis badges */}
              {matrixProps && (
                <Grid size={{ xs: 12 }}>
                  <Paper variant="outlined" sx={{ p: 2, bgcolor: 'action.hover' }}>
                    <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>🔍 Phân tích ma trận</Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                      <Chip size="small" label="Vuông" color={matrixProps.is_square ? "success" : "warning"} variant="outlined" />
                      {matrixProps.is_square && (
                        <>
                          <Chip size="small" label="Khả nghịch" color={invResult?.success ? "success" : "default"} variant="outlined" />
                          <Chip size="small" label="Đối xứng" color={matrixProps.is_symmetric ? "success" : "default"} variant="outlined" />
                          <Chip size="small" label="Xác định dương" color={matrixProps.is_positive_definite ? "success" : "default"} variant="outlined" />
                          <Chip size="small" label="Ba đường chéo" color={matrixProps.is_tridiagonal ? "success" : "default"} variant="outlined" />
                          <Chip size="small" label="Chéo trội" color={matrixProps.is_diagonally_dominant_strict ? "success" : "default"} variant="outlined" />
                        </>
                      )}
                    </Box>
                    {matrixProps.recommendations.length > 0 && (
                      <Box sx={{ mt: 1 }}>
                        {matrixProps.recommendations.map((rec, idx) => (
                          <Typography key={idx} variant="body2" sx={{ color: 'info.main', fontWeight: 500 }}>💡 {rec}</Typography>
                        ))}
                      </Box>
                    )}
                  </Paper>
                </Grid>
              )}

              <Grid size={{ xs: 12 }}>
                <Button
                  variant="contained" size="large" color="secondary"
                  onClick={handleInverse}
                  disabled={invLoading || (invMethod === 'cholesky' && !!matrixProps && !matrixProps.is_symmetric)}
                  startIcon={invLoading ? <CircularProgress size={20} /> : null}>
                  Tính ma trận nghịch đảo
                </Button>
              </Grid>
            </Grid>
          </Paper>

          {/* Inverse result */}
          {invResult && (
            <ResultCard success={invResult.success} message={invResult.message}>
              {invResult.success && (
                <Stack spacing={2} sx={{ mt: 1 }}>
                  {/* Determinant + Rank */}
                  <Paper variant="outlined" sx={{ p: 2 }}>
                    <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>📈 Định thức & Hạng</Typography>
                    <Box sx={{ display: 'flex', gap: 4, mt: 1 }}>
                      <Box>
                        <Typography variant="body2" color="text.secondary">det(A)</Typography>
                        <Typography variant="h6" sx={{ fontFamily: 'monospace' }}>
                          {invResult.determinant?.toFixed(6)}
                        </Typography>
                      </Box>
                      {invResult.rank !== undefined && (
                        <Box>
                          <Typography variant="body2" color="text.secondary">rank(A)</Typography>
                          <Typography variant="h6" sx={{ fontFamily: 'monospace' }}>
                            {invResult.rank}
                          </Typography>
                        </Box>
                      )}
                    </Box>
                  </Paper>

                  {/* Inverse matrix */}
                  {invResult.inverse && (
                    <Paper variant="outlined" sx={{ p: 2 }}>
                      <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>🔁 Ma trận nghịch đảo A⁻¹</Typography>
                      <Box sx={{ fontFamily: 'monospace', fontSize: '0.95rem', overflowX: 'auto' }}>
                        {invResult.inverse.map((row, ri) => (
                          <Box key={ri}>
                            | {row.map((v: number) => v.toFixed(6).padStart(12)).join(' ')} |
                          </Box>
                        ))}
                      </Box>
                      {invResult.inverse_latex && (
                        <Box sx={{ mt: 2 }}>
                          <FormulaRenderer latex={invResult.inverse_latex} />
                        </Box>
                      )}
                    </Paper>
                  )}

                  {/* Verification */}
                  {invResult.verification && (
                    <Paper variant="outlined" sx={{ p: 2 }}>
                      <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>✅ Kiểm chứng: A × A⁻¹</Typography>
                      <Box sx={{ fontFamily: 'monospace', fontSize: '0.9rem', overflowX: 'auto' }}>
                        {invResult.verification.map((row, ri) => (
                          <Box key={ri}>
                            | {row.map((v: number) => v.toFixed(6).padStart(12)).join(' ')} |
                          </Box>
                        ))}
                      </Box>
                      <Typography sx={{ mt: 1, fontWeight: 600, color: invResult.is_accurate ? 'success.main' : 'warning.main' }}>
                        {invResult.is_accurate
                          ? '✅ A × A⁻¹ ≈ I (ma trận đơn vị) — kết quả chính xác'
                          : '⚠️ A × A⁻¹ không gần I — kết quả có thể không chính xác do sai số'}
                      </Typography>
                    </Paper>
                  )}

                  {invResult.execution_time && (
                    <Typography variant="body2" color="text.secondary">
                      ⏱ {invResult.execution_time.toFixed(6)}s
                    </Typography>
                  )}
                </Stack>
              )}

              {!invResult.success && (
                <Typography color="error" sx={{ mt: 1 }}>{invResult.message}</Typography>
              )}
            </ResultCard>
          )}

          {/* Steps for inverse */}
          {invResult?.steps && invResult.steps.length > 0 && (
            <SolutionSteps steps={invResult.steps as any} method={invMethod} />
          )}
        </>
      )}
    </Container>
  );
}