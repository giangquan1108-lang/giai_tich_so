import { useState } from 'react';
import {
  Container, Typography, Select, MenuItem, FormControl,
  InputLabel, Button, Grid, Paper, Box, CircularProgress, TextField, Chip,
} from '@mui/material';
import MatrixLatexEditor from '../components/MatrixLatexEditor';
import IterationTable from '../components/IterationTable';
import ResultCard from '../components/ResultCard';
import SolutionSteps from '../components/SolutionSteps';

const BASE_URL = '/eigenvalues';

const methods = [
  { value: 'characteristic_polynomial', label: 'Đa thức đặc trưng', iterative: false },
  { value: 'power', label: 'Power Iteration (λ_max)', iterative: true },
  { value: 'inverse_power', label: 'Inverse Power Iteration', iterative: true },
  { value: 'rayleigh', label: 'Rayleigh Quotient Iteration', iterative: true },
  { value: 'jacobi', label: 'Jacobi Method (đối xứng)', iterative: false },
  { value: 'danielewski', label: 'Danielewski (Frobenius)', iterative: false },
];

export default function Eigenvalues() {
  const [size, setSize] = useState(3);
  const [matrix, setMatrix] = useState<number[][]>([
    [2, -1, 0],
    [-1, 2, -1],
    [0, -1, 2],
  ]);
  const [method, setMethod] = useState('characteristic_polynomial');
  const [epsilon, setEpsilon] = useState('0.000001');
  const [maxIter, setMaxIter] = useState('100');
  const [shift, setShift] = useState('0');
  const [initialGuess, setInitialGuess] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const isIterative = methods.find(m => m.value === method)?.iterative;

  const handleSolve = async () => {
    setLoading(true);
    try {
      const body: any = {
        A: matrix, method,
        epsilon: parseFloat(epsilon) || 1e-6,
        max_iterations: parseInt(maxIter) || 100,
      };
      if (method === 'inverse_power') body.shift = parseFloat(shift) || 0;
      if (isIterative && initialGuess) {
        body.initial_guess = initialGuess.split(',').map(s => parseFloat(s.trim())).filter(v => !isNaN(v));
      }
      const res = await fetch(BASE_URL + '/', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      setResult(await res.json());
    } catch {
      setResult({ success: false, message: 'Lỗi kết nối server.' });
    }
    setLoading(false);
  };

  return (
    <Container maxWidth="lg">
      <Typography variant="h4" sx={{ fontWeight: 700, mb: 3 }}>👁️ Trị riêng & Vector riêng</Typography>
      <Paper sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={3}>
          <Grid size={{ xs: 12, md: 5 }}>
            <FormControl fullWidth>
              <InputLabel>Phương pháp</InputLabel>
              <Select value={method} label="Phương pháp" onChange={(e) => setMethod(e.target.value)}>
                {methods.map(m => <MenuItem key={m.value} value={m.value}>{m.label}</MenuItem>)}
              </Select>
            </FormControl>
          </Grid>
          {isIterative && (
            <>
              <Grid size={{ xs: 6, md: 2 }}>
                <TextField fullWidth size="small" label="Epsilon" value={epsilon} onChange={(e) => setEpsilon(e.target.value)} />
              </Grid>
              <Grid size={{ xs: 6, md: 2 }}>
                <TextField fullWidth size="small" label="Max Iter" value={maxIter} onChange={(e) => setMaxIter(e.target.value)} />
              </Grid>
              {method === 'inverse_power' && (
                <Grid size={{ xs: 6, md: 2 }}>
                  <TextField fullWidth size="small" label="Shift sigma" value={shift} onChange={(e) => setShift(e.target.value)} />
                </Grid>
              )}
              <Grid size={{ xs: 12, md: method === 'inverse_power' ? 3 : 4 }}>
                <TextField fullWidth size="small" label="Initial guess (cách nhau dấu ,)" value={initialGuess}
                  onChange={(e) => setInitialGuess(e.target.value)} helperText={`VD: ${Array(size).fill('1').join(', ')}`} />
              </Grid>
            </>
          )}
          <Grid size={{ xs: 12 }}>
            <MatrixLatexEditor
              matrix={matrix} bMatrix={[]}
              onMatrixChange={setMatrix} onBMatrixChange={() => {}}
              rows={size} aCols={size} bCols={0}
              onRowsChange={setSize} onAColsChange={setSize}
              label="Matrix A"
              requireSquare={true}
              hideB={true}
            />
          </Grid>
          <Grid size={{ xs: 12 }}>
            <Button variant="contained" size="large" onClick={handleSolve} disabled={loading}
              startIcon={loading ? <CircularProgress size={20} /> : null}>
              Tính trị riêng
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {result && (
        <ResultCard success={result.success} message={result.message}>
          {result.success && (
            <Box sx={{ mt: 2 }}>
              {result.dominant_eigenvalue != null && (
                <Box>
                  <Typography variant="h6">λ_max = {result.dominant_eigenvalue.toFixed(10)}</Typography>
                  {result.dominant_eigenvector && (
                    <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                      v = [{result.dominant_eigenvector.map((v: number) => v.toFixed(7)).join(', ')}]^T
                    </Typography>
                  )}
                </Box>
              )}
              {result.eigenvalues && (
                <Box>
                  <Typography variant="h6" sx={{ mt: 1 }}>Trị riêng:</Typography>
                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', my: 1 }}>
                    {result.eigenvalues.map((ev: any, idx: number) => (
                      <Chip key={idx} label={`λ${idx+1} = ${typeof ev === 'number' ? ev.toFixed(7) : String(ev)}`}
                        color="primary" variant="outlined" />
                    ))}
                  </Box>
                </Box>
              )}
              {result.eigenpairs && result.eigenpairs.length > 0 && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="h6">Cặp (λ, v):</Typography>
                  {result.eigenpairs.map((pair: any, idx: number) => (
                    <Paper key={idx} variant="outlined" sx={{ p: 1.5, mt: 1 }}>
                      <Typography variant="body1" sx={{ fontWeight: 600 }}>
                        λ{idx+1} = {typeof pair.eigenvalue === 'number' ? pair.eigenvalue.toFixed(7) : String(pair.eigenvalue)}
                      </Typography>
                      {pair.eigenvector && (
                        <Typography variant="body2" sx={{ fontFamily: 'monospace', mt: 0.5 }}>
                          v{idx+1} = [{pair.eigenvector.map((v: number) => v.toFixed(7)).join(', ')}]^T
                        </Typography>
                      )}
                    </Paper>
                  ))}
                </Box>
              )}
              {result.iterations_count > 0 && (
                <Typography variant="body2" sx={{ mt: 1 }}>
                  Số vòng lặp: {result.iterations_count} | Sai số: {result.final_error.toExponential(4)}
                </Typography>
              )}
              {result.execution_time != null && (
                <Typography variant="body2" color="text.secondary">⏱ {result.execution_time.toFixed(7)}s</Typography>
              )}
            </Box>
          )}
        </ResultCard>
      )}
      {result?.iterations && result.iterations.length > 0 && (
        <IterationTable data={result.iterations} title="Bảng quá trình lặp" />
      )}
      {result?.steps && result.steps.length > 0 && (
        <SolutionSteps steps={result.steps} method={method} />
      )}
    </Container>
  );
}
