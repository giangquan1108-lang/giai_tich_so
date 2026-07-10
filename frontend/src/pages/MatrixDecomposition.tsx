import { useState } from 'react';
import {
  Container, Typography, Select, MenuItem, FormControl,
  InputLabel, Button, Grid, Paper, Box, CircularProgress,
} from '@mui/material';
import MatrixLatexEditor from '../components/MatrixLatexEditor';
import ResultCard from '../components/ResultCard';
import FormulaRenderer from '../components/FormulaRenderer';
import SolutionSteps from '../components/SolutionSteps';

const BASE_URL = '/matrix-decomposition';

const methods = [
  { value: 'lu', label: 'LU Decomposition (A = P·L·U)' },
  { value: 'cholesky', label: 'Cholesky Decomposition (A = L·L^T)' },
  { value: 'svd', label: 'SVD Decomposition (A = U·Σ·V^T)' },
];

export default function MatrixDecomposition() {
  const [rows, setRows] = useState(3);
  const [cols, setCols] = useState(3);
  const [matrix, setMatrix] = useState<number[][]>([
    [4, -2, 1],
    [-2, 4, -2],
    [1, -2, 4],
  ]);
  const [method, setMethod] = useState('lu');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const handleSolve = async () => {
    setLoading(true);
    try {
      const res = await fetch(BASE_URL + '/', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ A: matrix, method }),
      });
      setResult(await res.json());
    } catch {
      setResult({ success: false, message: 'Lỗi kết nối server.' });
    }
    setLoading(false);
  };

  const renderMatrix = (label: string, mat: number[][], latex?: string) => (
    <Box sx={{ mt: 1 }}>
      <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>{label}:</Typography>
      <Box sx={{ fontFamily: 'monospace', fontSize: '0.85rem', overflowX: 'auto' }}>
        {mat.map((row, ri) => (
          <Box key={ri}>| {row.map((v: number) => v.toFixed(7).padStart(10)).join(' ')} |</Box>
        ))}
      </Box>
      {latex && latex.length < 200 && <FormulaRenderer latex={latex} />}
    </Box>
  );

  return (
    <Container maxWidth="lg">
      <Typography variant="h4" sx={{ fontWeight: 700, mb: 3 }}>📦 Phân tách ma trận</Typography>
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
          <Grid size={{ xs: 12 }}>
            <MatrixLatexEditor
              matrix={matrix} bMatrix={[]}
              onMatrixChange={setMatrix} onBMatrixChange={() => {}}
              rows={rows} aCols={cols} bCols={0}
              onRowsChange={setRows} onAColsChange={setCols}
              label="Matrix A"
              hideB={true}
            />
          </Grid>
          <Grid size={{ xs: 12 }}>
            <Button variant="contained" size="large" onClick={handleSolve} disabled={loading}
              startIcon={loading ? <CircularProgress size={20} /> : null}>
              Phân tách
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {result && (
        <ResultCard success={result.success} message={result.message}>
          {result.success && (
            <Box sx={{ mt: 1 }}>
              {result.P && renderMatrix('P', result.P, result.P_latex)}
              {result.L && renderMatrix('L', result.L, result.L_latex)}
              {result.U && renderMatrix('U', result.U, result.U_latex)}
              {result.Vt && renderMatrix('V^T', result.Vt, result.Vt_latex)}
              {result.singular_values && (
                <Typography variant="body2" sx={{ mt: 1 }}>
                  Sigma = diag([{result.singular_values.map((v: number) => v.toFixed(4)).join(', ')}])
                  {' '}| rank(A) = {result.rank} | kappa(A) = {result.condition_number?.toFixed(4)}
                </Typography>
              )}
              {result.verification && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>Kiểm tra (P·A so với L·U):</Typography>
                  <Box sx={{ fontFamily: 'monospace', fontSize: '0.8rem', overflowX: 'auto' }}>
                    {result.verification.map((row: number[], ri: number) => (
                      <Box key={ri}>| {row.map((v: number) => v.toFixed(7).padStart(10)).join(' ')} |</Box>
                    ))}
                  </Box>
                  <Typography color={result.is_accurate ? 'success.main' : 'warning.main'} sx={{ mt: 0.5, fontSize: '0.85rem' }}>
                    {result.is_accurate ? '✅ P·A = L·U' : '⚠️ Kết quả không chính xác hoàn toàn'}
                  </Typography>
                </Box>
              )}
              {result.execution_time != null && (
                <Typography variant="body2" color="text.secondary">⏱ {result.execution_time.toFixed(7)}s</Typography>
              )}
            </Box>
          )}
        </ResultCard>
      )}
      {result?.steps && result.steps.length > 0 && (
        <SolutionSteps steps={result.steps} method={method} />
      )}
    </Container>
  );
}