import { useState } from 'react';
import {
  Container, Typography, Select, MenuItem, FormControl,
  InputLabel, Button, Grid, Paper, Box, CircularProgress,
} from '@mui/material';
import { type MatrixInverseResponse } from '../services/api';
import MatrixLatexEditor from '../components/MatrixLatexEditor';
import FormulaRenderer from '../components/FormulaRenderer';
import ResultCard from '../components/ResultCard';

const BASE_URL = '/matrix-inverse';

const methods = [
  { value: 'gauss_jordan', label: 'Gauss-Jordan Elimination' },
  { value: 'adjoint', label: 'Adjoint (Ma trận phụ hợp)' },
  { value: 'lu', label: 'LU Decomposition' },
  { value: 'cholesky', label: 'Cholesky Decomposition' },
  { value: 'pseudoinverse_svd', label: 'Pseudoinverse SVD (Moore-Penrose)' },
];

export default function MatrixInverse() {
  const [size, setSize] = useState(3);
  const [matrix, setMatrix] = useState<number[][]>([
    [1, 2, 0],
    [3, 4, 1],
    [0, 1, 2],
  ]);
  const [method, setMethod] = useState('gauss_jordan');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<MatrixInverseResponse | null>(null);

  const handleSolve = async () => {
    setLoading(true);
    try {
      const res = await fetch(BASE_URL + '/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ A: matrix, method }),
      });
      const data = await res.json();
      setResult(data);
    } catch {
      setResult({ success: false, message: 'Lỗi kết nối server.' });
    }
    setLoading(false);
  };

  return (
    <Container maxWidth="lg">
      <Typography variant="h4" sx={{ fontWeight: 700, mb: 3 }}>🔁 Ma trận nghịch đảo A⁻¹</Typography>
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
              rows={size} aCols={size} bCols={0}
              onRowsChange={setSize} onAColsChange={setSize}
              label="A (n×n)"
              requireSquare={true}
              hideB={true}
            />
          </Grid>
          <Grid size={{ xs: 12 }}>
            <Button variant="contained" size="large" onClick={handleSolve} disabled={loading}
              startIcon={loading ? <CircularProgress size={20} /> : null}>
              Tính ma trận nghịch đảo
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {result && (
        <ResultCard success={result.success} message={result.message}>
          {result.success && (
            <Box sx={{ mt: 2 }}>
              {result.determinant != null && (
                <Typography variant="body1">det(A) = {result.determinant.toFixed(6)}  |  rank(A) = {result.rank}</Typography>
              )}
              {result.singular_values && (
                <Typography variant="body2">Singular values: [{result.singular_values.map(v => v.toFixed(4)).join(', ')}]</Typography>
              )}
              {result.condition_number != null && (
                <Typography variant="body2">κ(A) = {result.condition_number.toFixed(4)}</Typography>
              )}
              {result.inverse && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="h6">A⁻¹:</Typography>
                  <Box sx={{ fontFamily: 'monospace', fontSize: '0.95rem', overflowX: 'auto' }}>
                    {result.inverse.map((row, ri) => (
                      <Box key={ri}>| {row.map((v: number) => v.toFixed(6).padStart(12)).join(' ')} |</Box>
                    ))}
                  </Box>
                </Box>
              )}
              {result.inverse_latex && <FormulaRenderer latex={result.inverse_latex} />}
              {result.verification && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="h6">A × A⁻¹:</Typography>
                  <Box sx={{ fontFamily: 'monospace', fontSize: '0.9rem', overflowX: 'auto' }}>
                    {result.verification.map((row, ri) => (
                      <Box key={ri}>| {row.map((v: number) => v.toFixed(6).padStart(12)).join(' ')} |</Box>
                    ))}
                  </Box>
                  <Typography color={result.is_accurate ? 'success.main' : 'warning.main'} sx={{ mt: 1 }}>
                    {result.is_accurate ? '✅ A × A⁻¹ ≈ I' : '⚠️ Kết quả không chính xác'}
                  </Typography>
                </Box>
              )}
              {result.execution_time != null && (
                <Typography variant="body2" color="text.secondary">⏱ {result.execution_time.toFixed(6)}s</Typography>
              )}
            </Box>
          )}
        </ResultCard>
      )}
    </Container>
  );
}