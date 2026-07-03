import { useState } from 'react';
import {
  Container, Typography, TextField, Select, MenuItem, FormControl,
  InputLabel, Button, Grid, Paper, Box, CircularProgress,
} from '@mui/material';
import { integrationAPI, type IntegrationResponse } from '../services/api';
import FormulaEditor from '../components/FormulaEditor';
import FormulaRenderer from '../components/FormulaRenderer';
import ResultCard from '../components/ResultCard';
import { toLatex } from '../utils/latex';

const methods = [
  { value: 'trapezoidal', label: 'Trapezoidal Rule' },
  { value: 'simpson_13', label: "Simpson's 1/3 Rule" },
  { value: 'simpson_38', label: "Simpson's 3/8 Rule" },
  { value: 'romberg', label: 'Romberg Integration' },
];

export default function Integration() {
  const [formData, setFormData] = useState({
    function: 'x^2',
    a: '0',
    b: '1',
    n: '100',
    method: 'trapezoidal',
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<IntegrationResponse | null>(null);

  const handleChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleSolve = async () => {
    setLoading(true);
    try {
      const response = await integrationAPI.solve({
        function: formData.function,
        a: parseFloat(formData.a) || 0,
        b: parseFloat(formData.b) || 0,
        n: parseInt(formData.n) || 100,
        method: formData.method,
      });
      setResult(response.data);
    } catch {
      setResult({
        success: false, message: 'Lỗi kết nối server.',
        iterations: [], formula: '',
      });
    }
    setLoading(false);
  };

  return (
    <Container maxWidth="lg">
      <Typography variant="h4" sx={{ fontWeight: 700, mb: 3 }}>
        📊 Tích phân số
      </Typography>

      <Paper sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={3}>
          <Grid size={{ xs: 12, md: 6 }}>
            <FormControl fullWidth>
              <InputLabel>Phương pháp</InputLabel>
              <Select value={formData.method} label="Phương pháp"
                onChange={(e) => handleChange('method', e.target.value)}>
                {methods.map(m => <MenuItem key={m.value} value={m.value}>{m.label}</MenuItem>)}
              </Select>
            </FormControl>
          </Grid>
          <Grid size={{ xs: 4, md: 2 }}>
            <TextField fullWidth type="text" inputMode="decimal" label="a" value={formData.a}
              onChange={(e) => handleChange('a', e.target.value)} />
          </Grid>
          <Grid size={{ xs: 4, md: 2 }}>
            <TextField fullWidth type="text" inputMode="decimal" label="b" value={formData.b}
              onChange={(e) => handleChange('b', e.target.value)} />
          </Grid>
          <Grid size={{ xs: 4, md: 2 }}>
            <TextField fullWidth type="text" inputMode="numeric" label="n" value={formData.n}
              onChange={(e) => handleChange('n', e.target.value)} />
          </Grid>

          <Grid size={{ xs: 12 }}>
            <FormulaEditor
              value={formData.function}
              onChange={(v) => handleChange('function', v)}
              placeholder="x^2, \sin(x), e^{-x^2}, \frac{1}{1+x^2}"
              label="Hàm số f(x) dưới dấu tích phân (LaTeX)"
              minRows={2}
            />
          </Grid>

          <Grid size={{ xs: 12 }}>
            <Button variant="contained" size="large" onClick={handleSolve} disabled={loading}
              startIcon={loading ? <CircularProgress size={20} /> : null}>
              Tính toán
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {result && (
        <>
          <ResultCard success={result.success} message={result.message}>
            {result.result !== undefined && result.result !== null && (
              <Box sx={{ mt: 1 }}>
                <Typography variant="h5" sx={{ fontWeight: 600, color: 'primary.main' }}>
                  ∫f(x)dx = {result.result}
                </Typography>
                {result.exact_value && (
                  <Typography>Giá trị đúng: {result.exact_value} | Sai số tuyệt đối: {result.error}</Typography>
                )}
                {result.relative_error != null && (
                  <Typography>Sai số tương đối: {result.relative_error}%</Typography>
                )}
              </Box>
            )}
          </ResultCard>

          <Box sx={{ mt: 2 }}>
            <Typography variant="h6">Bài toán:</Typography>
            <FormulaRenderer latex={`\\int_{${formData.a}}^{${formData.b}} \\left(${toLatex(formData.function)}\\right) \\, dx`} />
          </Box>

          {result.formula && (
            <Box sx={{ mt: 3 }}>
              <Typography variant="h6">Công thức:</Typography>
              <FormulaRenderer latex={result.formula} />
            </Box>
          )}
        </>
      )}
    </Container>
  );
}