import { useState } from 'react';
import {
  Container, Typography, TextField, Button, Grid, Paper, Box, Chip, CircularProgress,
} from '@mui/material';
import { comparisonAPI, type ComparisonResponse } from '../services/api';
import ResultCard from '../components/ResultCard';

const allMethods = [
  { value: 'bisection', label: 'Bisection' },
  { value: 'newton', label: 'Newton-Raphson' },
  { value: 'secant', label: 'Secant' },
  { value: 'fixed_point', label: 'Fixed Point' },
];

export default function Comparison() {
  const [formData, setFormData] = useState({
    function: 'x^3 - 6*x^2 + 11*x - 6',
    a: '0',
    b: '4',
    x0: '0.5',
    x1: '1.5',
    epsilon: '0.000001',
    max_iterations: '100',
    methods: ['bisection', 'newton', 'secant'],
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ComparisonResponse | null>(null);

  const handleChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleMethodToggle = (method: string) => {
    const methods = formData.methods.includes(method)
      ? formData.methods.filter(m => m !== method)
      : [...formData.methods, method];
    setFormData({ ...formData, methods });
  };

  const handleSolve = async () => {
    setLoading(true);
    try {
      const response = await comparisonAPI.compare({
        function: formData.function,
        a: parseFloat(formData.a) || 0,
        b: parseFloat(formData.b) || 0,
        x0: parseFloat(formData.x0) || 0,
        x1: parseFloat(formData.x1) || 0,
        epsilon: parseFloat(formData.epsilon) || 1e-6,
        max_iterations: parseInt(formData.max_iterations) || 100,
        methods: formData.methods,
        task_type: 'root_finding',
      });
      setResult(response.data);
    } catch {
      setResult({
        success: false, message: 'Lỗi kết nối server.', results: [],
      });
    }
    setLoading(false);
  };

  return (
    <Container maxWidth="lg">
      <Typography variant="h4" sx={{ fontWeight: 700, mb: 3 }}>
        ⚖️ So sánh thuật toán
      </Typography>

      <Paper sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={3}>
          <Grid size={{ xs: 12, md: 6 }}>
            <TextField fullWidth label="Hàm số f(x)" value={formData.function}
              onChange={(e) => handleChange('function', e.target.value)} />
          </Grid>
          <Grid size={{ xs: 12, md: 6 }}>
            <Typography variant="subtitle2" gutterBottom>Chọn phương pháp:</Typography>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              {allMethods.map(m => (
                <Chip key={m.value} label={m.label}
                  color={formData.methods.includes(m.value) ? 'primary' : 'default'}
                  onClick={() => handleMethodToggle(m.value)}
                  variant={formData.methods.includes(m.value) ? 'filled' : 'outlined'} />
              ))}
            </Box>
          </Grid>
          <Grid size={{ xs: 6, md: 2 }}>
            <TextField fullWidth type="text" inputMode="decimal" label="a" value={formData.a}
              onChange={(e) => handleChange('a', e.target.value)} />
          </Grid>
          <Grid size={{ xs: 6, md: 2 }}>
            <TextField fullWidth type="text" inputMode="decimal" label="b" value={formData.b}
              onChange={(e) => handleChange('b', e.target.value)} />
          </Grid>
          <Grid size={{ xs: 6, md: 2 }}>
            <TextField fullWidth type="text" inputMode="decimal" label="x₀" value={formData.x0}
              onChange={(e) => handleChange('x0', e.target.value)} />
          </Grid>
          <Grid size={{ xs: 6, md: 2 }}>
            <TextField fullWidth type="text" inputMode="decimal" label="Epsilon" value={formData.epsilon}
              onChange={(e) => handleChange('epsilon', e.target.value)} />
          </Grid>
          <Grid size={{ xs: 12 }}>
            <Button variant="contained" size="large" onClick={handleSolve}
              disabled={loading || formData.methods.length < 2}
              startIcon={loading ? <CircularProgress size={20} /> : null}>
              So sánh ({formData.methods.length} phương pháp)
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {result && (
        <>
          <ResultCard success={result.success} message={result.message}>
            {result.summary && (
              <Box sx={{ mt: 1 }}>
                <Typography>Phương pháp nhanh nhất: <strong>{(result.summary as any).fastest}</strong></Typography>
                <Typography>Ít vòng lặp nhất: <strong>{(result.summary as any).fewest_iterations}</strong></Typography>
              </Box>
            )}
          </ResultCard>
          {result.results && result.results.length > 0 && (
            <Paper variant="outlined" sx={{ p: 2, mt: 3 }}>
              <Typography variant="h6" sx={{ mb: 2 }}>Tổng quan so sánh:</Typography>
              <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                {(result.results as any[]).map((r: any) => (
                  <Paper key={r.method} variant="outlined" sx={{ p: 2, flex: 1, minWidth: 200 }}>
                    <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>{r.method}</Typography>
                    <Typography variant="body2">Nghiệm: {r.root?.toFixed(6) || 'N/A'}</Typography>
                    <Typography variant="body2">Vòng lặp: {r.iterations_count}</Typography>
                    <Typography variant="body2">Sai số: {r.final_error?.toExponential(4)}</Typography>
                    <Typography variant="body2">Thời gian: {r.execution_time}s</Typography>
                  </Paper>
                ))}
              </Box>
            </Paper>
          )}
        </>
      )}
    </Container>
  );
}