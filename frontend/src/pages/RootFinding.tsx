import { useState } from 'react';
import {
  Container, Typography, TextField, Select, MenuItem, FormControl,
  InputLabel, Button, Grid, Paper, Box, CircularProgress,
} from '@mui/material';
import { rootFindingAPI, type RootFindingResponse } from '../services/api';
import FormulaEditor from '../components/FormulaEditor';
import FormulaRenderer from '../components/FormulaRenderer';
import IterationTable from '../components/IterationTable';
import { toLatex } from '../utils/latex';
import ResultCard from '../components/ResultCard';

const methods = [
  { value: 'bisection', label: 'Phương pháp chia đôi (Bisection)', isFixedPoint: false },
  { value: 'newton', label: 'Newton-Raphson', isFixedPoint: false },
  { value: 'secant', label: 'Phương pháp dây cung (Secant)', isFixedPoint: false },
  { value: 'fixed_point', label: 'Lặp đơn (Fixed Point Iteration)', isFixedPoint: true },
];

const fixedPointExamples = [
  'cos(x)',
  '(x^3 + 1) / 2',
  'sqrt(x + 2)',
];

export default function RootFinding() {
  const [formData, setFormData] = useState({
    function: 'x^3 - 6*x^2 + 11*x - 6',
    a: '0',
    b: '4',
    x0: '0.5',
    x1: '1.5',
    epsilon: '0.000001',
    max_iterations: '100',
    method: 'bisection',
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<RootFindingResponse | null>(null);

  const isFixedPoint = formData.method === 'fixed_point';

  const handleChange = (field: string, value: string) => {
    setFormData(prev => {
      const next = { ...prev, [field]: value };
      if (field === 'method' && value === 'fixed_point' && prev.function === 'x^3 - 6*x^2 + 11*x - 6') {
        next.function = 'cos(x)';
      }
      return next;
    });
  };

  const handleSolve = async () => {
    setLoading(true);
    try {
      const response = await rootFindingAPI.solve({
        function: formData.function,
        a: formData.method === 'bisection' ? parseFloat(formData.a) || 0 : undefined,
        b: formData.method === 'bisection' ? parseFloat(formData.b) || 0 : undefined,
        x0: ['newton', 'fixed_point'].includes(formData.method) ? parseFloat(formData.x0) || 0 :
            formData.method === 'secant' ? parseFloat(formData.x0) || 0 : undefined,
        x1: formData.method === 'secant' ? parseFloat(formData.x1) || 0 : undefined,
        epsilon: parseFloat(formData.epsilon) || 1e-6,
        max_iterations: parseInt(formData.max_iterations) || 100,
        method: formData.method,
      });
      setResult(response.data);
    } catch {
      setResult({
        success: false,
        message: 'Lỗi kết nối server. Hãy đảm bảo backend đang chạy.',
        iterations_count: 0, final_error: 0, iterations: [], formula: '',
      });
    }
    setLoading(false);
  };

  return (
    <Container maxWidth="lg">
      <Typography variant="h4" sx={{ fontWeight: 700, mb: 3 }}>
        {isFixedPoint ? '🔄 Tìm điểm bất động x = g(x)' : '🔍 Tìm nghiệm phương trình f(x) = 0'}
      </Typography>

      <Paper sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={3}>
          <Grid size={{ xs: 12, md: 6 }}>
            <FormControl fullWidth>
              <InputLabel>Phương pháp</InputLabel>
              <Select value={formData.method} label="Phương pháp"
                onChange={(e) => handleChange('method', e.target.value)}>
                {methods.map((m) => (
                  <MenuItem key={m.value} value={m.value}>{m.label}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>

          {/* Fixed Point info alert */}
          {isFixedPoint && (
            <Grid size={{ xs: 12 }}>
              <Box sx={{ bgcolor: 'info.light', color: 'info.contrastText', p: 1.5, borderRadius: 1, fontSize: '0.875rem' }}>
                <strong>Phương pháp Lặp Đơn:</strong> Nhập <em>g(x)</em> sao cho <strong>x = g(x)</strong>.
                Ví dụ: g(x) = cos(x) → nhập <code>cos(x)</code>.
              </Box>
            </Grid>
          )}

          <Grid size={{ xs: 12 }}>
            <FormulaEditor
              value={formData.function}
              onChange={(v) => handleChange('function', v)}
              placeholder={isFixedPoint ? fixedPointExamples[0] : 'x^3 - 6*x^2 + 11*x - 6'}
              label={isFixedPoint ? 'g(x) với x = g(x)' : 'Hàm số f(x) = 0'}
              minRows={2}
            />
          </Grid>

          {/* Fixed Point examples */}
          {isFixedPoint && (
            <Grid size={{ xs: 12 }}>
              <Typography variant="caption" color="text.secondary">
                Ví dụ hợp lệ: {fixedPointExamples.map((ex, i) => (
                  <Box key={i} component="span" sx={{ mr: 2, fontFamily: 'monospace' }}>
                    g(x) = {ex}
                  </Box>
                ))}
              </Typography>
            </Grid>
          )}

          {(formData.method === 'bisection') && (
            <>
              <Grid size={{ xs: 6, md: 3 }}>
                <TextField fullWidth type="text" inputMode="decimal" label="a" value={formData.a}
                  onChange={(e) => handleChange('a', e.target.value)} />
              </Grid>
              <Grid size={{ xs: 6, md: 3 }}>
                <TextField fullWidth type="text" inputMode="decimal" label="b" value={formData.b}
                  onChange={(e) => handleChange('b', e.target.value)} />
              </Grid>
            </>
          )}

          {['newton', 'fixed_point'].includes(formData.method) && (
            <Grid size={{ xs: 6, md: 3 }}>
              <TextField fullWidth type="text" inputMode="decimal" label="x₀" value={formData.x0}
                onChange={(e) => handleChange('x0', e.target.value)} />
            </Grid>
          )}

          {formData.method === 'secant' && (
            <>
              <Grid size={{ xs: 6, md: 3 }}>
                <TextField fullWidth type="text" inputMode="decimal" label="x₀" value={formData.x0}
                  onChange={(e) => handleChange('x0', e.target.value)} />
              </Grid>
              <Grid size={{ xs: 6, md: 3 }}>
                <TextField fullWidth type="text" inputMode="decimal" label="x₁" value={formData.x1}
                  onChange={(e) => handleChange('x1', e.target.value)} />
              </Grid>
            </>
          )}

          <Grid size={{ xs: 6, md: 3 }}>
            <TextField fullWidth type="text" inputMode="decimal" label="Epsilon (ε)" value={formData.epsilon}
              onChange={(e) => handleChange('epsilon', e.target.value)} />
          </Grid>
          <Grid size={{ xs: 6, md: 3 }}>
            <TextField fullWidth type="text" inputMode="numeric" label="Số vòng lặp tối đa" value={formData.max_iterations}
              onChange={(e) => handleChange('max_iterations', e.target.value)} />
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
            {result.root !== undefined && (
              <Box sx={{ mt: 1 }}>
                <Typography variant="h5" sx={{ fontWeight: 600, color: 'primary.main' }}>
                  Nghiệm: x = {result.root.toFixed(10)}
                </Typography>
                <Typography variant="body1">
                  f(x) = {result.f_root?.toFixed(10)} | Số vòng lặp: {result.iterations_count} | Sai số: {result.final_error.toExponential(4)}
                </Typography>
              </Box>
            )}
          </ResultCard>

          <Box sx={{ mt: 2 }}>
            <Typography variant="h6">Bài toán:</Typography>
            <FormulaRenderer latex={isFixedPoint ? `x = ${toLatex(formData.function)}` : `${toLatex(formData.function)} = 0`} />
          </Box>

          {result.formula && (
            <Box sx={{ mt: 3 }}>
              <Typography variant="h6">Công thức:</Typography>
              <FormulaRenderer latex={result.formula} />
            </Box>
          )}

          {result.contraction_info && (
            <Box sx={{ mt: 2 }}>
              <Typography
                variant="body2"
                sx={{
                  color: result.contraction_info.startsWith('|g') && result.contraction_info.includes('≥ 1')
                    ? 'error.main' : 'success.main',
                  fontWeight: 500,
                }}
              >
                {result.contraction_info}
              </Typography>
            </Box>
          )}

          <IterationTable data={result.iterations} title="Bảng quá trình lặp" />
        </>
      )}
    </Container>
  );
}