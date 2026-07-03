import { useState } from 'react';
import {
  Container, Typography, TextField, Select, MenuItem, FormControl,
  InputLabel, Button, Grid, Paper, Box, IconButton, CircularProgress,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import { interpolationAPI, type InterpolationResponse } from '../services/api';
import FormulaRenderer from '../components/FormulaRenderer';
import ResultCard from '../components/ResultCard';

const methods = [
  { value: 'lagrange', label: 'Lagrange Interpolation' },
  { value: 'newton_forward', label: 'Newton Forward Differences' },
  { value: 'newton_backward', label: 'Newton Backward Differences' },
  { value: 'divided_differences', label: 'Divided Differences' },
];

export default function Interpolation() {
  const [points, setPoints] = useState([
    { x: '0', y: '1' },
    { x: '1', y: '3' },
    { x: '2', y: '7' },
    { x: '3', y: '13' },
  ]);
  const [xValue, setXValue] = useState('1.5');
  const [method, setMethod] = useState('lagrange');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<InterpolationResponse | null>(null);

  const addPoint = () => {
    const lastX = parseFloat(points[points.length - 1]?.x) || 0;
    setPoints([...points, { x: String(lastX + 1), y: '0' }]);
  };

  const removePoint = (idx: number) => {
    if (points.length > 2) setPoints(points.filter((_, i) => i !== idx));
  };

  const updatePoint = (idx: number, field: 'x' | 'y', value: string) => {
    const newPoints = [...points];
    newPoints[idx][field] = value;
    setPoints(newPoints);
  };

  const handleSolve = async () => {
    setLoading(true);
    try {
      const response = await interpolationAPI.solve({
        x_points: points.map(p => parseFloat(p.x) || 0),
        y_points: points.map(p => parseFloat(p.y) || 0),
        x_value: parseFloat(xValue) || 0,
        method,
      });
      setResult(response.data);
    } catch {
      setResult({ success: false, message: 'Lỗi kết nối server.', iterations: [], formula: '' });
    }
    setLoading(false);
  };

  return (
    <Container maxWidth="lg">
      <Typography variant="h4" sx={{ fontWeight: 700, mb: 3 }}>📉 Nội suy hàm số</Typography>
      <Paper sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={3}>
          <Grid size={{ xs: 12, md: 6 }}>
            <FormControl fullWidth>
              <InputLabel>Phương pháp</InputLabel>
              <Select value={method} label="Phương pháp" onChange={(e) => setMethod(e.target.value)}>
                {methods.map(m => <MenuItem key={m.value} value={m.value}>{m.label}</MenuItem>)}
              </Select>
            </FormControl>
          </Grid>
          <Grid size={{ xs: 6, md: 3 }}>
            <TextField fullWidth type="text" inputMode="decimal" label="Giá trị cần nội suy tại x ="
              value={xValue} onChange={(e) => setXValue(e.target.value)} />
          </Grid>
          <Grid size={{ xs: 12 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
              <Typography variant="subtitle2">Danh sách điểm:</Typography>
              <IconButton size="small" onClick={addPoint} color="primary"><AddIcon /></IconButton>
            </Box>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              {points.map((p, idx) => (
                <Box key={idx} sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                  <Typography variant="body2" sx={{ minWidth: 30 }}>P{idx}:</Typography>
                  <TextField size="small" type="text" inputMode="decimal" label="x" value={p.x}
                    onChange={(e) => updatePoint(idx, 'x', e.target.value)} sx={{ width: 100 }} />
                  <TextField size="small" type="text" inputMode="decimal" label="y" value={p.y}
                    onChange={(e) => updatePoint(idx, 'y', e.target.value)} sx={{ width: 100 }} />
                  <IconButton size="small" onClick={() => removePoint(idx)}
                    disabled={points.length <= 2} color="error"><DeleteIcon /></IconButton>
                </Box>
              ))}
            </Box>
          </Grid>
          <Grid size={{ xs: 12 }}>
            <Button variant="contained" size="large" onClick={handleSolve} disabled={loading}
              startIcon={loading ? <CircularProgress size={20} /> : null}>Tính toán</Button>
          </Grid>
        </Grid>
      </Paper>
      {result && (
        <>
          <ResultCard success={result.success} message={result.message}>
            {result.interpolated_value !== undefined && result.interpolated_value !== null && (
              <Box sx={{ mt: 1 }}>
                <Typography variant="h5" sx={{ fontWeight: 600, color: 'primary.main' }}>
                  f({xValue}) = {result.interpolated_value}
                </Typography>
              </Box>
            )}
          </ResultCard>
          <Box sx={{ mt: 2 }}>
            <Typography variant="h6">Dữ liệu:</Typography>
            <FormulaRenderer latex={`\\begin{cases} ${points.map(p => `f(${p.x}) = ${p.y}`).join(' \\\\ ')} \\end{cases}`} />
          </Box>
          {result.formula && (<Box sx={{ mt: 3 }}><Typography variant="h6">Công thức:</Typography><FormulaRenderer latex={result.formula} /></Box>)}
          {result.polynomial && (<Box sx={{ mt: 2 }}><Typography variant="h6">Đa thức nội suy:</Typography><FormulaRenderer latex={`P(x) = ${result.polynomial}`} /></Box>)}
          {result.divided_diff_table && result.divided_diff_table.length > 0 && (
            <Box sx={{ mt: 3 }}>
              <Typography variant="h6">Bảng sai phân:</Typography>
              <Paper variant="outlined" sx={{ p: 2, overflowX: 'auto' }}>
                <Box sx={{ fontFamily: 'monospace', fontSize: '0.85rem' }}>
                  {result.divided_diff_table.map((row, ri) => (
                    <Box key={ri} sx={{ mb: 0.5 }}>
                      {row.map((v, ci) => (<span key={ci} style={{ marginRight: 16 }}>{v?.toFixed(6) ?? '-'}</span>))}
                    </Box>
                  ))}
                </Box>
              </Paper>
            </Box>
          )}
        </>
      )}
    </Container>
  );
}