import { useState } from 'react';
import {
  Container, Typography, TextField, Select, MenuItem, FormControl,
  InputLabel, Button, Grid, Paper, Box, CircularProgress, Alert,
} from '@mui/material';
import { nonlinearSystemAPI, type NonlinearSystemResponse } from '../services/api';
import FormulaEditor from '../components/FormulaEditor';
import FormulaRenderer from '../components/FormulaRenderer';
import IterationTable from '../components/IterationTable';
import ResultCard from '../components/ResultCard';

const methods = [
  { value: 'newton', label: 'Newton Multivariable', description: 'Giải F(X)=0 dùng Jacobian' },
  { value: 'fixed_point', label: 'Fixed Point Multivariable', description: 'Lặp X=G(X)' },
];

const newtonExample = ['x^2 + y^2 - 4', 'x - y'];
const fixedPointExample = ['cos(y)', 'sin(x)'];

export default function NonlinearSystem() {
  const [numVars, setNumVars] = useState(2);
  const [initialGuess, setInitialGuess] = useState(['1.0', '1.0']);
  const [epsilon, setEpsilon] = useState('0.000001');
  const [maxIter, setMaxIter] = useState('100');
  const [method, setMethod] = useState('newton');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<NonlinearSystemResponse | null>(null);

  const isNewton = method === 'newton';
  const isFixedPoint = method === 'fixed_point';
  const variables = ['x', 'y', 'z'].slice(0, numVars);

  const [functions, setFunctions] = useState(newtonExample);

  const handleMethodChange = (newMethod: string) => {
    setMethod(newMethod);
    if (newMethod === 'fixed_point') {
      setFunctions(fixedPointExample.slice(0, numVars));
    } else {
      setFunctions(newtonExample.slice(0, numVars));
    }
  };

  const handleNumVarsChange = (n: number) => {
    setNumVars(n);
    if (isFixedPoint) {
      setFunctions(fixedPointExample.slice(0, n));
    } else {
      setFunctions(newtonExample.slice(0, n));
    }
    setInitialGuess(Array(n).fill('1.0'));
  };

  const handleGuessChange = (idx: number, value: string) => {
    const newGuess = [...initialGuess];
    newGuess[idx] = value;
    setInitialGuess(newGuess);
  };

  const handleSolve = async () => {
    setLoading(true);
    try {
      const response = await nonlinearSystemAPI.solve({
        functions: functions.filter(f => f.trim()),
        variables: variables.slice(0, numVars),
        initial_guess: initialGuess.slice(0, numVars).map(v => parseFloat(v) || 0),
        epsilon: parseFloat(epsilon) || 1e-6,
        max_iterations: parseInt(maxIter) || 100,
        method,
      });
      setResult(response.data);
    } catch {
      setResult({
        success: false, message: 'Lỗi kết nối server.',
        iterations_count: 0, final_error: 0, iterations: [], formula: '',
      });
    }
    setLoading(false);
  };

  const isFixedPointValid = () => {
    if (!isFixedPoint) return true;
    const activeFuncs = functions.filter(f => f.trim());
    if (activeFuncs.length === 0) return false;
    return activeFuncs.every(f => variables.some(v => f.includes(v)));
  };

  return (
    <Container maxWidth="lg">
      <Typography variant="h4" sx={{ fontWeight: 700, mb: 3 }}>
        📊 Hệ phương trình phi tuyến
      </Typography>

      <Paper sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={3}>
          <Grid size={{ xs: 12, md: 4 }}>
            <FormControl fullWidth>
              <InputLabel>Số biến</InputLabel>
              <Select value={numVars} label="Số biến"
                onChange={(e) => handleNumVarsChange(Number(e.target.value))}>
                {[2, 3, 4].map(n => <MenuItem key={n} value={n}>{n}</MenuItem>)}
              </Select>
            </FormControl>
          </Grid>
          <Grid size={{ xs: 12, md: 4 }}>
            <FormControl fullWidth>
              <InputLabel>Phương pháp</InputLabel>
              <Select value={method} label="Phương pháp"
                onChange={(e) => handleMethodChange(e.target.value)}>
                {methods.map(m => (
                  <MenuItem key={m.value} value={m.value}>
                    {m.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid size={{ xs: 6, md: 2 }}>
            <TextField fullWidth type="text" inputMode="decimal" label="Epsilon" value={epsilon}
              onChange={(e) => setEpsilon(e.target.value)} />
          </Grid>
          <Grid size={{ xs: 6, md: 2 }}>
            <TextField fullWidth type="text" inputMode="numeric" label="Max Iter" value={maxIter}
              onChange={(e) => setMaxIter(e.target.value)} />
          </Grid>

          {/* Method description */}
          <Grid size={{ xs: 12 }}>
            <Alert severity="info" sx={{ mb: 0 }}>
              {methods.find(m => m.value === method)?.description}
              {isNewton && ' — Nhập các phương trình dạng F(X)=0.'}
              {isFixedPoint && ' — Nhập các hàm G(X) sao cho X = G(X). Ví dụ: x = 4/(x-y).'}
            </Alert>
          </Grid>

          {/* Fixed Point warning */}
          {isFixedPoint && !isFixedPointValid() && (
            <Grid size={{ xs: 12 }}>
              <Alert severity="warning">
                Fixed Point Iteration yêu cầu nhập g(x,y) sao cho x=g₁(x,y), y=g₂(x,y), không phải f(x,y)=0.
                Biểu thức phải chứa biến (x, y, ...).
              </Alert>
            </Grid>
          )}

          {/* Function inputs */}
          {Array.from({ length: numVars }, (_, i) => {
            const varList = variables.join(',');
            const label = isNewton
              ? `f${i + 1}(${varList}) = 0`
              : `g${i + 1}(${varList}) → ${variables[i]} = g${i + 1}(${varList})`;

            const placeholder = isNewton
              ? `VD: ${newtonExample[i % newtonExample.length]}`
              : `VD: ${fixedPointExample[i % fixedPointExample.length]}`;

            return (
              <Grid key={i} size={{ xs: 12 }}>
                <FormulaEditor
                  value={functions[i] || ''}
                  onChange={(v) => {
                    const newFuncs = [...functions];
                    newFuncs[i] = v;
                    setFunctions(newFuncs);
                  }}
                  placeholder={placeholder}
                  label={label}
                  minRows={2}
                />
              </Grid>
            );
          })}

          {/* Initial guesses */}
          {Array.from({ length: numVars }, (_, i) => (
            <Grid key={`ig-${i}`} size={{ xs: 6, md: 3 }}>
              <TextField fullWidth type="text" inputMode="decimal" label={`${variables[i]}₀`}
                value={initialGuess[i] ?? ''}
                onChange={(e) => handleGuessChange(i, e.target.value)} />
            </Grid>
          ))}

          <Grid size={{ xs: 12 }}>
            <Button variant="contained" size="large" onClick={handleSolve}
              disabled={loading || (isFixedPoint && functions.filter(f => f.trim()).length === 0)}
              startIcon={loading ? <CircularProgress size={20} /> : null}>
              Tính toán
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {result && (
        <>
          <ResultCard success={result.success} message={result.message}>
            {result.success && result.solution && (
              <Box sx={{ mt: 1 }}>
                <Typography variant="h5" sx={{ fontWeight: 600, color: 'success.main' }}>
                  Nghiệm: ({result.solution.map(v => v.toFixed(7)).join(', ')})
                </Typography>
                <Typography>Số vòng lặp: {result.iterations_count} | Sai số: {result.final_error != null ? result.final_error.toExponential(4) : 'N/A'}</Typography>
              </Box>
            )}
            {!result.success && result.solution && (
              <Box sx={{ mt: 1 }}>
                <Typography variant="h6" sx={{ fontWeight: 600, color: 'warning.main' }}>
                  Giá trị cuối cùng (chưa hội tụ):
                </Typography>
                <Typography sx={{ fontFamily: 'monospace' }}>
                  ({result.solution.map(v => v.toFixed(7)).join(', ')})
                </Typography>
                <Typography sx={{ mt: 1 }}>
                  Số vòng lặp đã thực hiện: {result.iterations_count} | Sai số cuối: {result.final_error != null ? result.final_error.toExponential(4) : 'N/A'}
                </Typography>
              </Box>
            )}
            {/* Stopping criterion explanation */}
            {result.stopping_criterion && (
              <Alert severity={result.success ? 'success' : 'info'} sx={{ mt: 1 }}>
                <strong>Điều kiện dừng:</strong> {result.stopping_criterion}
              </Alert>
            )}
          </ResultCard>

          <Box sx={{ mt: 2 }}>
            <Typography variant="h6">Hệ phương trình:</Typography>
            {isNewton ? (
              <FormulaRenderer latex={`\\begin{cases} ${functions.filter(f => f.trim()).slice(0, numVars).join(' = 0 \\\\ ')} = 0 \\end{cases}`} />
            ) : (
              <FormulaRenderer latex={`\\begin{cases} ${variables.map((v, i) => `${v} = ${functions[i]?.trim() || '...'}`).join(' \\\\ ')} \\end{cases}`} />
            )}
          </Box>

          {result.formula && (
            <Box sx={{ mt: 3 }}>
              <Typography variant="h6">Công thức:</Typography>
              <FormulaRenderer latex={result.formula} />
            </Box>
          )}

          {result.jacobian && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="h6">Jacobian:</Typography>
              <FormulaRenderer latex={`J = \\begin{pmatrix} ${result.jacobian.map(row => row.map(v => v.toFixed(4)).join(' & ')).join(' \\\\ ')} \\end{pmatrix}`} />
            </Box>
          )}

          {result.contraction_warning && (
            <Alert severity="warning" sx={{ mt: 2 }}>
              {result.contraction_warning}
            </Alert>
          )}

          {/* Jacobian properties across iterations */}
          {result.jacobian_properties && result.jacobian_properties.length > 0 && (
            <Box sx={{ mt: 3 }}>
              <Typography variant="h6">Tính chất ma trận Jacobi:</Typography>
              <Paper variant="outlined" sx={{ p: 1.5, mt: 1, maxHeight: 300, overflow: 'auto' }}>
                <Box component="table" sx={{ borderCollapse: 'collapse', width: '100%', fontSize: '0.85rem' }}>
                  <thead>
                    <tr>
                      <th style={{ padding: '4px 8px', borderBottom: '2px solid #ccc', textAlign: 'left' }}>k</th>
                      <th style={{ padding: '4px 8px', borderBottom: '2px solid #ccc', textAlign: 'left' }}>det(J)</th>
                      <th style={{ padding: '4px 8px', borderBottom: '2px solid #ccc', textAlign: 'left' }}>cond(J)</th>
                      <th style={{ padding: '4px 8px', borderBottom: '2px solid #ccc', textAlign: 'left' }}>Trạng thái</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.jacobian_properties.map((jp: any, idx: number) => {
                      const det = jp.det;
                      const detNum = typeof det === 'number' ? det : parseFloat(det);
                      const isNearlySingular = typeof detNum === 'number' && Math.abs(detNum) < 1e-8;
                      const isIllConditioned = typeof jp.cond === 'number' && jp.cond > 1e8;
                      return (
                        <tr key={idx} style={{ borderBottom: '1px solid #eee' }}>
                          <td style={{ padding: '4px 8px' }}>{jp.k}</td>
                          <td style={{
                            padding: '4px 8px', fontFamily: 'monospace',
                            color: isNearlySingular ? '#d32f2f' : '#2e7d32',
                          }}>
                            {typeof det === 'number' ? det.toExponential(4) : det}
                          </td>
                          <td style={{
                            padding: '4px 8px', fontFamily: 'monospace',
                            color: isIllConditioned ? '#d32f2f' : 'inherit',
                          }}>
                            {typeof jp.cond === 'number' ? jp.cond.toExponential(4) : jp.cond}
                          </td>
                          <td style={{ padding: '4px 8px', fontSize: '0.8rem' }}>
                            {isNearlySingular ? '⚠ Suy biến' : isIllConditioned ? '⚠ Điều kiện xấu' : '✅ Ổn định'}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </Box>
              </Paper>
            </Box>
          )}

          <IterationTable data={result.iterations} title="Bảng quá trình lặp" />
        </>
      )}
    </Container>
  );
}