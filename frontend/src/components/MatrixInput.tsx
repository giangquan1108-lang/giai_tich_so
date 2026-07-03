import { useState, useEffect } from 'react';
import { Box, TextField, Typography } from '@mui/material';
import FormulaRenderer from './FormulaRenderer';

interface MatrixInputProps {
  size: number;
  matrix: number[][];
  vector: number[];
  onMatrixChange: (matrix: number[][]) => void;
  onVectorChange: (vector: number[]) => void;
  onSizeChange?: (size: number) => void;
  showPreview?: boolean;
}

export default function MatrixInput({
  size,
  matrix,
  vector,
  onMatrixChange,
  onVectorChange,
  showPreview = true,
}: MatrixInputProps) {
  // Internal string state for real-time typing (enables negative numbers)
  const [matrixStrings, setMatrixStrings] = useState<string[][]>(
    matrix.map(row => row.map(v => String(v)))
  );
  const [vectorStrings, setVectorStrings] = useState<string[]>(
    vector.map(v => String(v))
  );

  useEffect(() => {
    setMatrixStrings(matrix.map(row => row.map(v => String(v))));
    setVectorStrings(vector.map(v => String(v)));
  }, [matrix, vector]);

  const handleMatrixValueChange = (row: number, col: number, value: string) => {
    const newStrings = matrixStrings.map(r => [...r]);
    newStrings[row][col] = value;
    setMatrixStrings(newStrings);
    const newMatrix = matrix.map((r) => [...r]);
    newMatrix[row][col] = parseFloat(value) || 0;
    onMatrixChange(newMatrix);
  };

  const handleVectorValueChange = (idx: number, value: string) => {
    const newStrings = [...vectorStrings];
    newStrings[idx] = value;
    setVectorStrings(newStrings);
    const newVector = [...vector];
    newVector[idx] = parseFloat(value) || 0;
    onVectorChange(newVector);
  };

  // Generate LaTeX for matrix preview
  const matrixLatex = matrix.length > 0 && matrix[0].length > 0
    ? `A = \\begin{pmatrix} ${matrix.map(row => row.join(' & ')).join(' \\\\ ')} \\end{pmatrix}, \\quad B = \\begin{pmatrix} ${vector.join(' \\\\ ')} \\end{pmatrix}`
    : '';

  return (
    <Box>
      <Typography variant="subtitle2" gutterBottom>
        Ma trận A ({size}×{size})
      </Typography>
      <Box sx={{ overflowX: 'auto', mb: 2 }}>
        {matrix.map((row, i) => (
          <Box key={i} sx={{ display: 'flex', gap: 1, mb: 0.5 }}>
            {row.map((_val, j) => (
              <TextField
                key={j}
                size="small"
                type="text"
                inputMode="decimal"
                value={matrixStrings[i]?.[j] ?? ''}
                onChange={(e) => handleMatrixValueChange(i, j, e.target.value)}
                sx={{ width: 70 }}
                slotProps={{ input: { style: { textAlign: 'center' } } }}
              />
            ))}
            <Typography sx={{ alignSelf: 'center', mx: 1 }}>|</Typography>
            <TextField
              size="small"
              type="text"
              inputMode="decimal"
              value={vectorStrings[i] ?? ''}
              onChange={(e) => handleVectorValueChange(i, e.target.value)}
              sx={{ width: 70 }}
              slotProps={{ input: { style: { textAlign: 'center' } } }}
              label={`b${i + 1}`}
            />
          </Box>
        ))}
      </Box>

      {/* ✅ XEM TRƯỚC MA TRẬN DẠNG LaTeX - cập nhật theo thời gian thực */}
      {showPreview && matrixLatex && (
        <Box sx={{ pt: 1, borderTop: '1px solid', borderColor: 'divider' }}>
          <Typography variant="subtitle2" color="text.secondary" gutterBottom>
            📐 Xem trước ma trận:
          </Typography>
          <FormulaRenderer latex={matrixLatex} />
        </Box>
      )}
    </Box>
  );
}