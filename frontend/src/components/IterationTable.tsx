import { useRef, useEffect } from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Typography,
} from '@mui/material';

interface IterationTableProps {
  data: Record<string, unknown>[];
  title?: string;
  /** Map of key → LaTeX header label */
  headerMap?: Record<string, string>;
}

const DEFAULT_HEADER_MAP: Record<string, string> = {
  k: 'k',
  a: 'a',
  b: 'b',
  x_k: 'x_k',
  f_x_k: 'f(x_k)',
  f_prime_x_k: "f'(x_k)",
  x_prev: 'x_{k-1}',
  x_next: 'x_{k+1}',
  g_x_k: 'g(x_k)',
  error: '\\varepsilon',
  basis_index: 'i',
  l_i: 'l_i(x)',
  y_i: 'y_i',
  order: 'j',
  differences: '\\Delta^j f',
  level: 'i',
  h: 'h',
  R_i0: 'R_{i,0}',
  extrapolations: 'R_{i,j}',
  description: 'Description',
  sum_f: '\\Sigma f(x)',
  range: '[x_i, x_{i+1}]',
  // Newton multivariable
  x: 'X^{(k)}',
  F_x: 'F(X^{(k)})',
  jacobian: 'J(X^{(k)})',
  dx: '\\Delta X',
  x_new: 'X^{(k+1)}',
  G_x: 'G(X^{(k)})',
  // Integration
  'f(a)': 'f(a)',
  'f(b)': 'f(b)',
  '4*odd_sum': '4\\sum_{odd}',
  '2*even_sum': '2\\sum_{even}',
  total: '\\Sigma',
};

export default function IterationTable({ data, title, headerMap }: IterationTableProps) {
  const headerRefs = useRef<Map<string, HTMLSpanElement>>(new Map());

  if (!data || data.length === 0) return null;

  const headers = Object.keys(data[0]);
  const effectiveMap = { ...DEFAULT_HEADER_MAP, ...headerMap };

  // Type LaTeX in headers after render
  useEffect(() => {
    try {
      headerRefs.current.forEach((el, key) => {
        const latex = effectiveMap[key] || key;
        if (latex === key) return;
        el.innerHTML = `\\(${latex}\\)`;
      });
      if ((window as any).MathJax?.typesetPromise) {
        const elements = Array.from(headerRefs.current.values());
        (window as any).MathJax.typesetPromise(elements).catch(() => {});
      }
    } catch {
      // MathJax may not be ready — degrade gracefully
    }
  }, [data, headerMap]);

  return (
    <>
      {title && (
        <Typography variant="h6" sx={{ mt: 3, mb: 1 }}>
          {title}
        </Typography>
      )}
      <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 400 }}>
        <Table stickyHeader size="small">
          <TableHead>
            <TableRow>
              {headers.map((header) => (
                <TableCell key={header} sx={{ fontWeight: 700, whiteSpace: 'nowrap' }}>
                  <span
                    ref={(el) => {
                      if (el) headerRefs.current.set(header, el);
                    }}
                  >
                    {effectiveMap[header] && effectiveMap[header] !== header
                      ? `\\(${effectiveMap[header]}\\)`
                      : header}
                  </span>
                </TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {data.map((row, idx) => (
              <TableRow key={idx} hover>
                {headers.map((header) => (
                  <TableCell key={header}>
                    {formatCell(row[header])}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </>
  );
}

function formatCell(value: unknown): string {
  if (value === null || value === undefined) return '-';
  if (typeof value === 'number') {
    if (Number.isInteger(value)) return value.toString();
    if (Math.abs(value) < 1e-10) return '0';
    if (Math.abs(value) >= 1e4) return value.toExponential(6);
    return value.toFixed(7);
  }
  if (typeof value === 'string') return value;
  if (Array.isArray(value)) {
    if (value.every((v) => typeof v === 'number')) {
      return `[${value.map(v => v.toFixed(7)).join(', ')}]`;
    }
    return `[${value.join(', ')}]`;
  }
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}
