import { useState } from 'react';
import {
  Box, Typography, Paper, Chip, Collapse, IconButton, Button, Stack,
  Tooltip,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import FormulaRenderer from './FormulaRenderer';

interface StepItem {
  step: number;
  description: string;
  matrix?: number[][];
  solution?: number[][];
  solution_vector?: number[];
  solution_lines?: string[];
  back_substitution?: string[];
  row_swap?: boolean;
  row1?: number;
  row2?: number;
  row_operations?: { target_row: number; factor: number; pivot_row: number }[];
  pivot_col?: number;
  pivot_row?: number;
  phase?: string;
  inverse?: number[][];
  forward_sweep?: { alpha: number; beta: number }[];
}

interface Props {
  steps: StepItem[];
  method: string;
  nCols?: number; // number of solution columns (for multi-column B)
}

const phaseColors: Record<string, { bg: string; label: string; color: string }> = {
  row_swap: { bg: '#fff3e0', label: 'Đổi hàng', color: '#e65100' },
  elimination: { bg: '#e3f2fd', label: 'Khử', color: '#0d47a1' },
  normalize: { bg: '#e8f5e9', label: 'Chuẩn hóa', color: '#1b5e20' },
  upper_triangular: { bg: '#fce4ec', label: 'Tam giác trên', color: '#880e4f' },
  back_substitution: { bg: '#ede7f6', label: 'Thế ngược', color: '#311b92' },
  rref: { bg: '#e0f7fa', label: 'RREF', color: '#004d40' },
  solution: { bg: '#f1f8e9', label: 'Nghiệm', color: '#33691e' },
};

function renderMatrix(mat: number[][], precision: number = 6) {
  if (!mat || mat.length === 0) return null;
  return (
    <Box sx={{ fontFamily: '"Fira Code", "Consolas", monospace', fontSize: '0.82rem', overflowX: 'auto', mt: 0.5 }}>
      {mat.map((row, ri) => (
        <Box key={ri} sx={{ whiteSpace: 'nowrap' }}>
          |{' '}
          {row.map((v, ci) => (
            <span key={ci}>
              {typeof v === 'number'
                ? (Math.abs(v) < 1e-15 ? ' 0.'.padEnd(precision + 5) + ' ' : v.toFixed(precision).padStart(precision + 3))
                : String(v).padStart(precision + 3)}{' '}
              {ci < row.length - 1 ? '' : '|'}
            </span>
          ))}
        </Box>
      ))}
    </Box>
  );
}

function methodTitle(method: string): string {
  const map: Record<string, string> = {
    gaussian: 'Phương pháp khử Gauss (Gaussian Elimination)',
    gauss_jordan: 'Phương pháp Gauss-Jordan',
    lu: 'Phương pháp phân rã LU (Doolittle)',
    cholesky: 'Phương pháp Cholesky',
    thomas: 'Thuật toán Thomas (TDMA)',
    jacobi: 'Phương pháp lặp Jacobi',
    gauss_seidel: 'Phương pháp lặp Gauss-Seidel',
    sor: 'Phương pháp SOR (Successive Over-Relaxation)',
  };
  return map[method] || method;
}

export default function SolutionSteps({ steps, method, nCols }: Props) {
  const [expanded, setExpanded] = useState(true);
  const [expandedPhases, setExpandedPhases] = useState<Record<string, boolean>>({});

  const togglePhase = (phase: string) => {
    setExpandedPhases(prev => ({ ...prev, [phase]: !prev[phase] }));
  };

  const isPhaseExpanded = (phase: string) => expandedPhases[phase] !== false; // default expanded

  // Group steps by phase
  const phases: { phase: string; steps: StepItem[] }[] = [];
  let currentPhase = '';
  let currentSteps: StepItem[] = [];

  for (const step of steps) {
    const phase = step.phase || '';
    if (phase && phase !== currentPhase) {
      if (currentSteps.length > 0) {
        phases.push({ phase: currentPhase || 'steps', steps: currentSteps });
      }
      currentPhase = phase;
      currentSteps = [step];
    } else {
      currentSteps.push(step);
    }
  }
  if (currentSteps.length > 0) {
    phases.push({ phase: currentPhase || 'steps', steps: currentSteps });
  }

  const scrollToPhase = (phaseKey: string) => {
    const el = document.getElementById(`phase-${phaseKey}`);
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  // Generate LaTeX for copy
  const generateLatex = (): string => {
    const lines: string[] = [];
    lines.push(`\\textbf{${methodTitle(method)}}`);
    lines.push('');
    for (const step of steps) {
      lines.push(`\\textbf{Bước ${step.step}:} ${step.description}`);
      if (step.matrix) {
        lines.push('$$');
        lines.push('\\begin{bmatrix}');
        lines.push(
          step.matrix.map(row => row.map(v => v.toFixed(4)).join(' & ')).join(' \\\\ ')
        );
        lines.push('\\end{bmatrix}');
        lines.push('$$');
      }
      if (step.back_substitution) {
        lines.push('\\begin{aligned}');
        step.back_substitution.forEach(s => {
          lines.push(s.replace(/·/g, '\\cdot') + ' \\\\');
        });
        lines.push('\\end{aligned}');
      }
      if (step.solution_lines) {
        lines.push('\\begin{aligned}');
        step.solution_lines.forEach(s => lines.push(s + ' \\\\'));
        lines.push('\\end{aligned}');
      }
      if (step.solution) {
        lines.push('$$');
        lines.push('X = \\begin{bmatrix}');
        lines.push(step.solution.map(row => row.map(v => v.toFixed(6)).join(' & ')).join(' \\\\ '));
        lines.push('\\end{bmatrix}');
        lines.push('$$');
      }
      lines.push('');
    }
    return lines.join('\n');
  };

  const copyLatex = () => {
    const latex = generateLatex();
    navigator.clipboard.writeText(latex).then(() => {
      // brief visual feedback handled by button
    });
  };

  const exportPdf = () => {
    const content = generateLatex();
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `solution_${method}.tex`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <Box sx={{ mt: 3 }}>
      {/* Header with collapsible toggle */}
      <Paper sx={{ p: 2, mb: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 1 }}>
        <Stack direction="row" spacing={1} sx={{ alignItems: 'center' }}>
          <IconButton size="small" onClick={() => setExpanded(!expanded)}>
            <ExpandMoreIcon sx={{
              transform: expanded ? 'rotate(0deg)' : 'rotate(-90deg)',
              transition: 'transform 0.3s',
            }} />
          </IconButton>
          <Typography variant="h6" sx={{ fontWeight: 700 }}>
            📝 Chi tiết lời giải
          </Typography>
          <Chip label={methodTitle(method)} size="small" color="primary" variant="outlined" />
        </Stack>
        <Stack direction="row" spacing={1}>
          <Tooltip title="Copy LaTeX">
            <Button size="small" startIcon={<ContentCopyIcon />} onClick={copyLatex} variant="outlined">
              Copy LaTeX
            </Button>
          </Tooltip>
          <Tooltip title="Xuất file .tex">
            <Button size="small" startIcon={<PictureAsPdfIcon />} onClick={exportPdf} variant="outlined" color="secondary">
              Xuất .tex
            </Button>
          </Tooltip>
        </Stack>
      </Paper>

      <Collapse in={expanded}>
        {/* Phase navigation pills */}
        {phases.length > 1 && (
          <Box sx={{ mb: 2, display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
            {phases.map(({ phase }) => {
              const colors = phaseColors[phase] || { bg: '#f5f5f5', label: phase, color: '#616161' };
              return (
                <Chip
                  key={phase}
                  label={colors.label}
                  size="small"
                  onClick={() => scrollToPhase(phase)}
                  sx={{ bgcolor: colors.bg, color: colors.color, cursor: 'pointer', fontWeight: 500 }}
                />
              );
            })}
          </Box>
        )}

        {/* Phase groups */}
        {phases.map(({ phase, steps: phaseSteps }) => (
          <Box key={phase} id={`phase-${phase}`} sx={{ mb: 2 }}>
            {phases.length > 1 && (
              <Typography
                variant="subtitle2"
                sx={{
                  mb: 1, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 0.5,
                  color: (phaseColors[phase] || { color: '#616161' }).color,
                }}
                onClick={() => togglePhase(phase)}
              >
                <ExpandMoreIcon sx={{
                  fontSize: 18,
                  transform: isPhaseExpanded(phase) ? 'rotate(0deg)' : 'rotate(-90deg)',
                  transition: 'transform 0.2s',
                }} />
                {(phaseColors[phase] || { label: phase }).label}
              </Typography>
            )}
            <Collapse in={isPhaseExpanded(phase)}>
              <Stack spacing={1}>
                {phaseSteps.map((step, idx) => (
                  <StepCard key={idx} step={step} method={method} nCols={nCols} />
                ))}
              </Stack>
            </Collapse>
          </Box>
        ))}
      </Collapse>
    </Box>
  );
}

function StepCard({ step, method, nCols }: { step: StepItem; method: string; nCols?: number }) {
  const phaseInfo = step.phase ? phaseColors[step.phase] : null;

  return (
    <Paper
      variant="outlined"
      sx={{
        p: 1.5,
        borderLeft: phaseInfo ? `4px solid ${phaseInfo.color}` : undefined,
        bgcolor: phaseInfo ? phaseInfo.bg : 'background.paper',
      }}
    >
      {/* Step header */}
      <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 0.5, fontSize: '0.85rem' }}>
        Bước {step.step}: {step.description}
      </Typography>

      {/* LaTeX row operations (for Gauss) */}
      {(step as any).row_operations_latex && (step as any).row_operations_latex.length > 0 && (
        <Box sx={{ mb: 0.5 }}>
          {(step as any).row_operations_latex.map((latex: string, i: number) => (
            <Box key={i} sx={{ mb: 0.2 }}>
              <FormulaRenderer latex={`$${latex}$`} display={false} />
            </Box>
          ))}
        </Box>
      )}

      {/* Row operations detail (non-LaTeX fallback) */}
      {!((step as any).row_operations_latex) && step.row_operations && step.row_operations.length > 0 && (
        <Box sx={{ mb: 0.5 }}>
          {step.row_operations.map((op, i) => (
            <Chip
              key={i}
              size="small"
              label={`R${op.target_row} = R${op.target_row} − ${op.factor}·R${op.pivot_row}`}
              sx={{ mr: 0.5, mb: 0.5, fontFamily: 'monospace', fontSize: '0.75rem' }}
            />
          ))}
        </Box>
      )}

      {/* LaTeX matrix display (for Gauss) */}
      {(step as any).matrix_latex ? (
        <Box sx={{ mt: 1, overflowX: 'auto' }}>
          <FormulaRenderer latex={`$${(step as any).matrix_latex}$`} display={true} />
        </Box>
      ) : (
        /* Plain matrix fallback */
        step.matrix && renderMatrix(step.matrix)
      )}

      {/* Back substitution / Solution LaTeX (for Gauss back-substitution & Gauss-Jordan solution) */}
      {(step as any).back_substitution_latex && (
        <Box sx={{ mt: 1, bgcolor: 'rgba(0,0,0,0.03)', p: 1.5, borderRadius: 1 }}>
          <Typography variant="caption" sx={{ fontWeight: 600, display: 'block', mb: 0.5 }}>
            {step.phase === 'solution' ? '✅ Nghiệm:' : '🔄 Thế ngược:'}
          </Typography>
          <FormulaRenderer latex={`$${(step as any).back_substitution_latex}$`} display={true} />
        </Box>
      )}

      {/* Back substitution text fallback */}
      {!((step as any).back_substitution_latex) && step.back_substitution && step.back_substitution.length > 0 && (
        <Box sx={{ mt: 1, bgcolor: 'rgba(0,0,0,0.03)', p: 1, borderRadius: 1 }}>
          <Typography variant="caption" sx={{ fontWeight: 600, display: 'block', mb: 0.5 }}>
            🔄 Thế ngược:
          </Typography>
          {step.back_substitution.map((line, i) => (
            <Typography key={i} variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.8rem', lineHeight: 1.6 }}>
              {line}
            </Typography>
          ))}
        </Box>
      )}

      {/* Solution lines (Gauss-Jordan RREF) */}
      {step.solution_lines && step.solution_lines.length > 0 && (
        <Box sx={{ mt: 1, bgcolor: 'rgba(0,0,0,0.03)', p: 1, borderRadius: 1 }}>
          <Typography variant="caption" sx={{ fontWeight: 600, display: 'block', mb: 0.5 }}>
            ✅ Nghiệm:
          </Typography>
          {step.solution_lines.map((line, i) => (
            <Typography key={i} variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.85rem', lineHeight: 1.6, fontWeight: 600 }}>
              {line}
            </Typography>
          ))}
        </Box>
      )}

      {/* Solution vector (single) */}
      {step.solution_vector && !step.solution_lines && (
        <Box sx={{ mt: 0.5 }}>
          <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.85rem', fontWeight: 600 }}>
            X = [{step.solution_vector.join(', ')}]<sup>T</sup>
          </Typography>
        </Box>
      )}

      {/* Solution matrix */}
      {step.solution && (
        <Box sx={{ mt: 1 }}>
          <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.5 }}>Kết quả:</Typography>
          {renderMatrix(step.solution)}
        </Box>
      )}

      {/* Inverse matrix */}
      {step.inverse && (
        <Box sx={{ mt: 1 }}>
          <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.5 }}>A⁻¹:</Typography>
          {renderMatrix(step.inverse)}
        </Box>
      )}
    </Paper>
  );
}
