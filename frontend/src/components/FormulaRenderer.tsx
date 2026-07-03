import { useEffect, useRef, useState } from 'react';
import { Box, Paper, Typography } from '@mui/material';
import { toLatex } from '../utils/latex';

declare global {
  interface Window {
    MathJax?: {
      typesetPromise?: (elements?: HTMLElement[]) => Promise<void>;
      typeset?: (elements?: HTMLElement[]) => void;
      startup?: { promise: Promise<void> };
    };
  }
}

interface FormulaRendererProps {
  latex: string;
  display?: boolean;
}

export default function FormulaRenderer({ latex, display = true }: FormulaRendererProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [mathJaxReady, setMathJaxReady] = useState(false);

  useEffect(() => {
    const checkMathJax = () => {
      if (window.MathJax?.startup?.promise) {
        window.MathJax.startup.promise.then(() => {
          setMathJaxReady(true);
        });
      } else if (window.MathJax?.typesetPromise) {
        setMathJaxReady(true);
      } else {
        setTimeout(checkMathJax, 300);
      }
    };
    checkMathJax();
  }, []);

  useEffect(() => {
    if (!containerRef.current || !latex) return;

    // Auto-convert raw math expression to LaTeX if it doesn't already contain LaTeX commands
    const latexString = latex.includes('\\') ? latex : toLatex(latex);

    containerRef.current.innerHTML = display
      ? `\\[${latexString}\\]`
      : `\\(${latexString}\\)`;

    if (window.MathJax?.typesetPromise) {
      window.MathJax.typesetPromise([containerRef.current]).catch(() => {});
    }
  }, [latex, display, mathJaxReady]);

  if (!latex) return null;

  return (
    <Paper
      variant="outlined"
      sx={{
        p: 2,
        my: 1,
        textAlign: 'center',
        backgroundColor: (theme) =>
          theme.palette.mode === 'dark' ? 'grey.900' : 'grey.50',
        overflow: 'auto',
      }}
    >
      {!mathJaxReady ? (
        <Typography variant="body2" color="text.secondary">
          ⏳ Đang tải MathJax...
        </Typography>
      ) : (
        <Box ref={containerRef} sx={{ fontSize: '1.15rem', minHeight: '1.5em', py: 0.5 }} />
      )}
    </Paper>
  );
}