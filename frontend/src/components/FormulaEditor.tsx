import { useState, useRef, useEffect, useCallback } from 'react';
import { Box, TextField, Typography, IconButton, Tooltip, Paper } from '@mui/material';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import ClearIcon from '@mui/icons-material/Clear';
import { isValidLatex, toLatex } from '../utils/latex';

interface FormulaEditorProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  label?: string;
  minRows?: number;
  /** Hàm bổ sung để convert giá trị LaTeX trước khi gửi (nếu cần Python expression) */
  onCopyLatex?: (latex: string) => void;
}

export default function FormulaEditor({
  value,
  onChange,
  placeholder = 'Nhập công thức LaTeX (VD: x^2 + \\sin(x) - 1)',
  label = 'Công thức LaTeX',
  minRows = 2,
  onCopyLatex,
}: FormulaEditorProps) {
  const [isValid, setIsValid] = useState(true);
  const [copied, setCopied] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    setIsValid(value.trim() === '' || isValidLatex(value));
  }, [value]);

  const handleClear = useCallback(() => {
    onChange('');
    if (textareaRef.current) textareaRef.current.focus();
  }, [onChange]);

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(value);
    setCopied(true);
    if (onCopyLatex) onCopyLatex(value);
    setTimeout(() => setCopied(false), 1500);
  }, [value, onCopyLatex]);

  return (
    <Paper variant="outlined" sx={{ p: 1.5, mb: 1 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 0.5 }}>
        <Typography variant="subtitle2" color="text.secondary">
          📐 {label}
        </Typography>
        <Box>
          {value && (
            <>
              <Tooltip title={copied ? '✓ Đã copy!' : 'Sao chép LaTeX'}>
                <IconButton size="small" onClick={handleCopy}>
                  <ContentCopyIcon fontSize="small" />
                </IconButton>
              </Tooltip>
              <Tooltip title="Xóa">
                <IconButton size="small" onClick={handleClear}>
                  <ClearIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </>
          )}
        </Box>
      </Box>

      {/* Textarea nhập LaTeX */}
      <TextField
        inputRef={textareaRef}
        fullWidth
        multiline
        minRows={minRows}
        maxRows={8}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        error={!isValid && value.trim() !== ''}
        helperText={!isValid && value.trim() !== '' ? '⚠️ Cú pháp LaTeX không hợp lệ (kiểm tra dấu ngoặc {})' : ''}
        sx={{
          '& .MuiInputBase-root': {
            fontFamily: '"JetBrains Mono", "Fira Code", "Consolas", monospace',
            fontSize: '0.95rem',
          },
        }}
        slotProps={{
          input: {
            sx: { fontFamily: 'monospace', fontSize: '0.95rem' },
          },
        }}
      />

      {/* Preview MathJax realtime */}
      {value.trim() && (
        <Box
          sx={{
            mt: 1.5,
            pt: 1.5,
            borderTop: '1px solid',
            borderColor: 'divider',
            minHeight: 48,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            bgcolor: (theme) =>
              theme.palette.mode === 'dark' ? 'grey.900' : 'grey.50',
            borderRadius: 1,
            p: 1,
          }}
        >
          <MathJaxPreview latex={toLatex(value)} />
        </Box>
      )}
    </Paper>
  );
}

/**
 * Inline MathJax preview component that handles dynamic re-rendering
 */
function MathJaxPreview({ latex }: { latex: string }) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    containerRef.current.innerHTML = `\\[${latex}\\]`;
    if (window.MathJax?.typesetPromise) {
      window.MathJax.typesetPromise([containerRef.current]).catch(() => {});
    }
  }, [latex]);

  return (
    <Box
      ref={containerRef}
      sx={{
        fontSize: '1.2rem',
        textAlign: 'center',
        width: '100%',
        overflow: 'auto',
        py: 0.5,
      }}
    />
  );
}