import { Card, CardContent, Typography, Box, Chip } from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';

interface ResultCardProps {
  success: boolean;
  message: string;
  result?: Record<string, unknown>;
  children?: React.ReactNode;
}

export default function ResultCard({ success, message, result, children }: ResultCardProps) {
  return (
    <Card variant="outlined" sx={{ mt: 3, borderLeft: 4, borderColor: success ? 'success.main' : 'error.main' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
          {success ? (
            <CheckCircleIcon color="success" />
          ) : (
            <ErrorIcon color="error" />
          )}
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Kết quả
          </Typography>
          <Chip
            label={success ? 'Thành công' : 'Thất bại'}
            color={success ? 'success' : 'error'}
            size="small"
          />
        </Box>
        <Typography variant="body1" sx={{ mb: 2 }}>
          {message}
        </Typography>
        {result && (
          <Box sx={{ mb: 2 }}>
            {Object.entries(result).map(([key, value]) => (
              <Box key={key} sx={{ display: 'flex', gap: 1, mb: 0.5 }}>
                <Typography variant="body2" sx={{ fontWeight: 600, minWidth: 120 }}>
                  {key}:
                </Typography>
                <Typography variant="body2">
                  {typeof value === 'number' ? value.toFixed(10) : String(value)}
                </Typography>
              </Box>
            ))}
          </Box>
        )}
        {children}
      </CardContent>
    </Card>
  );
}