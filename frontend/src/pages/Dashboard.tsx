import { Box, Grid, Card, CardContent, Typography, CardActionArea, Container } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { modules } from '../config/modules';

export default function Dashboard() {
  const navigate = useNavigate();

  const dashboardCards = modules.filter(m => m.showInDashboard);

  return (
    <Container maxWidth="lg">
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" sx={{ fontWeight: 700, mb: 1 }}>
          📐 Numerical Analysis Laboratory
        </Typography>
        <Typography variant="h6" color="text.secondary">
          Nền tảng học tập và thực hành Giải tích số
        </Typography>
      </Box>

      <Grid container spacing={3}>
        {dashboardCards.map((mod) => (
          <Grid key={mod.path} size={{ xs: 12, sm: 6, md: 4 }}>
            <Card
              variant="outlined"
              sx={{
                height: '100%',
                transition: '0.3s',
                '&:hover': {
                  boxShadow: 6,
                  transform: 'translateY(-4px)',
                },
              }}
            >
              <CardActionArea onClick={() => navigate(mod.path)} sx={{ p: 3 }}>
                <CardContent>
                  <Box sx={{ color: mod.color ?? 'primary.main', mb: 2 }}>{mod.icon}</Box>
                  <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
                    {mod.title}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {mod.description}
                  </Typography>
                </CardContent>
              </CardActionArea>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Box sx={{ mt: 6, p: 3, bgcolor: (t) => t.palette.mode === 'dark' ? 'grey.900' : 'grey.50', borderRadius: 2 }}>
        <Typography variant="h5" sx={{ fontWeight: 600, mb: 2 }}>
          🎯 Mục tiêu
        </Typography>
        <Typography variant="body1" sx={{ mb: 1 }}>
          • Nhập dữ liệu toán học và thực hiện các thuật toán giải tích số
        </Typography>
        <Typography variant="body1" sx={{ mb: 1 }}>
          • Quan sát toàn bộ quá trình tính toán từng bước
        </Typography>
        <Typography variant="body1" sx={{ mb: 1 }}>
          • Xem sai số và kết quả trực quan
        </Typography>
        <Typography variant="body1">
          • So sánh hiệu năng các phương pháp khác nhau
        </Typography>
      </Box>
    </Container>
  );
}