import { Container, Typography, Paper, Grid, Card, CardContent } from '@mui/material';

export default function About() {
  return (
    <Container maxWidth="lg">
      <Typography variant="h4" sx={{ fontWeight: 700, mb: 3 }}>
        ℹ️ Giới thiệu
      </Typography>

      <Paper sx={{ p: 4, mb: 4 }}>
        <Typography variant="h5" sx={{ fontWeight: 600, mb: 2 }}>
          Numerical Analysis Laboratory
        </Typography>
        <Typography variant="body1" sx={{ mb: 2, lineHeight: 1.8 }}>
          Đây là nền tảng học tập và thực hành môn <strong>Giải tích số</strong> (Numerical Analysis) 
          dành cho sinh viên bậc đại học. Hệ thống cung cấp các công cụ trực quan để:
        </Typography>
        <Typography variant="body1" sx={{ mb: 1 }}>✅ Nhập dữ liệu toán học và thực hiện các thuật toán giải tích số</Typography>
        <Typography variant="body1" sx={{ mb: 1 }}>✅ Quan sát toàn bộ quá trình tính toán từng bước</Typography>
        <Typography variant="body1" sx={{ mb: 1 }}>✅ Xem sai số và đồ thị trực quan bằng Plotly</Typography>
        <Typography variant="body1" sx={{ mb: 1 }}>✅ So sánh hiệu năng các phương pháp khác nhau</Typography>
        <Typography variant="body1">✅ Tất cả công thức được hiển thị bằng MathJax</Typography>
      </Paper>

      <Typography variant="h5" sx={{ fontWeight: 600, mb: 2 }}>
        📚 Các mô-đun
      </Typography>

      <Grid container spacing={2} sx={{ mb: 4 }}>
        {[
          { title: 'Tìm nghiệm PT', desc: 'Bisection, Newton-Raphson, Secant, Fixed Point Iteration' },
          { title: 'Hệ PT phi tuyến', desc: 'Newton Multivariable, Fixed Point Multivariable' },
          { title: 'PT Ma trận AX=B', desc: 'Gaussian, Gauss-Jordan, LU, Cholesky' },
          { title: 'Hệ PT tuyến tính', desc: 'Jacobi, Gauss-Seidel, SOR' },
          { title: 'Nội suy', desc: 'Lagrange, Newton Forward/Backward, Divided Differences' },
          { title: 'Tích phân số', desc: 'Trapezoidal, Simpson 1/3, Simpson 3/8, Romberg' },
        ].map((mod) => (
          <Grid key={mod.title} size={{ xs: 12, sm: 6, md: 4 }}>
            <Card variant="outlined" sx={{ height: '100%' }}>
              <CardContent>
                <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>{mod.title}</Typography>
                <Typography variant="body2" color="text.secondary">{mod.desc}</Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Paper sx={{ p: 3, bgcolor: 'grey.50' }}>
        <Typography variant="h6" sx={{ mb: 1 }}>🛠️ Công nghệ</Typography>
        <Typography variant="body2">
          Frontend: React + TypeScript + Vite + Material UI + Plotly + MathJax
        </Typography>
        <Typography variant="body2">
          Backend: FastAPI + NumPy + SciPy
        </Typography>
      </Paper>
    </Container>
  );
}