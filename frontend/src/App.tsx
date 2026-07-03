import { useState, useMemo } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme, CssBaseline, Box, IconButton } from '@mui/material';
import DarkModeIcon from '@mui/icons-material/DarkMode';
import LightModeIcon from '@mui/icons-material/LightMode';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import RootFinding from './pages/RootFinding';
import NonlinearSystem from './pages/NonlinearSystem';
import LinearSystem from './pages/LinearSystem';
import Interpolation from './pages/Interpolation';
import Integration from './pages/Integration';
import Comparison from './pages/Comparison';
import About from './pages/About';

const DRAWER_WIDTH = 280;

export default function App() {
  const [darkMode, setDarkMode] = useState(false);

  const theme = useMemo(
    () =>
      createTheme({
        palette: {
          mode: darkMode ? 'dark' : 'light',
          primary: {
            main: darkMode ? '#90caf9' : '#1565c0',
          },
          secondary: {
            main: '#7c4dff',
          },
          background: {
            default: darkMode ? '#121212' : '#f5f5f5',
            paper: darkMode ? '#1e1e1e' : '#ffffff',
          },
        },
        typography: {
          fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
        },
        components: {
          MuiCard: {
            defaultProps: {
              variant: 'outlined',
            },
          },
        },
      }),
    [darkMode]
  );

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <BrowserRouter>
        <Box sx={{ display: 'flex', minHeight: '100vh' }}>
          <Sidebar darkMode={darkMode} onToggleDarkMode={() => setDarkMode(!darkMode)} />

          <Box component="main" sx={{
            flexGrow: 1,
            p: 3,
            ml: { md: `${DRAWER_WIDTH}px` },
            minHeight: '100vh',
          }}>
            {/* Top bar */}
            <Box sx={{
              display: 'flex',
              justifyContent: 'flex-end',
              alignItems: 'center',
              mb: 1,
            }}>
              <IconButton onClick={() => setDarkMode(!darkMode)} color="inherit">
                {darkMode ? <LightModeIcon /> : <DarkModeIcon />}
              </IconButton>
            </Box>

            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/root-finding" element={<RootFinding />} />
              <Route path="/nonlinear-system" element={<NonlinearSystem />} />
              <Route path="/linear-system" element={<LinearSystem />} />
              <Route path="/interpolation" element={<Interpolation />} />
              <Route path="/integration" element={<Integration />} />
              <Route path="/comparison" element={<Comparison />} />
              <Route path="/about" element={<About />} />
            </Routes>
          </Box>
        </Box>
      </BrowserRouter>
    </ThemeProvider>
  );
}