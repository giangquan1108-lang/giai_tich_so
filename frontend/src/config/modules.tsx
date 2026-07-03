import { type ReactNode } from 'react';
import DashboardIcon from '@mui/icons-material/Dashboard';
import SearchIcon from '@mui/icons-material/Search';
import AccountTreeIcon from '@mui/icons-material/AccountTree';
import LinearScaleIcon from '@mui/icons-material/LinearScale';
import TimelineIcon from '@mui/icons-material/Timeline';
import FunctionsIcon from '@mui/icons-material/Functions';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';
import InfoIcon from '@mui/icons-material/Info';

export interface ModuleEntry {
  /** Menu item text (Sidebar) */
  text: string;
  /** Card title (Dashboard) */
  title: string;
  /** Card description (Dashboard) */
  description: string;
  /** Route path */
  path: string;
  /** Icon element (shared by Sidebar & Dashboard) */
  icon: ReactNode;
  /** Card accent color (Dashboard) — falls back to primary */
  color?: string;
  /** If false, exclude from Dashboard cards (e.g. Dashboard itself) */
  showInDashboard: boolean;
  /** If false, exclude from Sidebar */
  showInSidebar: boolean;
}

export const modules: ModuleEntry[] = [
  {
    text: 'Dashboard',
    title: 'Dashboard',
    description: 'Tổng quan nền tảng Numerical Analysis',
    path: '/',
    icon: <DashboardIcon />,
    color: '#1565c0',
    showInDashboard: false,
    showInSidebar: true,
  },
  {
    text: 'Tìm nghiệm PT',
    title: 'Tìm nghiệm phương trình',
    description: 'Bisection, Newton-Raphson, Secant, Fixed Point Iteration',
    path: '/root-finding',
    icon: <SearchIcon sx={{ fontSize: 48 }} />,
    color: '#1976d2',
    showInDashboard: true,
    showInSidebar: true,
  },
  {
    text: 'Hệ PT phi tuyến',
    title: 'Hệ PT phi tuyến',
    description: 'Newton Multivariable, Fixed Point Multivariable',
    path: '/nonlinear-system',
    icon: <AccountTreeIcon sx={{ fontSize: 48 }} />,
    color: '#388e3c',
    showInDashboard: true,
    showInSidebar: true,
  },
  {
    text: 'Hệ PT đại số tuyến tính',
    title: 'Hệ PT đại số tuyến tính AX = B',
    description: 'Gaussian, Gauss-Jordan, LU, Cholesky, Thomas, Jacobi, Gauss-Seidel, SOR',
    path: '/linear-system',
    icon: <LinearScaleIcon sx={{ fontSize: 48 }} />,
    color: '#7b1fa2',
    showInDashboard: true,
    showInSidebar: true,
  },
  {
    text: 'Nội suy',
    title: 'Nội suy',
    description: 'Lagrange, Newton Forward/Backward, Divided Differences',
    path: '/interpolation',
    icon: <TimelineIcon sx={{ fontSize: 48 }} />,
    color: '#e65100',
    showInDashboard: true,
    showInSidebar: true,
  },
  {
    text: 'Tích phân số',
    title: 'Tích phân số',
    description: 'Trapezoidal, Simpson 1/3, Simpson 3/8, Romberg',
    path: '/integration',
    icon: <FunctionsIcon sx={{ fontSize: 48 }} />,
    color: '#00695c',
    showInDashboard: true,
    showInSidebar: true,
  },
  {
    text: 'So sánh thuật toán',
    title: 'So sánh thuật toán',
    description: 'So sánh hiệu năng các phương pháp khác nhau',
    path: '/comparison',
    icon: <CompareArrowsIcon sx={{ fontSize: 48 }} />,
    color: '#283593',
    showInDashboard: true,
    showInSidebar: true,
  },
  {
    text: 'Giới thiệu',
    title: 'Giới thiệu',
    description: 'Thông tin về nền tảng',
    path: '/about',
    icon: <InfoIcon />,
    color: '#455a64',
    showInDashboard: false,
    showInSidebar: true,
  },
];