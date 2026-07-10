import { type ReactNode } from 'react';
import DashboardIcon from '@mui/icons-material/Dashboard';
import SearchIcon from '@mui/icons-material/Search';
import AccountTreeIcon from '@mui/icons-material/AccountTree';
import LinearScaleIcon from '@mui/icons-material/LinearScale';
import InfoIcon from '@mui/icons-material/Info';
import InvertColorsIcon from '@mui/icons-material/InvertColors';
import GridOnIcon from '@mui/icons-material/GridOn';
import VisibilityIcon from '@mui/icons-material/Visibility';

export interface ModuleEntry {
  text: string;
  title: string;
  description: string;
  path: string;
  icon: ReactNode;
  color?: string;
  showInDashboard: boolean;
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
    text: 'Ma trận nghịch đảo',
    title: 'Ma trận nghịch đảo A⁻¹',
    description: 'Gauss-Jordan, Adjoint, LU, Cholesky, Pseudoinverse SVD',
    path: '/matrix-inverse',
    icon: <InvertColorsIcon sx={{ fontSize: 48 }} />,
    color: '#c62828',
    showInDashboard: true,
    showInSidebar: true,
  },
  {
    text: 'Phân tách ma trận',
    title: 'Phân tách ma trận',
    description: 'LU, Cholesky, QR, SVD, Schur Decomposition',
    path: '/matrix-decomposition',
    icon: <GridOnIcon sx={{ fontSize: 48 }} />,
    color: '#e65100',
    showInDashboard: true,
    showInSidebar: true,
  },
  {
    text: 'Trị riêng & Vector riêng',
    title: 'Trị riêng & Vector riêng',
    description: 'Characteristic Polynomial, Power, Inverse Power, Rayleigh, QR, Jacobi',
    path: '/eigenvalues',
    icon: <VisibilityIcon sx={{ fontSize: 48 }} />,
    color: '#00695c',
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