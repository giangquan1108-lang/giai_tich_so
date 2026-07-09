import axios from 'axios';

const api = axios.create({
  headers: {
    'Content-Type': 'application/json',
  },
});

// Root Finding
export interface RootFindingRequest {
  function: string;
  a?: number;
  b?: number;
  x0?: number;
  x1?: number;
  epsilon: number;
  max_iterations: number;
  method: string;
}

export interface RootFindingResponse {
  success: boolean;
  message: string;
  root?: number;
  f_root?: number;
  iterations_count: number;
  final_error: number;
  iterations: Record<string, unknown>[];
  formula: string;
  convergence_data?: {
    iterations: number[];
    errors: number[];
    x_values: number[];
  };
  contraction_info?: string;
}

// Nonlinear System
export interface NonlinearSystemRequest {
  functions: string[];
  variables: string[];
  initial_guess: number[];
  epsilon: number;
  max_iterations: number;
  method: string;
}

export interface NonlinearSystemResponse {
  success: boolean;
  message: string;
  solution?: number[];
  jacobian?: number[][];
  iterations_count: number;
  final_error: number;
  iterations: Record<string, unknown>[];
  formula: string;
  contraction_warning?: string;
}

// Unified Linear System (AX=B) — direct + iterative methods
export interface LinearSystemRequest {
  A: number[][];
  B: number[][];  // m×p matrix (general case). For column vector: [[b1], [b2], ...]
  method: string;  // gaussian, gauss_jordan, lu, cholesky, thomas, jacobi, gauss_seidel, sor
  initial_guess?: number[];
  epsilon?: number;
  max_iterations?: number;
  omega?: number;
}

export interface MatrixProperties {
  is_square: boolean;
  is_symmetric: boolean;
  is_positive_definite: boolean;
  is_tridiagonal: boolean;
  is_diagonally_dominant_strict: boolean;
  is_diagonally_dominant_weak: boolean;
  recommendations: string[];
}

export interface LinearSystemResponse {
  success: boolean;
  message: string;
  formula: string;
  execution_time?: number;
  solution?: number[][];  // n×p solution matrix
  // Solution classification
  solution_type?: 'unique' | 'inconsistent' | 'infinite';
  rank_A?: number;
  rank_augmented?: number;
  analysis?: Record<string, unknown>;
  // Infinite solutions
  free_variables?: string[];
  particular_solution?: number[][];  // n×p for infinite solutions with multi-column B
  basis_vectors?: number[][];
  general_solution_latex?: string;
  // Direct method steps
  steps?: Record<string, unknown>[];
  pivot_info?: Record<string, unknown>[];
  // Iterative method fields
  iterations: Record<string, unknown>[];
  iterations_count: number;
  final_error: number;
  convergence_data?: Record<string, unknown>;
  // Extra metadata
  diagonally_dominant?: boolean;
  effective_epsilon?: number;
  machine_epsilon?: number;
  omega?: number;
  // Matrix property analysis
  matrix_properties?: MatrixProperties;
}

// Backward-compatible aliases (MatrixSolver redirects to LinearSystem)
export type MatrixSolverRequest = LinearSystemRequest;
export type MatrixSolverResponse = LinearSystemResponse;

// Matrix Inverse
export interface MatrixInverseRequest {
  A: number[][];
  method?: string;  // gauss_jordan, adjoint, lu, cholesky
}

export interface MatrixInverseResponse {
  success: boolean;
  message: string;
  determinant?: number;
  rank?: number;
  singular_values?: number[];
  condition_number?: number;
  inverse?: number[][];
  inverse_latex?: string;
  verification?: number[][];
  is_accurate?: boolean;
  steps?: Record<string, unknown>[];
  execution_time?: number;
}

// Interpolation
export interface InterpolationRequest {
  x_points: number[];
  y_points: number[];
  x_value?: number;
  method: string;
}

export interface InterpolationResponse {
  success: boolean;
  message: string;
  interpolated_value?: number;
  polynomial?: string;
  divided_diff_table?: number[][];
  iterations: Record<string, unknown>[];
  formula: string;
}

// Integration
export interface IntegrationRequest {
  function: string;
  a: number;
  b: number;
  n: number;
  method: string;
}

export interface IntegrationResponse {
  success: boolean;
  message: string;
  result?: number;
  exact_value?: number;
  error?: number;
  relative_error?: number;
  iterations: Record<string, unknown>[];
  formula: string;
  plot_data?: {
    x: number[];
    y: number[];
    a: number;
    b: number;
  };
}

// Comparison
export interface ComparisonRequest {
  function: string;
  a?: number;
  b?: number;
  x0?: number;
  x1?: number;
  epsilon: number;
  max_iterations: number;
  methods: string[];
  task_type: string;
}

export interface ComparisonResponse {
  success: boolean;
  message: string;
  results: Record<string, unknown>[];
  summary?: Record<string, unknown>;
}

// API calls
export const rootFindingAPI = {
  solve: (data: RootFindingRequest) =>
    api.post<RootFindingResponse>('/root-finding/', data),
};

export const nonlinearSystemAPI = {
  solve: (data: NonlinearSystemRequest) =>
    api.post<NonlinearSystemResponse>('/nonlinear-system/', data),
};

export const linearSystemAPI = {
  solve: (data: LinearSystemRequest) =>
    api.post<LinearSystemResponse>('/linear-system/solve', data),
  properties: (A: number[][]) =>
    api.post<MatrixProperties>('/linear-system/properties', { A }),
  inverse: (A: number[][], method: string = 'gauss_jordan') =>
    api.post<MatrixInverseResponse>('/linear-system/inverse', { A, method }),
};

// Backward-compatible alias
export const matrixSolverAPI = {
  solve: (data: LinearSystemRequest) =>
    api.post<LinearSystemResponse>('/linear-system/solve', data),
};

export const interpolationAPI = {
  solve: (data: InterpolationRequest) =>
    api.post<InterpolationResponse>('/interpolation/', data),
};

export const integrationAPI = {
  solve: (data: IntegrationRequest) =>
    api.post<IntegrationResponse>('/integration/', data),
};

export const comparisonAPI = {
  compare: (data: ComparisonRequest) =>
    api.post<ComparisonResponse>('/comparison/', data),
};

export default api;