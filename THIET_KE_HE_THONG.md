# THIẾT KẾ HỆ THỐNG — Numerical Analysis Laboratory

> **Phiên bản:** 1.0.0  
> **Ngày:** 09/07/2026  
> **Mô tả:** Tài liệu thiết kế kiến trúc và kỹ thuật cho nền tảng học tập Giải tích số.

---

## Mục lục

1. [Tổng quan hệ thống](#1-tổng-quan-hệ-thống)
2. [Kiến trúc tổng thể](#2-kiến-trúc-tổng-thể)
3. [Thiết kế Backend](#3-thiết-kế-backend)
   - [3.1 Cấu trúc thư mục](#31-cấu-trúc-thư-mục-backend)
   - [3.2 Entry Point — main.py](#32-entry-point--mainpy)
   - [3.3 Tầng API (Routers)](#33-tầng-api-routers)
   - [3.4 Tầng Thuật toán (Algorithms)](#34-tầng-thuật-toán-algorithms)
   - [3.5 Tầng Schema (Pydantic Models)](#35-tầng-schema-pydantic-models)
   - [3.6 Tầng Tiện ích (Utils)](#36-tầng-tiện-ích-utils)
   - [3.7 Luồng xử lý Request-Response](#37-luồng-xử-lý-request-response)
   - [3.8 Danh sách API Endpoints](#38-danh-sách-api-endpoints)
4. [Thiết kế Frontend](#4-thiết-kế-frontend)
   - [4.1 Cấu trúc thư mục](#41-cấu-trúc-thư-mục-frontend)
   - [4.2 Công nghệ](#42-công-nghệ-frontend)
   - [4.3 Cây Component](#43-cây-component)
   - [4.4 Hệ thống Routing](#44-hệ-thống-routing)
   - [4.5 Hệ thống Config Module](#45-hệ-thống-config-module)
   - [4.6 Tầng Service (API Client)](#46-tầng-service-api-client)
   - [4.7 Utilities (LaTeX helpers)](#47-utilities-latex-helpers)
   - [4.8 Các page chính](#48-các-page-chính)
5. [Luồng dữ liệu giữa Frontend và Backend](#5-luồng-dữ-liệu-giữa-frontend-và-backend)
6. [Triển khai & Vận hành](#6-triển-khai--vận-hành)
7. [Các điểm kỹ thuật nổi bật](#7-các-điểm-kỹ-thuật-nổi-bật)

---

## 1. Tổng quan hệ thống

**Numerical Analysis Laboratory** là một nền tảng web hỗ trợ học tập và thực hành môn Giải tích số (Numerical Analysis). Hệ thống cho phép người dùng:

- Nhập dữ liệu toán học qua giao diện (công thức LaTeX, ma trận, điểm dữ liệu).
- Thực hiện các thuật toán giải tích số với tham số tùy chỉnh.
- Xem toàn bộ quá trình tính toán từng bước (dạng bảng, công thức LaTeX, ma trận).
- So sánh hiệu năng giữa các phương pháp khác nhau.
- Xem kết quả trực quan bao gồm sai số, đồ thị hội tụ.

### Các mô-đun chức năng

| Mô-đun | Các phương pháp |
|--------|-----------------|
| **Tìm nghiệm phương trình** `f(x)=0` | Bisection, Newton-Raphson, Secant, Fixed Point Iteration |
| **Hệ phương trình phi tuyến** `F(X)=0` | Newton Multivariable, Fixed Point Multivariable |
| **Hệ phương trình đại số tuyến tính** `AX=B` | Gaussian, Gauss-Jordan, LU, Cholesky, Thomas, Jacobi, Gauss-Seidel, SOR |
| **Nội suy** | Lagrange, Newton Forward, Newton Backward, Divided Differences |
| **Tích phân số** | Trapezoidal, Simpson 1/3, Simpson 3/8, Romberg |
| **So sánh thuật toán** | So sánh hiệu năng các phương pháp tìm nghiệm |

---

## 2. Kiến trúc tổng thể

```
┌──────────────────────────────────────────────────────────┐
│                    Frontend (React 19 + TS)               │
│  Port 5173 (Vite Dev Server)                              │
│                                                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │ Dashboard│  │RootFind..│  │LinSystem │  │Integratio│ │
│  │          │  │NonlinSys │  │Interpol..│  │Comparison│ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘ │
│       │             │             │             │         │
│  ┌────┴─────────────┴─────────────┴─────────────┴────┐   │
│  │          Services (api.ts - Axios)                 │   │
│  └──────────────────────┬────────────────────────────┘   │
└─────────────────────────┼────────────────────────────────┘
                          │ HTTP/JSON
               ┌──────────┴──────────┐
               │  Vite Proxy Config  │
               └──────────┬──────────┘
                          │
┌─────────────────────────┼────────────────────────────────┐
│               Backend (FastAPI + Python)                   │
│  Port 8001 (Uvicorn)                                       │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │                    FastAPI App                        │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐             │ │
│  │  │Root Find.│ │Nonlin.Sys│ │Linear Sys│             │ │
│  │  │ Router   │ │ Router   │ │ Router   │ ...         │ │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘             │ │
│  └───────┼────────────┼────────────┼───────────────────┘ │
│          │            │            │                      │
│  ┌───────┴────────────┴────────────┴───────────────────┐ │
│  │                   Algorithms Layer                   │ │
│  │  ┌────────────┐ ┌────────────┐ ┌──────────────────┐ │ │
│  │  │ Root Find. │ │ Linear Sys │ │ Integration ...  │ │ │
│  │  │ Algorithms │ │ Algorithms │ │                  │ │ │
│  │  └────────────┘ └────────────┘ └──────────────────┘ │ │
│  └──────────────────────┬──────────────────────────────┘ │
│                         │                                 │
│  ┌──────────────────────┴──────────────────────────────┐ │
│  │                   Utils Layer                        │ │
│  │  ┌──────────────────────────────┐                    │ │
│  │  │  math_parser.py (SymPy)      │                    │ │
│  │  └──────────────────────────────┘                    │ │
│  └─────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

---

## 3. Thiết kế Backend

### 3.1 Cấu trúc thư mục (Backend)

```
backend/
├── requirements.txt            # Dependencies: fastapi, uvicorn, numpy, scipy, pydantic, sympy
├── app/
│   ├── main.py                 # Khởi tạo FastAPI, CORS, Router mounts
│   ├── algorithms/             # Tầng thuật toán (Business Logic)
│   │   ├── __init__.py
│   │   ├── root_finding.py     # Bisection, Newton-Raphson, Secant, Fixed Point Iteration
│   │   ├── nonlinear_system.py # Newton Multivariable, Fixed Point Multivariable
│   │   ├── linear_system.py    # Gauss, Gauss-Jordan, LU, Cholesky, Thomas, Jacobi, GS, SOR, Inverse
│   │   ├── interpolation.py    # Lagrange, Newton Forward/Backward, Divided Differences
│   │   └── integration.py      # Trapezoidal, Simpson 1/3, 3/8, Romberg
│   ├── routers/                # Tầng API Routes (Presentation)
│   │   ├── __init__.py
│   │   ├── root_finding.py     # POST /root-finding/
│   │   ├── nonlinear_system.py # POST /nonlinear-system/
│   │   ├── linear_system.py    # POST /linear-system/solve, /properties, /inverse
│   │   ├── interpolation.py    # POST /interpolation/
│   │   ├── integration.py      # POST /integration/
│   │   └── comparison.py       # POST /comparison/
│   ├── schemas/                # Tầng Schema (Pydantic Request/Response Models)
│   │   ├── __init__.py
│   │   ├── common.py
│   │   ├── root_finding.py
│   │   ├── nonlinear_system.py
│   │   ├── linear_system.py
│   │   ├── interpolation.py
│   │   ├── integration.py
│   │   └── comparison.py
│   └── utils/                  # Tầng tiện ích
│       ├── __init__.py
│       └── math_parser.py      # SymPy parser, LaTeX→Python, Preprocessing expressions
├── audit_gauss_jordan.py       # Test/Audit scripts
├── test_all.py
├── test_end_to_end.py
├── test_inverse.py
├── test_parser_fix.py
├── test_thomas.py
└── backend_test.txt
```

### 3.2 Entry Point — main.py

```python
app = FastAPI(title="Numerical Analysis Platform", version="1.0.0")

# CORS: Mở toàn bộ (allow_origins=["*"])
# Router mounts:
#   /root-finding      → root_finding.router
#   /nonlinear-system  → nonlinear_system.router
#   /linear-system     → linear_system.router
#   /interpolation     → interpolation.router
#   /integration       → integration.router
#   /comparison        → comparison.router
# Health: GET /, GET /health
```

### 3.3 Tầng API (Routers)

Mỗi Router chịu trách nhiệm:
1. Nhận HTTP Request (POST).
2. Parse và validate dữ liệu đầu vào qua Pydantic schema.
3. Parse biểu thức toán học: LaTeX → Python (qua `latex_to_python`) hoặc raw Python string.
4. Dispatch đến algorithm tương ứng dựa trên tham số `method`.
5. Trả về JSON response theo schema.

**Ví dụ Router root_finding.py:**

```python
router = APIRouter()

@router.post("/", response_model=RootFindingResponse)
async def solve_root_finding(request: RootFindingRequest):
    # 1. Parse function string (LaTeX hoặc Python)
    # 2. Gọi root_finding.bisection / newton_raphson / secant / fixed_point_iteration
    # 3. Return RootFindingResponse
```

**Router so sánh (comparison.py):**
- Hiện chỉ hỗ trợ `task_type="root_finding"`.
- Gọi lần lượt từng phương pháp trong `request.methods`.
- Trả về summary: `total_methods`, `successful_methods`, `fastest`, `fewest_iterations`.

### 3.4 Tầng Thuật toán (Algorithms)

Đây là tầng cốt lõi, chứa toàn bộ logic tính toán giải tích số.

#### 3.4.1 Root Finding (`root_finding.py`) — 429 dòng

| Hàm | Phương pháp | Đặc điểm |
|-----|------------|----------|
| `bisection(f, a, b, ε, max_iter)` | Chia đôi | Kiểm tra f(a)·f(b) < 0. Theo dõi `a, b, x_k, f_x_k, error` mỗi vòng lặp |
| `newton_raphson(f, x0, ε, max_iter, f_prime?)` | Newton-Raphson | Nếu không cung cấp `f'`, dùng central difference. Kiểm tra NaN/Inf, đạo hàm ≈ 0, step quá lớn |
| `secant(f, x0, x1, ε, max_iter)` | Dây cung | Kiểm tra chia cho 0 khi `f(x1) ≈ f(x0)` |
| `fixed_point_iteration(f, x0, ε, max_iter)` | Lặp đơn | Kiểm tra điều kiện co `\|g'(x₀)\|`. Kiểm tra phân kỳ (giá trị > 10^10, NaN, complex, overflow) |

**Helpers chung:**
- `_effective_epsilon(ε)`: Clamp ε tối thiểu 1e-15.
- `_is_converged(fx, error, ε)`: Kiểm tra hội tụ với machine epsilon.
- `_get_convergence_data(iterations)`: Trích xuất dữ liệu hội tụ cho đồ thị (iterations, errors, x_values).

**Response chung:** Tất cả hàm trả về dict với cấu trúc:
```python
{
    "success": bool,
    "message": str,
    "root": float | None,
    "f_root": float | None,
    "iterations_count": int,
    "final_error": float,
    "iterations": list[dict],       # Log từng bước
    "convergence_data": dict,       # Dữ liệu cho đồ thị hội tụ
    "execution_time": float,
    "effective_epsilon": float,
    "machine_epsilon": float,
    "contraction_info": str | None, # Chỉ cho Fixed Point
}
```

#### 3.4.2 Nonlinear System (`nonlinear_system.py`) — 252 dòng

| Hàm | Đặc điểm |
|-----|----------|
| `newton_multivariable(functions, variables, x0, ε, max_iter)` | Tính Jacobian bằng sai phân hữu hạn (central differences). Giải `J·dx = -F` bằng `np.linalg.solve`. Kiểm tra phân kỳ, NaN, Jacobian suy biến. |
| `fixed_point_multivariable(functions, variables, x0, ε, max_iter)` | Kiểm tra điều kiện co qua spectral norm của Jacobian (SVD). Cảnh báo nếu `||JG||₂ > 1`. |

**Response:** Dict với `success`, `solution`, `iterations`, `contraction_warning`, v.v.

#### 3.4.3 Linear System (`linear_system.py`) — 1559 dòng (file lớn nhất)

**Phương pháp trực tiếp (Direct Methods):**

| Hàm | Đặc điểm |
|-----|----------|
| `gaussian_elimination(A, B)` | Partial pivoting. Ghi chi tiết từng bước: chọn pivot, row swap, elimination, upper triangular, back substitution. Hỗ trợ B m×p. |
| `gauss_jordan(A, B)` | Single pass RREF với full elimination trên tất cả hàng. Pivot được normalize về 1. Ghi tất cả steps. Hỗ trợ B m×p. Single source of truth: `_compute_rref_full()`. |
| `lu_decomposition(A, B)` | Doolittle LU với partial pivoting: A = PLU. Giải L·Y = P·B, U·X = Y. Yêu cầu ma trận vuông. |
| `cholesky_decomposition(A, B)` | Cholesky: A = L·L^T. Kiểm tra đối xứng và xác định dương. Yêu cầu ma trận vuông. |
| `thomas_algorithm(A, B)` | TDMA cho ma trận ba đường chéo. O(n) mỗi cột. Kiểm tra cấu trúc tridiagonal, zero diagonal. |

**Phương pháp lặp (Iterative Methods)** — chỉ hỗ trợ B(m×1):

| Hàm | Đặc điểm |
|-----|----------|
| `jacobi(A, B, x0?, ε, max_iter)` | Cập nhật đồng thời: `x_i^{k+1} = (b_i - Σ_{j≠i} a_ij·x_j^k) / a_ii`. Kiểm tra chéo trội. |
| `gauss_seidel(A, B, x0?, ε, max_iter)` | Cập nhật tuần tự: dùng giá trị mới nhất. |
| `sor(A, B, x0?, ε, max_iter, ω)` | Successive Over-Relaxation: `x_new = (1-ω)·x_old + ω·x_GS`. |

**Ma trận nghịch đảo (Matrix Inverse)** — 7 phương pháp:

| Hàm | Đặc điểm |
|-----|----------|
| `matrix_inverse_gauss_jordan(A)` | [A\|I] → [I\|A⁻¹]. Ghi steps chi tiết. |
| `matrix_inverse_adjoint(A)` | A⁻¹ = Adj(A)/det(A). Tính cofactor → adjoint. |
| `matrix_inverse_cholesky(A)` | Cholesky: A = L·L^T, tính A⁻¹ = (L⁻¹)^T·L⁻¹. Yêu cầu ma trận SPD. |
| `matrix_inverse_bordering(A)` | Viền quanh (Frobenius-Schur). Xây dựng dần A⁻¹ từ ma trận con 1×1. |
| `matrix_inverse_jacobi(A, ε, max_iter)` | Lặp Jacobi: X⁽ᵏ⁺¹⁾ = -D⁻¹(L+U)·X⁽ᵏ⁾ + D⁻¹. Yêu cầu chéo trội. |
| `matrix_inverse_gauss_seidel(A, ε, max_iter)` | Lặp Gauss-Seidel: cập nhật tuần tự từng cột của X⁻¹. |
| `matrix_inverse_newton(A, ε, max_iter)` | Newton-Schulz: X⁽ᵏ⁺¹⁾ = X⁽ᵏ⁾·(2I - A·X⁽ᵏ⁾). Hội tụ bậc 2. |

**Phân tích hệ thống `_analyze_system()`:**
- Tính rank(A) và rank([A\|B]) qua SVD.
- Phân loại: `unique` (nghiệm duy nhất), `inconsistent` (vô nghiệm), `infinite` (vô số nghiệm).
- Với infinite: trích xuất pivot columns, free variables, particular solution, basis vectors.

**Phát hiện thuộc tính ma trận `_detect_matrix_properties()`:**
- Kiểm tra: vuông, đối xứng, xác định dương, ba đường chéo, chéo trội (strict/weak).
- Đưa ra khuyến nghị phương pháp phù hợp.

**Hiển thị LaTeX:**
- `_matrix_to_latex()`: Chuyển ma trận sang định dạng LaTeX `bmatrix` / `array` (có augmented với vertical bar).
- Tất cả steps đều có `matrix_latex` để frontend hiển thị công thức đẹp.

#### 3.4.4 Interpolation (`interpolation.py`) — 256 dòng

| Hàm | Đặc điểm |
|-----|----------|
| `lagrange_interpolation(x, y, x_val?)` | Xây dựng đa thức Lagrange. Trả về polynomial (dạng string), interpolated_value. |
| `newton_forward_interpolation(x, y, x_val?)` | Bảng sai phân chia. Đa thức dạng Newton tiến. |
| `newton_backward_interpolation(x, y, x_val?)` | Newton lùi — dùng backward differences. |
| `divided_differences(x, y, x_val?)` | Bảng tỉ sai phân đầy đủ. |

**Response chứa:** `polynomial` (string), `divided_diff_table`, `interpolated_value`, `iterations`.

#### 3.4.5 Integration (`integration.py`) — 289 dòng

| Hàm | Đặc điểm |
|-----|----------|
| `trapezoidal(f, a, b, n)` | h/2·[f(a) + 2Σf(x_i) + f(b)] |
| `simpson_one_third(f, a, b, n)` | h/3·[f₀ + 4Σ(odd) + 2Σ(even) + f_n]. Tự động tăng n nếu lẻ. |
| `simpson_three_eighth(f, a, b, n)` | 3h/8·[...]. Tự động tăng n nếu không chia hết cho 3. |
| `romberg(f, a, b, n)` | Richardson extrapolation: R(i,j) = R(i,j-1) + (R(i,j-1)−R(i-1,j-1))/(4^j−1). Trả về bảng Romberg. |

**Tính giá trị đúng:** Dùng `scipy.integrate.quad()` làm reference.  
**Response chứa:** `result`, `exact_value`, `error`, `plot_data` (x, y, a, b).

### 3.5 Tầng Schema (Pydantic Models)

Mỗi mô-đun có 2 schema chính: `XxxRequest` và `XxxResponse`. Một số module có thêm schema phụ.

| Schema | Thuộc tính chính |
|--------|-----------------|
| `RootFindingRequest` | function, a?, b?, x0?, x1?, epsilon, max_iterations, method |
| `RootFindingResponse` | success, message, root?, f_root?, iterations_count, final_error, iterations, formula, convergence_data, contraction_info |
| `NonlinearSystemRequest` | functions (list), variables (list), initial_guess, epsilon, max_iterations, method |
| `NonlinearSystemResponse` | success, message, solution?, jacobian?, iterations, contraction_warning |
| `LinearSystemRequest` | A (m×n), B (m×p), method, initial_guess?, epsilon, max_iterations, omega? |
| `LinearSystemResponse` | success, message, formula, solution?, solution_type?, rank_A, rank_augmented, free_variables, particular_solution, basis_vectors, general_solution_latex, steps, iterations, convergence_data, matrix_properties, diagonally_dominant |
| `MatrixPropertiesRequest` | A |
| `MatrixPropertiesResponse` | is_square, is_symmetric, is_positive_definite, is_tridiagonal, is_diagonally_dominant_strict, is_diagonally_dominant_weak, recommendations |
| `MatrixInverseRequest` | A, method |
| `MatrixInverseResponse` | success, message, determinant, rank, inverse, inverse_latex, verification, is_accurate, steps |
| `IntegrationRequest` | function, a, b, n, method |
| `IntegrationResponse` | success, message, result?, exact_value?, error?, relative_error?, iterations, formula, plot_data |
| `InterpolationRequest` | x_points, y_points, x_value?, method |
| `InterpolationResponse` | success, message, interpolated_value?, polynomial?, divided_diff_table?, iterations, formula |
| `ComparisonRequest` | function?, a?, b?, x0?, x1?, epsilon, max_iterations, methods, task_type |
| `ComparisonResponse` | success, message, results (list), summary? |

### 3.6 Tầng Tiện ích (Utils)

#### `math_parser.py` — 303 dòng

**Chức năng chính:**

1. **`_preprocess(expr: str) → str`**: Tiền xử lý biểu thức toán học.
   - Strip `$$`, `\[\]`.
   - Chuyển `^` → `**`.
   - Xử lý `e**x` → `exp(x)`.
   - Chuyển fractional power `x**(1/3)` → `_real_cbrt(x)` (tránh complex numbers).
   - Mask/unmask tên hàm số (sin, cos, ...) để implicit multiplication không phá vỡ.
   - Xử lý implicit multiplication: `2x` → `2*x`, `x(x+1)` → `x*(x+1)`, `)(` → `)*(`, v.v.

2. **`parse_function(expr: str) → Callable[[float], float]`**: Parse biểu thức 1 biến `x` bằng SymPy, lambdify thành Python callable.

3. **`parse_multivariable_function(expr, variables) → Callable[..., float]`**: Parse hàm nhiều biến.

4. **`compute_numerical_jacobian(functions, variables, point, h=1e-8)`**: Tính Jacobian bằng central finite differences. Dùng cho Newton Multivariable và Fixed Point Multivariable.

5. **`latex_to_python(latex: str) → str`**: Chuyển LaTeX sang Python expression.
   - `\sin` → `sin`, `\cos` → `cos`, `\pi` → `pi`
   - `^{...}` → `**(...)`
   - `\frac{a}{b}` → `(a)/(b)`
   - `\cdot` / `\times` → `*`
   - `\left|...\right|` → `abs(...)`

6. **Custom real-root functions:** `_real_cbrt(x)`, `_real_root(x, n)` tránh trả về số phức cho căn bậc lẻ của số âm.

### 3.7 Luồng xử lý Request-Response

```
Client (Frontend)
    │
    │  POST /root-finding/ {"function": "x^3 - 6*x^2 + 11*x - 6", "method": "bisection", ...}
    ▼
Router (root_finding.py)
    │  1. Validate request qua Pydantic schema
    │  2. Parse function: latex_to_python() rồi parse_function()
    │  3. Dispatch: if method == "bisection" → root_finding.bisection(f, a, b, ε, max_iter)
    ▼
Algorithm (root_finding.py: bisection)
    │  1. Kiểm tra f(a)·f(b) < 0
    │  2. Loop: c = (a+b)/2, tính fc, error
    │  3. Log mỗi bước vào iterations list
    │  4. Kiểm tra hội tụ (_is_converged)
    │  5. Trả về dict kết quả
    ▼
Router
    │  Map dict kết quả → RootFindingResponse (Pydantic)
    ▼
Client (Frontend)
    │  Nhận JSON, render kết quả
```

### 3.8 Danh sách API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| GET | `/` | Welcome message |
| GET | `/health` | Health check |
| POST | `/root-finding/` | Tìm nghiệm phương trình |
| POST | `/nonlinear-system/` | Giải hệ phi tuyến |
| POST | `/linear-system/solve` | Giải hệ AX = B (tất cả phương pháp) |
| POST | `/linear-system/properties` | Phân tích thuộc tính ma trận A |
| POST | `/linear-system/inverse` | Tính ma trận nghịch đảo A⁻¹ |
| POST | `/interpolation/` | Nội suy |
| POST | `/integration/` | Tích phân số |
| POST | `/comparison/` | So sánh thuật toán |

---

## 4. Thiết kế Frontend

### 4.1 Cấu trúc thư mục (Frontend)

```
frontend/
├── index.html                  # Entry HTML, load MathJax CDN, Google Fonts
├── package.json                # Dependencies (React 19, MUI 9, Plotly, MathJax, ...)
├── vite.config.ts              # Vite config: proxy to backend port 8001
├── tsconfig.json
├── public/
│   ├── favicon.svg
│   └── icons.svg
└── src/
    ├── main.tsx                # ReactDOM entry
    ├── App.tsx                 # Root: Theme, Router, Sidebar layout
    ├── index.css
    ├── assets/
    │   └── hero.png, react.svg, vite.svg
    ├── config/
    │   └── modules.tsx         # Module registry (menu items, dashboard cards)
    ├── services/
    │   └── api.ts              # Axios client, request/response types, API methods
    ├── utils/
    │   └── latex.ts            # LaTeX utilities (toLatex, rawToPython, latexToPython)
    ├── components/
    │   ├── FormulaEditor.tsx   # Editor LaTeX với preview real-time
    │   ├── FormulaRenderer.tsx # Hiển thị công thức MathJax
    │   ├── IterationTable.tsx  # Bảng dữ liệu vòng lặp (có LaTeX headers)
    │   ├── MatrixInput.tsx     # Legacy: nhập ma trận + vector (đã thay)
    │   ├── MatrixLatexEditor.tsx # Editor ma trận A×B với keyboard navigation
    │   ├── ResultCard.tsx      # Card hiển thị kết quả (success/error)
    │   ├── Sidebar.tsx         # Sidebar navigation (responsive)
    │   └── SolutionSteps.tsx   # Hiển thị từng bước giải (phân phase, LaTeX)
    └── pages/
        ├── Dashboard.tsx       # Trang chủ, grid cards
        ├── RootFinding.tsx     # Tìm nghiệm PT
        ├── NonlinearSystem.tsx # Hệ PT phi tuyến
        ├── LinearSystem.tsx    # AX = B + Ma trận nghịch đảo (2 tabs)
        ├── Interpolation.tsx   # Nội suy
        ├── Integration.tsx     # Tích phân số
        ├── Comparison.tsx      # So sánh thuật toán
        └── About.tsx           # Giới thiệu
```

### 4.2 Công nghệ (Frontend)

| Công nghệ | Mục đích |
|-----------|----------|
| **React 19** + TypeScript | UI framework |
| **Vite 8** | Build tool, dev server |
| **Material UI 9** (MUI) | Component library (Drawer, Table, Card, ...) |
| **React Router 7** | Client-side routing |
| **Axios** | HTTP client |
| **MathJax 3** (CDN) | Render công thức LaTeX |
| **Plotly.js 3** | Đồ thị (đã import trong package.json) |
| **Emotion** (MUI dependency) | CSS-in-JS |

### 4.3 Cây Component

```
App
├── ThemeProvider (MUI: dark/light mode)
│   └── BrowserRouter
│       ├── Sidebar (Drawer: permanent desktop, temporary mobile)
│       │   ├── Logo "📐 Numerical Lab"
│       │   ├── NavItems (từ modules.tsx)
│       │   └── Footer version
│       └── Main Content
│           ├── IconButton (toggle dark/light)
│           └── Routes
│               ├── "/" → Dashboard
│               ├── "/root-finding" → RootFinding
│               │   ├── FormulaEditor (nhập f(x))
│               │   ├── FormulaRenderer (hiển thị LaTeX)
│               │   ├── ResultCard (kết quả)
│               │   └── IterationTable (bảng lặp)
│               ├── "/nonlinear-system" → NonlinearSystem
│               │   ├── FormulaEditor × n (nhập F_i)
│               │   └── IterationTable
│               ├── "/linear-system" → LinearSystem (2 tabs)
│               │   ├── Tab "Giải AX=B"
│               │   │   ├── MatrixLatexEditor (nhập A, B)
│               │   │   ├── ResultCard
│               │   │   ├── SolutionSteps (chi tiết steps)
│               │   │   └── IterationTable (cho iterative methods)
│               │   └── Tab "Ma trận nghịch đảo"
│               │       ├── MatrixLatexEditor (chỉ nhập A)
│               │       ├── ResultCard (A⁻¹, verification)
│               │       └── SolutionSteps
│               ├── "/interpolation" → Interpolation
│               │   ├── Dynamic point list (add/remove)
│               │   ├── FormulaRenderer
│               │   └── Divided difference table
│               ├── "/integration" → Integration
│               │   ├── FormulaEditor
│               │   └── FormulaRenderer (công thức, kết quả)
│               ├── "/comparison" → Comparison
│               │   ├── Method selector chips
│               │   └── Result cards comparison
│               └── "/about" → About
```

### 4.4 Hệ thống Routing

```typescript
<Route path="/" element={<Dashboard />} />
<Route path="/root-finding" element={<RootFinding />} />
<Route path="/nonlinear-system" element={<NonlinearSystem />} />
<Route path="/linear-system" element={<LinearSystem />} />
<Route path="/interpolation" element={<Interpolation />} />
<Route path="/integration" element={<Integration />} />
<Route path="/comparison" element={<Comparison />} />
<Route path="/about" element={<About />} />
```

### 4.5 Hệ thống Config Module (`modules.tsx`)

`modules.tsx` là registry tập trung cho cả Sidebar và Dashboard. Định nghĩa `ModuleEntry[]`:

```typescript
interface ModuleEntry {
  text: string;           // Tên hiển thị trong Sidebar
  title: string;          // Tiêu đề card Dashboard
  description: string;    // Mô tả card Dashboard
  path: string;           // Route path
  icon: ReactNode;        // Icon MUI
  color?: string;         // Màu nhấn Dashboard card
  showInDashboard: bool;  // Có hiện trên Dashboard không?
  showInSidebar: bool;    // Có hiện trong Sidebar không?
}
```

Có 8 module được định nghĩa: Dashboard, Tìm nghiệm PT, Hệ PT phi tuyến, Hệ PT đại số tuyến tính, Nội suy, Tích phân số, So sánh thuật toán, Giới thiệu.

### 4.6 Tầng Service (API Client) — `api.ts`

File `api.ts` định nghĩa:
1. **Axios instance** với base config `Content-Type: application/json`.
2. **TypeScript interfaces** cho tất cả Request/Response types (mirror backend Pydantic schemas).
3. **API object methods** tổ chức theo module:

```typescript
export const rootFindingAPI = { solve: (data) => api.post('/root-finding/', data) };
export const nonlinearSystemAPI = { solve: (data) => api.post('/nonlinear-system/', data) };
export const linearSystemAPI = {
  solve: (data) => api.post('/linear-system/solve', data),
  properties: (A) => api.post('/linear-system/properties', { A }),
  inverse: (A, method) => api.post('/linear-system/inverse', { A, method }),
};
export const interpolationAPI = { solve: (data) => api.post('/interpolation/', data) };
export const integrationAPI = { solve: (data) => api.post('/integration/', data) };
export const comparisonAPI = { compare: (data) => api.post('/comparison/', data) };
```

### 4.7 Utilities (LaTeX helpers) — `latex.ts`

| Hàm | Chức năng |
|-----|----------|
| `toLatex(expr: string)` | Chuyển raw input → LaTeX hiển thị: `^` → `^{}` , `*` → `\cdot`, `sin` → `\sin`, `exp(...)` → `e^{...}`, `abs` → `\|...\|` |
| `rawToPython(expr: string)` | Chuyển raw → Python: `^` → `**`, `2x` → `2*x`, `e**...` → `exp(...)` |
| `latexToPython(latex: string)` | Chuyển LaTeX → Python: `\sin` → `sin`, `^{...}` → `**(...)`, `\frac{a}{b}` → `(a)/(b)` |
| `isValidLatex(latex: string)` | Kiểm tra dấu `{}` cân bằng |

### 4.8 Các page chính

#### Dashboard
- Grid 2×3 cards cho 6 mô-đun chức năng (từ `modules.filter(m => m.showInDashboard)`).
- Section "Mục tiêu" ở cuối trang.

#### RootFinding
- Select method (4 options).
- FormulaEditor cho f(x) hoặc g(x) (với Fixed Point).
- Input fields động: a, b cho Bisection; x₀ cho Newton/FixedPoint; x₀, x₁ cho Secant.
- Result: root, f(root), số vòng lặp, sai số.
- IterationTable: bảng chi tiết từng bước lặp.

#### NonlinearSystem
- Select số biến (2, 3, 4) + phương pháp (Newton / Fixed Point).
- Dynamic FormulaEditor cho mỗi phương trình.
- Initial guess cho từng biến.
- Result: vector nghiệm, jacobian.

#### LinearSystem (trang phức tạp nhất, 666 dòng)
- **2 tabs**: Giải AX=B | Ma trận nghịch đảo.
- **Tab Giải AX=B:**
  - Select method (8 options: 5 direct + 3 iterative).
  - MatrixLatexEditor: nhập A(m×n) và B(m×p) với keyboard navigation (arrow keys, tab).
  - Phân tích ma trận tự động (chips: vuông, đối xứng, xác định dương, ba đường chéo, chéo trội).
  - Khuyến nghị phương pháp.
  - Result classification: unique / inconsistent / infinite.
- **Tab Ma trận nghịch đảo:**
  - 4 phương pháp: Gauss-Jordan, Adjoint, LU, Cholesky.
  - Hiển thị: det(A), rank(A), A⁻¹, verification (A·A⁻¹ ≈ I).

#### Interpolation
- Dynamic point list: add/remove điểm.
- Select method (4 options).
- Hiển thị: đa thức nội suy, bảng tỉ sai phân, giá trị tại x.

#### Integration
- Select method (4 options).
- FormulaEditor cho f(x).
- Input a, b, n.
- Hiển thị: kết quả, giá trị đúng, sai số tuyệt đối, sai số tương đối, công thức LaTeX.

#### Comparison
- TextField cho f(x) (đơn giản, không dùng FormulaEditor).
- Chip selector cho 4 phương pháp.
- So sánh: nghiệm, số vòng lặp, sai số, thời gian.
- Summary: fastest, fewest_iterations.

---

## 5. Luồng dữ liệu giữa Frontend và Backend

### Ví dụ luồng "Giải hệ AX = B bằng Gauss-Jordan"

```
User tương tác                    Frontend                             Backend
─────────────                     ────────                             ───────
1. Nhập ma trận A, B    ──→      MatrixLatexEditor state
2. Chọn method           ──→      state.method = "gauss_jordan"
3. Click "Giải hệ"       ──→      handleSolve()
                                  │
                                  ├── linearSystemAPI.solve({
                                  │     A: [[10,-1,2],[-1,11,-1],[2,-1,10]],
                                  │     B: [[6],[25],[-11]],
                                  │     method: 'gauss_jordan'
                                  │   })
                                  │
                                  │   POST /linear-system/solve ──→  Router nhận request
                                  │                                   │
                                  │                                   ├── Validate schema
                                  │                                   ├── Dispatch: gauss_jordan()
                                  │                                   │   ├── _compute_rref_full()
                                  │                                   │   │   ├── Augment [A|B]
                                  │                                   │   │   ├── Partial pivoting mỗi cột
                                  │                                   │   │   ├── Normalize pivot
                                  │                                   │   │   ├── Eliminate all other rows
                                  │                                   │   │   └── Ghi steps (matrix + LaTeX)
                                  │                                   │   ├── _extract_solution_from_rref()
                                  │                                   │   └── Return dict
                                  │                                   │
                                  │   ←── LinearSystemResponse JSON ──┤
                                  │
                                  ├── setResult(response.data)
                                  │
4. Render kết quả          ←──── result.solution, result.steps
   ├── ResultCard: "X = (0.6, 2.2727, -0.7909)"
   ├── FormulaRenderer: Công thức Gauss-Jordan
   └── SolutionSteps:
       ├── Phase: row_swap → Đổi hàng
       ├── Phase: normalize → Chuẩn hóa
       ├── Phase: eliminate → Khử
       ├── Phase: rref → RREF
       └── Phase: solution → Nghiệm
```

---

## 6. Triển khai & Vận hành

### Khởi động

**Cách 1: Manual**
```bash
# Terminal 1 — Backend
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --log-level error

# Terminal 2 — Frontend
cd frontend
npm run dev
```

**Cách 2: Script (start.bat)**
```batch
start "Backend" cmd /k "python -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --log-level error"
start "Frontend" cmd /k "npm run dev"
```

### Ports
| Service | Port | URL |
|---------|------|-----|
| Backend (FastAPI) | 8001 | http://localhost:8001 |
| Backend API Docs | 8001 | http://localhost:8001/docs |
| Frontend (Vite) | 5173 | http://localhost:5173 |

### Proxy Configuration
Vite dev server proxy tất cả API paths (`/root-finding`, `/linear-system`, ...) đến `http://localhost:8001`.

### Dependencies
- **Python:** fastapi, uvicorn, numpy, scipy, pydantic, sympy
- **Node:** react 19, @mui/material 9, react-router-dom 7, axios, plotly.js, mathjax (CDN), typescript 6

---

## 7. Các điểm kỹ thuật nổi bật

### 7.1 Xử lý toán học an toàn

- **Machine epsilon awareness:** Các thuật toán sử dụng `MACHINE_EPSILON` (~2.22e-16) để kiểm tra hội tụ thực tế, không ép người dùng chọn ε nhỏ hơn giới hạn phần cứng.
- **Real root functions:** Triển khai `_real_cbrt()` và `_real_root()` để tránh trả về số phức khi tính căn bậc lẻ của số âm (ví dụ: ∛(-8) = -2 thay vì NaN/complex).
- **Divergence detection:** Kiểm tra `|x| > 10^10`, NaN, Inf, overflow, giá trị phức, step quá lớn.
- **Numerical Jacobian:** Dùng central differences (`h=1e-8`) cho Newton Multivariable.

### 7.2 Hiển thị LaTeX xuyên suốt

- **MathJax CDN** được load trong `index.html`.
- **Frontend:** `FormulaRenderer` tự động typeset MathJax. `FormulaEditor` cho phép nhập LaTeX với preview real-time.
- **Backend:** Tất cả steps ma trận đều được chuyển thành LaTeX (`_matrix_to_latex()`) để frontend hiển thị đẹp.
- **Bảng lặp:** Headers tự động chuyển thành LaTeX (x_k → `\(x_k\)`, f'(x_k) → `\(f'(x_k)\)`).

### 7.3 Phân tích ma trận thông minh

- Tự động detect: vuông, đối xứng, xác định dương, ba đường chéo, chéo trội.
- Đưa ra khuyến nghị phương pháp phù hợp (VD: "Ma trận ba đường chéo → khuyến nghị Thomas Algorithm").
- Phân loại nghiệm: unique, inconsistent, infinite.
- Với vô số nghiệm: trích xuất biến tự do, nghiệm riêng, vector cơ sở, nghiệm tổng quát LaTeX.

### 7.4 Keyboard Navigation cho Ma trận

`MatrixLatexEditor` hỗ trợ di chuyển bằng phím mũi tên, Tab, Enter giữa các ô trong ma trận A và B, cho phép nhập dữ liệu nhanh.

### 7.5 Dark Mode

Theme MUI với toggle dark/light, tự động điều chỉnh màu sắc (primary, background, paper) theo mode.

### 7.6 Responsive Design

- Sidebar: permanent trên desktop (md+), temporary drawer trên mobile.
- Matrix editor: cuộn ngang khi ma trận lớn.

### 7.7 Giải thuật từng bước chi tiết

- **Direct methods** (Gauss, Gauss-Jordan): Ghi lại từng thao tác hàng (swap, normalize, eliminate) kèm LaTeX.
- **Iterative methods:** Ghi lại giá trị x sau mỗi vòng lặp.
- **SolutionSteps component:** Gộp steps theo phase, cho phép collapse/expand, copy LaTeX, export `.tex`.

### 7.8 Hỗ trợ B đa cột

Linear System solver hỗ trợ B là ma trận m×p (không chỉ vector đơn), giải đồng thời nhiều hệ với cùng A.

---

## Phụ lục: Thống kê Code

| Thành phần | File chính | Số dòng |
|------------|-----------|---------|
| Backend entry | `main.py` | 62 |
| Root finding algorithms | `algorithms/root_finding.py` | 429 |
| Nonlinear system | `algorithms/nonlinear_system.py` | 252 |
| Linear system | `algorithms/linear_system.py` | 1,559 |
| Interpolation | `algorithms/interpolation.py` | 256 |
| Integration | `algorithms/integration.py` | 289 |
| Math parser | `utils/math_parser.py` | 303 |
| API routers | 6 files | ~430 |
| Schemas | 7 files | ~260 |
| **Backend tổng** | | **~3,840** |
| Frontend App | `App.tsx` | 91 |
| Configuration | `modules.tsx` | 111 |
| API Service | `api.ts` | 242 |
| LaTeX utils | `latex.ts` | 210 |
| Components | 7 files | ~1,400 |
| Pages | 8 files | ~1,600 |
| **Frontend tổng** | | **~3,650** |
| **Tổng toàn hệ thống** | | **~7,500 dòng** |

---

*Tài liệu được sinh tự động từ source code. Cập nhật lần cuối: 09/07/2026.*