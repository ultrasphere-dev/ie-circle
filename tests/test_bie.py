from __future__ import annotations

from typing import Any

from ie_circle._bie import (
    ArrayFunction,
    KernelFunction,
    QuadratureType,
    nystrom,
    nystrom_lhs,
    nystrom_rhs,
)
from ie_circle._quadrature import (
    cot_power_quadrature,
    log_cot_power_quadrature,
    trapezoidal_quadrature,
)

# Test a simple, uniquely solvable integral equation:
# a(x) φ(x)
# + ∫_0^{2π} [ c0
#            + c1 log(4 sin^2((x-y)/2))
#            + c2 log(4 sin^2((x-y)/2)) cot((x-y)/2)
#            + c3 cot((x-y)/2)
#            + c4 cot^2((x-y)/2) ] φ(y) dy
# = rhs(x).
# We use constant kernels so the expected discretized matrix is a weighted sum
# of quadrature weight matrices plus a diagonal term a(x).


def _constant_kernel(value: float) -> KernelFunction:
    def _kernel(x, y):
        return (x + y) * 0 + value

    return _kernel


def _constant_rhs(value: float) -> ArrayFunction:
    def _rhs(x):
        return (x * 0) + value

    return _rhs


def test_nystrom_lhs_combined_kernels(xp: Any, device: Any, dtype: Any) -> None:
    n = 6

    a0 = 2
    c0 = 0.5
    c1 = -1.0
    c2 = 0.75
    c3 = 1.25
    c4 = -0.25

    kernel = {
        (QuadratureType.NO_SINGULARITY, 0): _constant_kernel(c0),
        (QuadratureType.LOG_COT_POWER, 0): _constant_kernel(c1),
        (QuadratureType.LOG_COT_POWER, 1): _constant_kernel(c2),
        (QuadratureType.COT_POWER, 1): _constant_kernel(c3),
        (QuadratureType.COT_POWER, 2): _constant_kernel(c4),
    }

    a = _constant_rhs(a0)

    x_nodes, A = nystrom_lhs(a, kernel, n, xp=xp, device=device, dtype=dtype)

    _, w = trapezoidal_quadrature(n, xp=xp, device=device, dtype=dtype)
    w_scalar = w[0]
    n_quad = 2 * n - 1
    idx = (
        xp.arange(n_quad, device=device, dtype=xp.int64)[:, None]
        + xp.arange(n_quad, device=device, dtype=xp.int64)[None, :]
    ) % n_quad

    _, w_log0 = log_cot_power_quadrature(n, 0, xp=xp, device=device, dtype=dtype)
    _, w_log1 = log_cot_power_quadrature(n, 1, xp=xp, device=device, dtype=dtype)
    _, w_cot1 = cot_power_quadrature(n, 1, xp=xp, device=device, dtype=dtype)
    _, w_cot2 = cot_power_quadrature(n, 2, xp=xp, device=device, dtype=dtype)

    w_log0_mat = w_log0[idx]
    w_log1_mat = w_log1[idx]
    w_cot1_mat = w_cot1[idx]
    w_cot2_mat = w_cot2[idx]

    a_vals = a(x_nodes)
    A_expected = xp.eye(n_quad, dtype=dtype, device=device) * a_vals[:, None] + (
        c0 * w_scalar + c1 * w_log0_mat + c2 * w_log1_mat + c3 * w_cot1_mat + c4 * w_cot2_mat
    )

    assert xp.max(xp.abs(A - A_expected)) < 1e-10


def test_nystrom_interpolant_matches_solution(xp: Any, device: Any, dtype: Any) -> None:
    n = 6

    a0 = 2
    c0 = 0.5
    c1 = -1.0
    c2 = 0.75
    c3 = 1.25
    c4 = -0.25
    rhs_value = 3.0

    kernel = {
        (QuadratureType.NO_SINGULARITY, 0): _constant_kernel(c0),
        (QuadratureType.LOG_COT_POWER, 0): _constant_kernel(c1),
        (QuadratureType.LOG_COT_POWER, 1): _constant_kernel(c2),
        (QuadratureType.COT_POWER, 1): _constant_kernel(c3),
        (QuadratureType.COT_POWER, 2): _constant_kernel(c4),
    }

    a = _constant_rhs(a0)
    rhs = _constant_rhs(rhs_value)

    x_nodes, A = nystrom_lhs(a, kernel, n, xp=xp, device=device, dtype=dtype)
    _, b = nystrom_rhs(rhs, n, xp=xp, device=device, dtype=dtype)
    sol = xp.linalg.solve(A, b)

    interp = nystrom(a, kernel, rhs, n, xp=xp, device=device, dtype=dtype)
    phi_eval = interp(x_nodes)

    assert xp.max(xp.abs(phi_eval - sol)) < 1e-10
