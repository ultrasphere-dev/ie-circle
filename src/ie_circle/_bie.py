from enum import StrEnum
from typing import Any, Protocol

from array_api._2024_12 import Array, ArrayNamespaceFull
from array_api_shape_check import check_shapes

from ._quadrature import (
    cot_power_quadrature,
    log_cot_power_quadrature,
    trapezoidal_quadrature,
)


class QuadratureType(StrEnum):
    NO_SINGULARITY = "no_singularity"
    LOG_COT_POWER = "log_cot_power"
    COT_POWER = "cot_power"


class KernelFunction(Protocol):
    def __call__(self, x: Array, y: Array, /) -> Array:
        """
        Kernel function.

        Parameters
        ----------
        x : Array
            An array of shape (...).
        y : Array
            An array of shape (...).

        Returns
        -------
        Array
            The kernel function values of shape (..., ...(B), C, C)
            where C is the number of circles
            and B is the batch shape for equations.

        """
        ...


class ArrayFunction(Protocol):
    def __call__(self, x: Array, /) -> Array:
        """
        Array-valued function.

        Parameters
        ----------
        x : Array
            An array of shape (...).

        Returns
        -------
        Array
            Function values of shape (..., ...(B), C)
            where C is the number of circles
            and B is the batch shape for equations.

        """
        ...


class NystromInterpolant(Protocol):
    sol: Array

    def __call__(self, x: Array, /) -> Array:
        """
        Nyström interpolant.

        Parameters
        ----------
        x : Array
            Evaluation points of shape (...).

        Returns
        -------
        Array
            Interpolated values of shape (..., ...(B), C)
            where C is the number of circles
            and B is the batch shape for equations.

        """
        ...


Kernel = dict[tuple[QuadratureType, int], KernelFunction]


def nystrom_lhs(
    a: ArrayFunction,
    kernel: Kernel,
    n: int,
    xp: ArrayNamespaceFull,
    device: Any,
    dtype: Any,
    *,
    t_start_quadrature: float = 0,
    t_start: float = 0,
) -> tuple[Array, Array]:
    r"""
    Returns the left-hand side matrix $A$ of the Nystrom method for the integral equation.

    $$
    a(x) \phi (x)
    + \int_0^{2\pi} \Bigg( K_{\mathrm{reg}}(x, y)
    + \sum_{n\ge 0} K_{\log,n}(x, y)
    \log\left(4\sin^2\frac{x - y}{2}\right)\cot^n\!\left(\frac{x - y}{2}\right)
    + \sum_{n\ge 0} K_{\cot,n}(x, y)
    \cot^n\!\left(\frac{x - y}{2}\right) \Bigg)\,\phi (y)\,dy
    = \text{rhs} (x)
    $$

    Parameters
    ----------
    a : ArrayFunction
        Multiplicative term $a(x)$
        of (...) -> (..., ...(B), C)
        where C is the number of circles
        and B is the batch shape for equations.
    kernel : Kernel
        Kernel functions keyed by ``(QuadratureType, order)``
        of shape (...), (...) -> (..., ...(B), C, C)
        where C is the number of circles
        and B is the batch shape for equations.
    n : int
        Number of discretization points / 2
    xp : ArrayNamespaceFull
        The array namespace.
    device : Any
        The device.
    dtype : Any
        The dtype.
    t_start_quadrature : float
        Quadrature node shift applied in weight construction.
    t_start : float
        Shift applied to evaluation points for $a(x)$ and kernel functions.

    Returns
    -------
    tuple[Array, Array]
        The roots $x_j$ of shape (2n - 1,)
        and the left-hand side matrix $A$ of shape (..., ...(B), C, C).

    """
    x, w = trapezoidal_quadrature(n, t_start=t_start, xp=xp, device=device, dtype=dtype)
    n_quad = 2 * n - 1
    idx_roll = (
        xp.arange(n_quad, device=device, dtype=xp.int64)[:, None]
        + xp.arange(n_quad, device=device, dtype=xp.int64)[None, :]
    ) % n_quad

    # (n_quad, *B, C)
    a_vals = a(x)
    info_a = check_shapes("Q*BC", a_vals, names="a_vals")
    B_shape = info_a.unique["b"].shape_broadcasted
    B_ndim = len(B_shape)
    C = info_a.unique["c"].shape_broadcasted[-1]
    a_vals_expanded = (
        a_vals[:, None, ..., :, None]
        * xp.eye(n_quad)[(...,) + (None,) * (B_ndim + 2)]
        * xp.eye(C)[(None,) * (B_ndim + 2) + (...,)]
    )
    # (n_quad, 1)
    x_kernel = x[:, None]
    # (1, n_quad)
    y_kernel = x[None, :]

    weight_by_key: dict[tuple[QuadratureType, int], Array] = {}
    for quad_type, order in kernel:
        if quad_type == QuadratureType.NO_SINGULARITY:
            weight_by_key[(quad_type, order)] = w[idx_roll][(...,) + (None,) * (B_ndim + 2)]
        elif quad_type == QuadratureType.LOG_COT_POWER:
            _, w_log_vec = log_cot_power_quadrature(
                n,
                order,
                t_start=t_start_quadrature,
                xp=xp,
                device=device,
                dtype=dtype,
            )
            weight_by_key[(quad_type, order)] = w_log_vec[idx_roll][(...,) + (None,) * (B_ndim + 2)]
        elif quad_type == QuadratureType.COT_POWER:
            _, w_cauchy_vec = cot_power_quadrature(
                n,
                order,
                t_start=t_start_quadrature,
                xp=xp,
                device=device,
                dtype=dtype,
            )
            weight_by_key[(quad_type, order)] = w_cauchy_vec[idx_roll][
                (...,) + (None,) * (B_ndim + 2)
            ]
        else:
            msg = f"Unsupported quadrature type: {quad_type}"  # type: ignore[unreachable]
            raise ValueError(msg)

    terms = [
        kernel_fn(x_kernel, y_kernel) * weight_by_key[(quad_type, order)]
        for (quad_type, order), kernel_fn in kernel.items()
    ]

    A = a_vals_expanded + xp.sum(xp.stack(terms), axis=0)
    return x, A


def nystrom_rhs(
    rhs: ArrayFunction,
    n: int,
    xp: ArrayNamespaceFull,
    device: Any,
    dtype: Any,
    *,
    t_start: float = 0,
) -> tuple[Array, Array]:
    r"""
    Returns the quadrature nodes and right-hand side vector.

    Parameters
    ----------
    rhs : ArrayFunction
        Right-hand side function $\mathrm{rhs}(x)$.
    n : int
        Number of discretization points / 2.
    xp : ArrayNamespaceFull
        The array namespace.
    device : Any
        The device.
    dtype : Any
        The dtype.
    t_start : float
        Grid shift for the quadrature nodes.

    Returns
    -------
    tuple[Array, Array]
        The roots $x_j$ of shape (2n - 1,)
        and the RHS vector of shape (..., ...(B), C).

    """
    x, _ = trapezoidal_quadrature(n, t_start=t_start, xp=xp, device=device, dtype=dtype)
    b_vec = rhs(x)
    return x, b_vec


def nystrom(
    a: ArrayFunction,
    kernel: Kernel,
    rhs: ArrayFunction,
    n: int,
    xp: ArrayNamespaceFull,
    device: Any,
    dtype: Any,
    *,
    t_start_quadrature: float = 0,
    t_start: float = 0,
) -> NystromInterpolant:
    r"""
    Solves integral equations using the Nyström method.

    $$
    a(x) \phi (x)
    + \int_0^{2\pi} \Bigg( K_{\mathrm{reg}}(x, y)
    + \sum_{n\ge 0} K_{\log,n}(x, y)
    \log\left(4\sin^2\frac{x - y}{2}\right)\cot^n\!\left(\frac{x - y}{2}\right)
    + \sum_{n\ge 0} K_{\cot,n}(x, y)
    \cot^n\!\left(\frac{x - y}{2}\right) \Bigg)\,\phi (y)\,dy
    = \text{rhs} (x)
    $$

    Parameters
    ----------
    a : ArrayFunction
        Multiplicative term $a(x)$.
    kernel : Kernel
        Kernel functions keyed by ``(QuadratureType, order)``.
    rhs : ArrayFunction
        Right-hand side function $\mathrm{rhs}(x)$.
    n : int
        Number of discretization points / 2.
    xp : ArrayNamespaceFull
        The array namespace.
    device : Any
        The device.
    dtype : Any
        The dtype.
    t_start_quadrature : float
        Quadrature node shift applied in weight construction.
    t_start : float
        Shift applied to evaluation points for $a(x)$ and kernel functions.

    Returns
    -------
    NystromInterpolant
        An object with ``sol`` and ``__call__`` implementing the Nyström interpolant.

    """
    x, A = nystrom_lhs(
        a,
        kernel,
        n,
        xp,
        device,
        dtype,
        t_start_quadrature=t_start_quadrature,
        t_start=t_start,
    )
    _, b = nystrom_rhs(rhs, n, xp, device, dtype, t_start=t_start)
    sol = xp.linalg.solve(A, b)

    n_quad = 2 * n - 1
    m = xp.arange(-(n - 1), n, device=device)
    x_nodes = x
    phase_nodes = xp.exp((-1j) * m[:, None] * x_nodes[None, :])
    coeffs = xp.sum(sol[None, :] * phase_nodes, axis=1)

    class _Interpolant:
        def __init__(self, sol_values: Array) -> None:
            self.sol = sol_values

        def __call__(self, x_eval: Array, /) -> Array:
            x_flat = xp.reshape(x_eval, (-1,))
            phase_eval = xp.exp(1j * m[:, None] * x_flat[None, :])
            values = (1 / n_quad) * xp.sum(coeffs[:, None] * phase_eval, axis=0)
            return xp.reshape(values, x_flat.shape)

    return _Interpolant(sol)
