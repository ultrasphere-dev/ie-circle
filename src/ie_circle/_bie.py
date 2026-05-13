from enum import StrEnum
from typing import Any, Protocol

from array_api._2024_12 import Array, ArrayNamespaceFull
from array_api_compat import array_namespace
from array_api_shape_check import check_shapes
from batch_tensorsolve import btensorsolve

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
            The kernel function values of shape (..., ...(B), C(x), C(y))
            where [..., i, j] corresponds to kernel((x, i), (y, j)),
            C is the number of circles
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
            where [..., i] corresponds to func((x, i)),
            C is the number of circles
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
    *,
    n: int,
    xp: ArrayNamespaceFull,
    device: Any,
    dtype: Any,
    t_start_quadrature: float | None = None,
    t_start_factor_quadrature: float | None = None,
    t_start: float | None = None,
    t_start_factor: float | None = None,
) -> Array:
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
        of shape (...), (...) -> (..., ...(B), C(x), C(y))
        where C is the number of circles
        and B is the batch shape for equations.
    n : int
        The maximum order - 1.
    xp : ArrayNamespaceFull
        The array namespace.
    device : Any
        The device.
    dtype : Any
        The dtype.
    t_start_quadrature : float | None
        Grid shift $t_\mathrm{start}$.
        Applied to column points.
    t_start_factor_quadrature : float | None
        Grid shift as a multiple of $h = 2\pi/(2n-1)$.
        Applied to column points.
    t_start : float | None
        Grid shift $t_\mathrm{start}$.
        Applied to row points.
    t_start_factor : float | None
        Grid shift as a multiple of $h = 2\pi/(2n-1)$.
        Applied to row points.

    Returns
    -------
    Array
        The left-hand side matrix $A$ of shape (...(B), Q(x), C(x), Q(y), C(y)).

    """
    x, w = trapezoidal_quadrature(n, t_start=t_start, xp=xp, device=device, dtype=dtype)
    y, _ = trapezoidal_quadrature(n, t_start=t_start_quadrature, xp=xp, device=device, dtype=dtype)
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

    weight_by_key: dict[tuple[QuadratureType, int], Array] = {}
    for quad_type, order in kernel:
        if quad_type == QuadratureType.NO_SINGULARITY:
            _, w = trapezoidal_quadrature(
                n, t_start=t_start_quadrature, xp=xp, device=device, dtype=dtype
            )
        elif quad_type == QuadratureType.LOG_COT_POWER:
            _, w = log_cot_power_quadrature(
                n,
                order,
                t_start=t_start_quadrature,
                xp=xp,
                device=device,
                dtype=dtype,
            )
        elif quad_type == QuadratureType.COT_POWER:
            _, w = cot_power_quadrature(
                n,
                order,
                t_start=t_start_quadrature,
                xp=xp,
                device=device,
                dtype=dtype,
            )
        else:
            msg = f"Unsupported quadrature type: {quad_type}"  # type: ignore[unreachable]
            raise ValueError(msg)
        weight_by_key[(quad_type, order)] = w

    terms = [
        kernel_fn(x[:, None], y[None, :])
        * weight_by_key[(quad_type, order)][idx_roll][(...,) + (None,) * (B_ndim + 2)]
        for (quad_type, order), kernel_fn in kernel.items()
    ]

    # (Q(x), Q(y), *B, C(x), C(y))
    A = a_vals_expanded + xp.sum(xp.stack(terms), axis=0)
    # (*B, Q(x), C(x), Q(y), C(y))
    A = xp.moveaxis(A, 0, -3)
    A = xp.moveaxis(A, 1, -2)
    return A


def nystrom_rhs(
    rhs: ArrayFunction,
    *,
    n: int,
    xp: ArrayNamespaceFull,
    device: Any,
    dtype: Any,
    t_start: float | None = None,
    t_start_factor: float | None = None,
) -> Array:
    r"""
    Returns the quadrature nodes and right-hand side vector.

    Parameters
    ----------
    rhs : ArrayFunction
        Right-hand side function $\mathrm{rhs}(x)$
        of (...) -> (..., ...(B), C)
        where C is the number of circles
        and B is the batch shape for equations.
    n : int
        The maximum order - 1.
    xp : ArrayNamespaceFull
        The array namespace.
    device : Any
        The device.
    dtype : Any
        The dtype.
    t_start : float | None
        Grid shift $t_\mathrm{start}$.
    t_start_factor : float | None
        Grid shift as a multiple of $h = 2\pi/(2n-1)$.

    Returns
    -------
    tuple[Array, Array]
        The RHS vector of shape (...(B), Q, C)
        where C is the number of circles
        and B is the batch shape for equations.

    """
    x, _ = trapezoidal_quadrature(
        n, xp=xp, device=device, dtype=dtype, t_start=t_start, t_start_factor=t_start_factor
    )
    # (Q, *B, C)
    b = rhs(x)
    b = xp.moveaxis(b, 0, -2)
    return b


def trapezoidal_basis(
    x: Array,
    /,
    *,
    t_start: float | None,
    t_start_factor: float | None,
    n: int,
    xp: ArrayNamespaceFull,
    device: Any,
    dtype: Any,
) -> Array:
    r"""
    Evaluates the basis.

    $
    1/N' \sum_(\abs(m) < N) exp(-im(t_j + t_\mathrm{start})) * exp(imx)
    $

    Parameters
    ----------
    x : Array
        The points to evaluate of shape (...,).
    n : int
        The maximum order - 1.
    t_start : float | None
        Grid shift $t_\mathrm{start}$.
    t_start_factor : float | None
        Grid shift as a multiple of $h = 2\pi/(2n-1)$.
    xp : ArrayNamespaceFull
        The array namespace.
    device : Any
        The device.
    dtype : Any
        The dtype.

    Returns
    -------
    Array
        The basis evaluated at x of shape (..., n).

    """
    t, _ = trapezoidal_quadrature(n, xp=xp, device=device, dtype=dtype, t_start=t_start)
    n_quad = 2 * n - 1
    m = xp.arange(-(n - 1), n, device=device)
    return (
        1 / n_quad * xp.sum(xp.exp(-1j * m[None, :] * (t[:, None] - x[..., None, None])), axis=-1)
    )


def nystrom(
    a: ArrayFunction,
    kernel: Kernel,
    rhs: ArrayFunction,
    n: int,
    xp: ArrayNamespaceFull,
    device: Any,
    dtype: Any,
    *,
    t_start_quadrature: float | None = None,
    t_start_factor_quadrature: float | None = None,
    t_start: float | None = None,
    t_start_factor: float | None = None,
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
        Multiplicative term $a(x)$
        of (...) -> (..., ...(B), C)
        where C is the number of circles
        and B is the batch shape for equations.
    kernel : Kernel
        Kernel functions keyed by ``(QuadratureType, order)``
        of shape (...), (...) -> (..., ...(B), C(x), C(y))
        where C is the number of circles
        and B is the batch shape for equations.
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
    t_start_quadrature : float | None
        Grid shift $t_\mathrm{start}$.
        Applied to column points.
    t_start_factor_quadrature : float | None
        Grid shift as a multiple of $h = 2\pi/(2n-1)$.
        Applied to column points.
    t_start : float | None
        Grid shift $t_\mathrm{start}$.
        Applied to row points.
    t_start_factor : float | None
        Grid shift as a multiple of $h = 2\pi/(2n-1)$.
        Applied to row points.

    Returns
    -------
    NystromInterpolant
        An object with ``sol`` of shape (...(B), C) where C is the number of circles
        and B is the batch shape for equations,
        and callable to evaluate the Nyström interpolant at arbitrary points
        of (...) -> (..., ...(B), C).

    """
    A = nystrom_lhs(
        a,
        kernel,
        n=n,
        xp=xp,
        device=device,
        dtype=dtype,
        t_start_quadrature=t_start_quadrature,
        t_start=t_start,
    )
    b = nystrom_rhs(
        rhs, n=n, xp=xp, device=device, dtype=dtype, t_start=t_start, t_start_factor=t_start_factor
    )
    # (*B, Q, C)
    sol = btensorsolve(A, b, num_batch_axes=2)

    class _Interpolant:
        def __init__(self, sol_values: Array) -> None:
            self.sol = sol_values

        def __call__(self, x: Array, /) -> Array:
            xp = array_namespace(x, self.sol)
            # (..., Q)
            basis_x = trapezoidal_basis(
                x,
                n=n,
                t_start=t_start,
                t_start_factor=t_start_factor,
                xp=xp,
                device=device,
                dtype=dtype,
            )
            # (..., *B, Q, C)
            basis_x = basis_x[..., None, :, :]
            return xp.sum(self.sol * basis_x, axis=(-1, -2))

    return _Interpolant(sol)
