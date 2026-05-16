from enum import StrEnum
from typing import Any, Protocol

from array_api._2024_12 import Array, ArrayNamespaceFull
from array_api_compat import array_namespace
from array_api_shape_check import check_shapes
from batch_tensorsolve import btensorsolve

from ._basis import trapezoidal_basis
from ._quadrature import (
    _resolve_t_start,
    cot_power_quadrature,
    log_cot_power_quadrature,
    trapezoidal_quadrature,
)


class QuadratureType(StrEnum):
    NO_SINGULARITY = "no_singularity"
    r"""
    $$
    K_{\mathrm{reg}}(x, y)
    $$
    """
    LOG_COT_POWER = "log_cot_power"
    r"""
    $$
    K_{\log,n}(x, y) \log\left(4\sin^2\frac{x - y}{2}\right)\cot^n\!\left(\frac{x - y}{2}\right)

    If ``n`` is odd, be careful about the order of x and y, as
    $$
    \cot \left(\frac{x - y}{2}\right) = -\cot \left(\frac{y - x}{2}\right).
    $$
    """
    COT_POWER = "cot_power"
    r"""
    $$
    K_{\cot,n}(x, y) \cot^n\!\left(\frac{x - y}{2}\right)
    $$

    If ``n`` is odd, be careful about the order of x and y, as
    $$
    \cot \left(\frac{x - y}{2}\right) = -\cot \left(\frac{y - x}{2}\right).
    $$

    If ``n`` is 0, same as ``NO_SINGULARITY``.
    """


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


Kernels = dict[tuple[QuadratureType, int], KernelFunction]


def nystrom_lhs(
    a: ArrayFunction,
    kernels: Kernels,
    *,
    n: int,
    xp: ArrayNamespaceFull,
    device: Any,
    dtype: Any,
    t_start_sol: float | None = None,
    t_start_factor_sol: float | None = None,
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

    If ``n`` is odd, be careful about the order of x and y, as
    $$
    \cot \left(\frac{x - y}{2}\right) = -\cot \left(\frac{y - x}{2}\right).
    $$

    Parameters
    ----------
    a : ArrayFunction
        Multiplicative term $a(x)$
        of (...) -> (..., ...(B), C)
        where C is the number of circles
        and B is the batch shape for equations.
    kernels : Kernel
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
    t_start_sol : float | None
        Grid shift $t_\mathrm{start}$.
        Applied to column points.
    t_start_factor_sol : float | None
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
    t_start_ = _resolve_t_start(
        n,
        t_start=t_start,
        t_start_factor=t_start_factor,
    )
    t_start_sol_ = _resolve_t_start(
        n,
        t_start=t_start_sol,
        t_start_factor=t_start_factor_sol,
    )
    del t_start, t_start_factor, t_start_sol, t_start_factor_sol
    x, _ = trapezoidal_quadrature(n, t_start=t_start_, xp=xp, device=device, dtype=dtype)
    y, _ = trapezoidal_quadrature(n, t_start=t_start_sol_, xp=xp, device=device, dtype=dtype)
    # if not xp.all(xpx.isclose(x, y)):
    #     raise NotImplementedError(
    #         "Currently only supports the case where row and column points are the same."
    #     )
    n_quad = 2 * n - 1
    # Singular qudarature rule should be generated for each singular point y.
    # However, since if "roll" (minus) the quadrature along the y axis,
    # the quadrature rule for y=0 can be reused.
    # -> y
    # S R1 R2 R3 ...
    # R-1 S R1 R2 ...
    # R-2 R-1 S R1 ...
    idx_roll = (
        -xp.arange(n_quad, device=device, dtype=xp.int64)[:, None]
        + xp.arange(n_quad, device=device, dtype=xp.int64)[None, :]
    ) % n_quad

    # Evaluated at x
    # (n_quad, *B, C)
    a_vals = a(x)

    # Q(x), *B, C
    info_a = check_shapes("Q*BC", a_vals, names="a_vals")
    B_shape = info_a.unique["B"].shape_broadcasted
    B_ndim = len(B_shape)
    C = info_a.unique["C"].shape_broadcasted[-1]

    # (Q(x), Q(y), *B, C(x), C(y))
    a_vals = a_vals[:, None, ..., :, None] * xp.eye(C, device=device, dtype=dtype)
    basis_y_at_x = trapezoidal_basis(
        x,
        n=n,
        t_start=t_start_sol_,
        xp=xp,
        device=device,
        dtype=dtype,
    )[:, :, None, None]
    a_mat = a_vals * basis_y_at_x

    A_terms = [a_mat]
    for (quad_type, order), kernel in kernels.items():
        if quad_type == QuadratureType.NO_SINGULARITY:
            _, w = trapezoidal_quadrature(
                n,
                t_start=t_start_sol_ - t_start_,
                xp=xp,
                device=device,
                dtype=dtype,
            )
            w = w[None]
        elif quad_type == QuadratureType.LOG_COT_POWER:
            _, w = log_cot_power_quadrature(
                n,
                order,
                t_start=t_start_sol_ - t_start_,
                xp=xp,
                device=device,
                dtype=dtype,
            )
            # Currently cot^n ((\tau - t)/2)
            # Need to flip to cot^n ((t - \tau)/2)
            # to match the kernel implementation which assumes cot^n ((t - \tau)/2)
            w *= (-1) ** order
            w = w[idx_roll]
        elif quad_type == QuadratureType.COT_POWER:
            _, w = cot_power_quadrature(
                n,
                order,
                t_start=t_start_sol_ - t_start_,
                xp=xp,
                device=device,
                dtype=dtype,
            )
            # Currently \log(4\sin^2((\tau - t)/2))\cot^n ((\tau - t)/2)
            # Need to flip to \log(4\sin^2((t - \tau)/2))\cot^n ((t - \tau)/2)
            # to match the kernel implementation which assumes \log(4\sin^2((t - \tau)/2))
            w *= (-1) ** order
            w = w[idx_roll]
        else:
            msg = f"Unsupported quadrature type: {quad_type}"  # type: ignore[unreachable]
            raise ValueError(msg)

        # (Q, Q, *B, C(x), C(y))
        kernel_val = kernel(x[:, None], y[None, :])
        check_shapes(
            "QQ*BCC,QQ,Q*BC",
            kernel_val,
            w,
            (n_quad, *B_shape, C),
            names=f"kernels[{quad_type}{order}],w,none",
        )
        A_terms.append(kernel_val * w[(...,) + (None,) * (B_ndim + 2)])

    A_terms = xp.broadcast_arrays(*A_terms)
    # (Q(x), Q(y), *B, C(x), C(y))
    A = xp.sum(xp.stack(A_terms, axis=0), axis=0)
    # -> (Q(y), *B, Q(x), C(x), C(y))
    A = xp.moveaxis(A, 0, -3)
    # -> (*B, Q(x), C(x), Q(y), C(y))
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


def nystrom(
    a: ArrayFunction,
    kernels: Kernels,
    rhs: ArrayFunction,
    /,
    *,
    n: int,
    xp: ArrayNamespaceFull,
    device: Any,
    dtype: Any,
    t_start_sol: float | None = None,
    t_start_factor_sol: float | None = None,
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
    kernels : Kernel
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
    t_start_sol : float | None
        Grid shift $t_\mathrm{start}$.
        Applied to column points.
    t_start_factor_sol : float | None
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
        kernels,
        n=n,
        xp=xp,
        device=device,
        dtype=dtype,
        t_start_sol=t_start_sol,
        t_start_factor_sol=t_start_factor_sol,
        t_start=t_start,
        t_start_factor=t_start_factor,
    )
    b = nystrom_rhs(
        rhs, n=n, xp=xp, device=device, dtype=dtype, t_start=t_start, t_start_factor=t_start_factor
    )
    info = check_shapes("*BQCQC,*BQC", A, b)
    B_ndim = len(info.unique["B"].shape_broadcasted)
    # (*B, Q, C)
    sol = btensorsolve(A, b, num_batch_axes=B_ndim)
    # Solution is evaluated at
    # trapezoidal_quadrature(t_start_sol)
    # , not t_start

    class _Interpolant:
        def __init__(self, sol_values: Array) -> None:
            self.sol = sol_values

        def __call__(self, x: Array, /) -> Array:
            xp = array_namespace(x, self.sol)
            # (..., Q)
            basis_x = trapezoidal_basis(
                x,
                n=n,
                t_start=t_start_sol,
                t_start_factor=t_start_factor_sol,
                xp=xp,
                device=device,
                dtype=dtype,
            )
            # (..., *B, Q, C)
            basis_x = basis_x[(...,) + (None,) * B_ndim + (slice(None), None)]
            check_shapes("...*BQC,*BQC", basis_x, self.sol, names="basis_x,sol")
            return xp.sum(self.sol * basis_x, axis=(-1, -2))

    return _Interpolant(sol)
