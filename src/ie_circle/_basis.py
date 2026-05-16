from typing import Any

from array_api._2024_12 import Array, ArrayNamespaceFull

from ._quadrature import (
    trapezoidal_quadrature,
)


def trapezoidal_basis(
    x: Array,
    /,
    *,
    t_start: float | None = None,
    t_start_factor: float | None = None,
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
    t, _ = trapezoidal_quadrature(
        n, xp=xp, device=device, dtype=dtype, t_start=t_start, t_start_factor=t_start_factor
    )
    n_quad = 2 * n - 1
    m = xp.arange(-(n - 1), n, device=device)
    return (
        1 / n_quad * xp.sum(xp.exp(-1j * m[None, :] * (t[:, None] - x[..., None, None])), axis=-1)
    )
