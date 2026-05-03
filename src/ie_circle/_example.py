import math
from typing import Any

from array_api._2024_12 import Array, ArrayNamespaceFull

from ._bie import NystromInterpolant, QuadratureType, nystrom


def example_13_19(
    a: float,
    b: float,
    n_nodes: int,
    /,
    *,
    xp: ArrayNamespaceFull,
    device: Any = None,
    dtype: Any = None,
) -> NystromInterpolant:
    r"""
    Solves the integral equation.

    $$
    \frac{1}{2\pi} \int_0^{2\pi}
    \left[ \cot\left(\frac{\tau - t}{2}\right) + K(t, \tau) \right]
    \phi(\tau) \, d\tau = f(t)
    $$

    where
    $$
    K(t, \tau)
    = 2 - \frac{(a^2 - b^2) \sin(t + \tau)}{a^2 + b^2 - (a^2 - b^2) \cos(t + \tau)}
    $$
    and
    $$
    f(t) = e^{c \cos t} \sin(c \sin t) + e^{\cos t} \sin(\sin t),
    \qquad c = \frac{a - b}{a + b}.
    $$

    The exact solution is

    $$
    \phi(t) = 1 - e^{\cos t} \cos(\sin t).
    $$

    Parameters
    ----------
    a : float
        The parameter a.
    b : float
        The parameter b.
    n_nodes : int
        The number of nodes $n$.
    xp : ArrayNamespaceFull
        The array namespace.
    device : Any, optional
        The device.
    dtype : Any, optional
        The dtype.

    Returns
    -------
    NystromInterpolant
        The interpolant for the solution of the integral equation.

    """
    c = (a - b) / (a + b)

    def f(t: Array, /) -> Array:
        part1 = xp.exp(c * xp.cos(t)) * xp.sin(c * xp.sin(t))
        part2 = xp.exp(xp.cos(t)) * xp.sin(xp.sin(t))
        return part1 + part2

    def a_func(t: Array, /) -> Array:
        return xp.zeros_like(t)

    def k_reg(t: Array, tau: Array, /) -> Array:
        a2 = a**2
        b2 = b**2

        num = (a2 - b2) * xp.sin(t + tau)
        den = (a2 + b2) - (a2 - b2) * xp.cos(t + tau)

        k_val = 2.0 - (num / den)
        return k_val / (2 * math.pi)

    def k_cot(t: Array, tau: Array, /) -> Array:
        val = -1.0 / (2 * math.pi)
        result = xp.zeros_like(t + tau) + val
        return result

    kernel = {
        (QuadratureType.NO_SINGULARITY, 0): k_reg,
        (QuadratureType.COT_POWER, 1): k_cot,
    }

    return nystrom(
        a=a_func,
        kernel=kernel,
        rhs=f,
        n=n_nodes,
        xp=xp,
        device=device,
        dtype=dtype,
    )
