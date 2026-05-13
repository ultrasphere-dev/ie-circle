import math
from typing import Any

from array_api._2024_12 import Array, ArrayNamespaceFull
from array_api_compat import array_namespace

from ._bie import NystromInterpolant, QuadratureType, nystrom


def example_13_19_answer(t: Array, /) -> Array:
    r"""
    Returns the exact solution of the example integral equation.

    Parameters
    ----------
    t : Array
        The evaluation points of shape (...,).

    Returns
    -------
    Array
        The exact solution $\phi(t) = 1 - e^{\cos t} \cos(\sin t)$ of shape (...,).

    """
    xp = array_namespace(t)  # type: ignore[arg-type]
    return 1 - xp.exp(xp.cos(t)) * xp.cos(xp.sin(t))


def example_13_19(
    a: float,
    b: float,
    n: int,
    /,
    *,
    xp: ArrayNamespaceFull,
    device: Any = None,
    dtype: Any = None,
    t_start_quadrature: float | None = None,
    t_start_factor_quadrature: float | None = None,
    t_start: float | None = None,
    t_start_factor: float | None = None,
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
    \phi(t) = \text{example\_13\_19\_answer}(t) = 1 - e^{\cos t} \cos(\sin t).
    $$

    Parameters
    ----------
    a : float
        The parameter a.
    b : float
        The parameter b.
    n : int
        The maximum order - 1.
    xp : ArrayNamespaceFull
        The array namespace.
    device : Any, optional
        The device.
    dtype : Any, optional
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
        The interpolant for the solution of the integral equation.

    """
    c = (a - b) / (a + b)

    def f(t: Array, /) -> Array:
        part1 = xp.exp(c * xp.cos(t)) * xp.sin(c * xp.sin(t))
        part2 = xp.exp(xp.cos(t)) * xp.sin(xp.sin(t))
        return (part1 + part2)[..., None]

    def a_func(t: Array, /) -> Array:
        return xp.zeros_like(t)[..., None]

    def k_reg(t: Array, tau: Array, /) -> Array:
        a2 = a**2
        b2 = b**2

        num = (a2 - b2) * xp.sin(t + tau)
        den = (a2 + b2) - (a2 - b2) * xp.cos(t + tau)

        k_val = 2.0 - (num / den)
        return (k_val / (2 * math.pi))[..., None, None]

    def k_cot(t: Array, tau: Array, /) -> Array:
        val = -1.0 / (2 * math.pi)
        result = xp.zeros_like(t + tau) + val
        return result[..., None, None]

    kernel = {
        (QuadratureType.NO_SINGULARITY, 0): k_reg,
        (QuadratureType.COT_POWER, 1): k_cot,
    }

    return nystrom(
        a_func,
        kernel,
        f,
        n=n,
        xp=xp,
        device=device,
        dtype=dtype,
        t_start_quadrature=t_start_quadrature,
        t_start_factor_quadrature=t_start_factor_quadrature,
        t_start=t_start,
        t_start_factor=t_start_factor,
    )


def example_simple(
    n: int,
    /,
    *,
    xp: ArrayNamespaceFull,
    device: Any = None,
    dtype: Any = None,
    t_start_quadrature: float | None = None,
    t_start_factor_quadrature: float | None = None,
    t_start: float | None = None,
    t_start_factor: float | None = None,
) -> NystromInterpolant:
    r"""
    Solves the simple test integral equation.

    $$
    \phi(t) + \int_0^{2\pi} \cos(t - \tau) \phi(\tau) \, d\tau = (1 + \pi) \cos(t)
    $$

    The exact solution is

    $$
    \phi(t) = \text{example\_simple\_answer}(t) = \cos(t).
    $$

    Parameters
    ----------
    n : int
        The maximum order - 1.
    xp : ArrayNamespaceFull
        The array namespace.
    device : Any, optional
        The device.
    dtype : Any, optional
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
        The interpolant for the solution of the integral equation.

    """

    def f(t: Array, /) -> Array:
        return ((1 + math.pi) * xp.cos(t))[..., None]

    def a_func(t: Array, /) -> Array:
        return xp.ones_like(t)[..., None]

    def k_reg(t: Array, tau: Array, /) -> Array:
        return xp.cos(t - tau)[..., None, None]

    kernel = {
        (QuadratureType.NO_SINGULARITY, 0): k_reg,
    }

    return nystrom(
        a_func,
        kernel,
        f,
        n=n,
        xp=xp,
        device=device,
        dtype=dtype,
        t_start_quadrature=t_start_quadrature,
        t_start_factor_quadrature=t_start_factor_quadrature,
        t_start=t_start,
        t_start_factor=t_start_factor,
    )


def example_simple_answer(t: Array, /) -> Array:
    r"""
    Returns the exact solution of the simple test integral equation.

    Parameters
    ----------
    t : Array
        The evaluation points of shape (...,).

    Returns
    -------
    Array
        The exact solution $\phi(t) = \cos(t)$ of shape (...,).

    """
    xp = array_namespace(t)  # type: ignore[arg-type]
    return xp.cos(t)


def example_13_23_answer(t: Array, /) -> Array:
    r"""
    Returns the exact solution of the example 13.23 integral equation.

    Parameters
    ----------
    t : Array
        The evaluation points of shape (...,).

    Returns
    -------
    Array
        The exact solution $\phi(t) = e^{\cos t} \cos(t + \sin t)$ of shape (...,).

    """
    xp = array_namespace(t)  # type: ignore[arg-type]
    return xp.exp(xp.cos(t)) * xp.cos(t + xp.sin(t))


def example_13_23(
    a: float,
    b: float,
    n: int,
    /,
    *,
    xp: ArrayNamespaceFull,
    device: Any = None,
    dtype: Any = None,
    t_start_quadrature: float | None = None,
    t_start_factor_quadrature: float | None = None,
    t_start: float | None = None,
    t_start_factor: float | None = None,
) -> NystromInterpolant:
    r"""
    Solves the integral equation for Example 13.23.

    $$
    S_0 \phi - A \phi = f
    $$

    where
    $$
    (S_0 \phi)(t) := \frac{1}{2\pi} \int_0^{2\pi}
    \left\{ \ln\left(4 \sin^2 \frac{t - \tau}{2}\right) - 2 \right\}
    \phi(\tau) \, d\tau
    $$
    and
    $$
    (A \phi)(t) := \frac{1}{2\pi} \int_0^{2\pi}
    \left\{ K(t, \tau) \ln\left(4 \sin^2 \frac{t - \tau}{2}\right) + L(t, \tau) \right\}
    \phi(\tau) \, d\tau.
    $$

    For this example, $K(t, \tau) = 0$ and
    $$
    L(t, \tau) = -\ln\{a^2 + b^2 - (a^2 - b^2)\cos(t + \tau)\} - 3.
    $$

    The exact solution is
    $$
    \phi(t) = \text{example\_13\_23\_answer}(t) = e^{\cos t} \cos(t + \sin t)
    $$
    and the right-hand side is
    $$
    f(t) = 2 - e^{\cos t} \cos(\sin t) - e^{c \cos t} \cos(c \sin t),
    \qquad c = \frac{a - b}{a + b}.
    $$

    Parameters
    ----------
    a : float
        The parameter a.
    b : float
        The parameter b.
    n : int
        The maximum order - 1.
    xp : ArrayNamespaceFull
        The array namespace.
    device : Any, optional
        The device.
    dtype : Any, optional
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
        The interpolant for the solution of the integral equation.

    """
    c = (a - b) / (a + b)

    def f(t: Array, /) -> Array:
        part1 = xp.exp(xp.cos(t)) * xp.cos(xp.sin(t))
        part2 = xp.exp(c * xp.cos(t)) * xp.cos(c * xp.sin(t))
        return (2.0 - part1 - part2)[..., None]

    def a_func(t: Array, /) -> Array:
        return xp.zeros_like(t)[..., None]

    def k_reg(t: Array, tau: Array, /) -> Array:
        a2 = a**2
        b2 = b**2
        inner = a2 + b2 - (a2 - b2) * xp.cos(t + tau)
        k_val = -2 - (-xp.log(inner) - 3)
        return (k_val / (2 * math.pi))[..., None, None]

    def k_log(t: Array, tau: Array, /) -> Array:
        val = 1.0 / (2 * math.pi)
        result = xp.ones_like(t + tau) * val
        return result[..., None, None]

    kernel = {
        (QuadratureType.NO_SINGULARITY, 0): k_reg,
        (QuadratureType.LOG_COT_POWER, 0): k_log,
    }

    return nystrom(
        a_func,
        kernel,
        f,
        n=n,
        xp=xp,
        device=device,
        dtype=dtype,
        t_start_quadrature=t_start_quadrature,
        t_start_factor_quadrature=t_start_factor_quadrature,
        t_start=t_start,
        t_start_factor=t_start_factor,
    )
