from __future__ import annotations

import warnings
from functools import cache
from typing import Any

from array_api._2024_12 import Array, ArrayNamespaceFull


@cache
def harmonic_number(n: int, /) -> float:
    r"""Return the harmonic number $H_n = \sum_{k=1}^n 1/k$."""
    if n < 0:
        msg = "n must be non-negative."
        raise ValueError(msg)
    if n == 0:
        return 0
    return harmonic_number(n - 1) + 1 / n


def cot_power_fourier_integral_coefficients(
    n_harmonics: int,
    power: int,
    /,
    *,
    xp: ArrayNamespaceFull,
    device: Any,
    dtype: Any,
) -> Array:
    r"""
    Fourier coefficients of the finite-part integral of $\cot^{\mathrm{power}}(t/2)$.

    Returns $I_{m,\mathrm{power}}$ for $m = -(n_harmonics-1), \ldots, n_harmonics-1$.

    Parameters
    ----------
    n_harmonics : int
        Harmonics with order less than ``n_harmonics``.
    power : int
        The exponent ``n`` in $I_{m,n}$.
    xp : ArrayNamespaceFull
        The array namespace.
    device : Any
        The device.
    dtype : Any
        The dtype.

    Returns
    -------
    Array
        Complex-valued coefficients $I_{m,\mathrm{power}}$ of shape (2*n_harmonics - 1,).

    """
    if n_harmonics <= 0:
        msg = "n_harmonics must be positive."
        raise ValueError(msg)
    if power < 0:
        msg = "power must be non-negative."
        raise ValueError(msg)

    two_pi = 2 * xp.asarray(xp.pi, dtype=dtype)

    # m = -(n_harmonics-1), ..., (n_harmonics-1)
    m = xp.arange(-(n_harmonics - 1), n_harmonics, device=device)

    # Initial values
    i0 = xp.where(m == 0, two_pi, 0)
    i1 = xp.where(m == 0, 0, two_pi * 1j * xp.sign(m))

    if power == 0:
        return i0
    if power == 1:
        return i1

    i_nm2 = i0
    i_nm1 = i1
    for k in range(2, power + 1):
        i_n = (2j * m) / (k - 1) * i_nm1 - i_nm2
        i_nm2, i_nm1 = i_nm1, i_n
    return i_nm1


def log_cot_power_fourier_integral_coefficients(
    n_harmonics: int,
    power: int,
    /,
    *,
    xp: ArrayNamespaceFull,
    device: Any,
    dtype: Any,
) -> Array:
    r"""
    Fourier coefficients of the finite-part integral of
    $\log(4\sin^2(t/2))\,\cot^{\mathrm{power}}(t/2)$.

    Returns $J_{m,\mathrm{power}}$ for $m = -(n_harmonics-1), \ldots, n_harmonics-1$.

    Parameters
    ----------
    n_harmonics : int
        Harmonics with order less than ``n_harmonics``.
    power : int
        The exponent ``n`` in $J_{m,n}$.
    xp : ArrayNamespaceFull
        The array namespace.
    device : Any
        The device.
    dtype : Any
        The dtype.

    Returns
    -------
    Array
        Complex-valued coefficients $J_{m,\mathrm{power}}$ of shape (2*n_harmonics - 1,).

    """
    if n_harmonics <= 0:
        msg = "n_harmonics must be positive."
        raise ValueError(msg)
    if power < 0:
        msg = "power must be non-negative."
        raise ValueError(msg)

    two_pi = 2 * xp.pi

    m = xp.arange(-(n_harmonics - 1), n_harmonics, device=device)
    abs_m = xp.abs(m)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="divide by zero encountered in divide",
        )
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="invalid value encountered in multiply",
        )
        inv_abs_m = 1 / xp.astype(abs_m, dtype)

    # Initial values
    j0 = xp.where(m == 0, 0 + 0.0j, (-two_pi * inv_abs_m) + 0.0j)

    # Harmonic numbers are computed on CPU as Python scalars.
    h = xp.asarray(
        [harmonic_number(int(k)) for k in abs_m],
        device=device,
        dtype=dtype,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="invalid value encountered in multiply",
        )
        j1 = xp.where(
            m == 0,
            0 + 0.0j,
            two_pi * 1j * xp.sign(m) * (2 * h - inv_abs_m),
        )

    if power == 0:
        return j0
    if power == 1:
        return j1

    j_nm2 = j0
    j_nm1 = j1
    i_nm2 = cot_power_fourier_integral_coefficients(
        n_harmonics, 0, xp=xp, device=device, dtype=dtype
    )
    i_nm1 = cot_power_fourier_integral_coefficients(
        n_harmonics, 1, xp=xp, device=device, dtype=dtype
    )

    for k in range(2, power + 1):
        # Compute I_{m,k} in sync for the inhomogeneous term of J.
        i_n = (2j * m) / (k - 1) * i_nm1 - i_nm2
        j_n = (2j * m) / (k - 1) * j_nm1 - j_nm2 + (2 / (k - 1)) * i_n
        i_nm2, i_nm1 = i_nm1, i_n
        j_nm2, j_nm1 = j_nm1, j_n
    return j_nm1
