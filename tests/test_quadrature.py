from __future__ import annotations

from typing import Any

import pytest

from ie_circle._quadrature import (
    garrick_wittich_quadrature,
    kussmaul_martensen_kress_quadrature,
    trapezoidal_quadrature,
)


@pytest.mark.parametrize("t_start_factor", [0, 0.5])
@pytest.mark.parametrize("f_case", ["one", "exp1", "combo"])
def test_kussmaul_martensen_kress_quadrature_exactness(
    xp: Any, device: Any, dtype: Any, t_start_factor: float, f_case: str
) -> None:
    n = 6
    t, w = kussmaul_martensen_kress_quadrature(
        n, t_start_factor=t_start_factor, xp=xp, device=device, dtype=dtype
    )
    two_pi = xp.pi * 2

    if f_case == "one":
        f = xp.ones_like(t)
        expected = 0
    elif f_case == "exp1":
        f = xp.exp(1j * t)
        expected = -two_pi
        # From docs/quadrature.typ:
        # ∫ log(4 sin^2(t/2)) e^{i m t} dt = -2π/|m|.
    else:
        f = xp.ones_like(t) + xp.exp(1j * 3 * t) + xp.exp(-1j * 4 * t)
        expected = -(two_pi / 3) - (two_pi / 4)

    approx = xp.sum(w * f)
    assert xp.abs(approx - expected) < 1e-10


@pytest.mark.parametrize("t_start_factor", [0, 0.5])
@pytest.mark.parametrize("f_case", ["one", "exp1", "combo"])
def test_garrick_wittich_quadrature_exactness(
    xp: Any, device: Any, dtype: Any, t_start_factor: float, f_case: str
) -> None:
    n = 6
    t, w = garrick_wittich_quadrature(
        n, t_start_factor=t_start_factor, xp=xp, device=device, dtype=dtype
    )
    two_pi = xp.pi * 2

    if f_case == "one":
        f = xp.ones_like(t)
        expected = 0
    elif f_case == "exp1":
        f = xp.exp(1j * t)
        expected = two_pi * 1j
        # From docs/quadrature.typ:
        # p.v.∫ cot(t/2) e^{i m t} dt = 2π i sgn(m).
    else:
        f = xp.ones_like(t) + xp.exp(1j * 3 * t) + xp.exp(-1j * 4 * t)
        expected = 0

    approx = xp.sum(w * f)
    assert xp.abs(approx - expected) < 1e-10


@pytest.mark.parametrize("t_start_factor", [0, 0.5])
@pytest.mark.parametrize("f_case", ["one", "exp1", "combo"])
def test_trapezoidal_quadrature_exactness(
    xp: Any, device: Any, dtype: Any, t_start_factor: float, f_case: str
) -> None:
    n = 6
    t, w = trapezoidal_quadrature(
        n, t_start_factor=t_start_factor, xp=xp, device=device, dtype=dtype
    )
    two_pi = xp.pi * 2

    if f_case == "one":
        f = xp.ones_like(t)
        expected = two_pi  # Trapezoidal is exact on Fourier modes with |m|<n (here m=0).
    elif f_case == "exp1":
        f = xp.exp(1j * t)
        expected = 0
    else:
        f = xp.ones_like(t) + xp.exp(1j * 3 * t) + xp.exp(-1j * 4 * t)
        expected = two_pi  # Nonzero modes integrate to 0; only the constant term remains.

    approx = xp.sum(w * f)
    assert xp.abs(approx - expected) < 1e-10
