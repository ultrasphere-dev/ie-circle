from typing import Any

import array_api_extra as xpx
import pytest
from array_api.latest import ArrayNamespaceFull

from ie_circle._bie import trapezoidal_basis
from ie_circle._quadrature import trapezoidal_quadrature


@pytest.mark.parametrize("n", [32, 64])
@pytest.mark.parametrize("t_start_factor", [0.0, 0.3, 0.6])
def test_trapezoidal_basis(
    n: int,
    t_start_factor: float,
    xp: ArrayNamespaceFull,
    device: Any,
    dtype: Any,
) -> None:
    x = trapezoidal_quadrature(n, xp=xp, device=device, dtype=dtype, t_start_factor=t_start_factor)[
        0
    ]
    actual = trapezoidal_basis(
        x, n=n, xp=xp, device=device, dtype=dtype, t_start_factor=t_start_factor
    )
    expected = xp.eye(2 * n - 1, device=device, dtype=xp.result_type(dtype, 1j))
    assert actual.shape == (2 * n - 1, 2 * n - 1)
    assert xp.all(xpx.isclose(actual, expected))
