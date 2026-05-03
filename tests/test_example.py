from typing import Any

import pytest
from array_api._2024_12 import ArrayNamespaceFull

from ie_circle._example import example_13_19


@pytest.mark.parametrize("a", [3.0, 5.0])
@pytest.mark.parametrize("b", [1.0, 2.0])
@pytest.mark.parametrize("n_nodes", [32, 64])
def test_example_13_19(
    a: float,
    b: float,
    n_nodes: int,
    xp: ArrayNamespaceFull,
    device: Any,
    dtype: Any,
) -> None:
    interpolant = example_13_19(a, b, n_nodes, xp=xp, device=device, dtype=dtype)

    # test at random evaluation points
    # (since we are using array API, we specify fixed "randomly chosen" points or use linear space)
    eval_points = xp.asarray([0.1, 1.2, 2.3, 3.4, 4.5, 5.6], dtype=dtype, device=device)

    # \phi(t) = 1 - e^{\cos t} \cos(\sin t)
    expected_phi = 1.0 - xp.exp(xp.cos(eval_points)) * xp.cos(xp.sin(eval_points))

    phi_computed = interpolant(eval_points)

    # Nystrom method should converge very fast for analytic functions
    # For large n_nodes it should be exact to machine precision (or close)
    error = xp.max(xp.abs(phi_computed - expected_phi))
    assert float(error) < 1e-6
