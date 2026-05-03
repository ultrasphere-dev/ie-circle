import math
from typing import Any

import pytest
from array_api._2024_12 import ArrayNamespaceFull

from ie_circle._example import example_13_19, example_13_19_answer


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

    eval_points = xp.random.uniform(0, 2 * math.pi, size=(10,), device=device, dtype=dtype)

    expected_phi = example_13_19_answer(eval_points)

    phi_computed = interpolant(eval_points)

    error = xp.max(xp.abs(phi_computed - expected_phi))
    assert float(error) < 1e-6
