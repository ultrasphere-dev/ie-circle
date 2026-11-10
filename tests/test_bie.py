import math
from typing import Any

import pytest
from array_api.latest import Array, ArrayNamespace

from ie_circle import nystrom


@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
@pytest.mark.parametrize("t_start_factor", [0, 0.25])
def test_nystrom_batched_coefficient(
    xp: ArrayNamespace, device: Any, dtype: Any, batch_shape: tuple[int, ...], t_start_factor: float
) -> None:
    n_components = 2
    coefficients = xp.reshape(
        xp.arange(1, math.prod(batch_shape) * n_components + 1, dtype=dtype, device=device),
        (*batch_shape, n_components),
    )

    def a(t: Array, /) -> Array:
        return coefficients + 1 + xp.cos(t)[(...,) + (None,) * (len(batch_shape) + 1)] / 4

    def rhs(t: Array, /) -> Array:
        values = 1 + xp.cos(t) / 3 + xp.sin(2 * t) / 5
        return a(t) * values[(...,) + (None,) * (len(batch_shape) + 1)]

    solution = nystrom(
        a, {}, rhs, n=5, xp=xp, device=device, dtype=dtype, t_start_factor=t_start_factor
    )
    points = xp.asarray([0.1, 1, 2.3], dtype=dtype, device=device)
    values = solution(points)
    expected = 1 + xp.cos(points) / 3 + xp.sin(2 * points) / 5
    expected = expected[(...,) + (None,) * (len(batch_shape) + 1)]
    assert values.shape == (3, *batch_shape, n_components)
    assert bool(xp.all(xp.abs(values - expected) < 100 * xp.finfo(dtype).eps))
