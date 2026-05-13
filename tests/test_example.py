import math
from typing import Any

import array_api_extra as xpx
import pytest
from array_api._2024_12 import ArrayNamespaceFull

from ie_circle import trapezoidal_quadrature
from ie_circle._example import (
    example_13_19,
    example_13_19_answer,
    example_simple,
    example_simple_answer,
)


@pytest.mark.parametrize("a", [3.0, 5.0])
@pytest.mark.parametrize("b", [1.0, 2.0])
@pytest.mark.parametrize("n", [32])
def test_example_13_19_different_t_start(
    a: float,
    b: float,
    n: int,
    xp: ArrayNamespaceFull,
    device: Any,
    dtype: Any,
) -> None:
    interpolant_0 = example_13_19(
        a,
        b,
        n,
        xp=xp,
        device=device,
        dtype=dtype,
        t_start_factor_quadrature=0.0,
        t_start_factor=0.0,
    )
    interpolant_03 = example_13_19(
        a,
        b,
        n,
        xp=xp,
        device=device,
        dtype=dtype,
        t_start_factor_quadrature=0.3,
        t_start_factor=0.3,
    )
    assert xp.all(
        xpx.isclose(
            interpolant_0(
                trapezoidal_quadrature(n, xp=xp, device=device, dtype=dtype, t_start_factor=0.3)[0][
                    :3
                ]
            ),
            interpolant_03.sol[:3],
        )
    )


@pytest.mark.parametrize("a", [3.0, 5.0])
@pytest.mark.parametrize("b", [1.0, 2.0])
@pytest.mark.parametrize("n", [32, 64])
@pytest.mark.parametrize("t_start_factor_quadrature", [0.0, 0.3, 0.6])
@pytest.mark.parametrize("t_start_factor", [0.0, 0.3, 0.6])
def test_example_13_19(
    a: float,
    b: float,
    n: int,
    xp: ArrayNamespaceFull,
    device: Any,
    dtype: Any,
    t_start_factor_quadrature: float | None,
    t_start_factor: float | None,
) -> None:
    interpolant = example_13_19(
        a,
        b,
        n,
        xp=xp,
        device=device,
        dtype=dtype,
        t_start_factor_quadrature=t_start_factor_quadrature,
        t_start_factor=t_start_factor,
    )

    eval_points = xp.random.random_uniform(0, 2 * math.pi, (10,), device=device, dtype=dtype)

    expected = xp.astype(example_13_19_answer(eval_points), xp.result_type(dtype, 1j))
    actual = interpolant(eval_points)
    assert xp.all(xpx.isclose(actual, expected))


@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("t_start_factor_quadrature", [0.0, 0.3])
@pytest.mark.parametrize("t_start_factor", [0.0, 0.3])
def test_example_simple(
    n: int,
    xp: ArrayNamespaceFull,
    device: Any,
    dtype: Any,
    t_start_factor_quadrature: float | None,
    t_start_factor: float | None,
) -> None:
    interpolant = example_simple(
        n,
        xp=xp,
        device=device,
        dtype=dtype,
        t_start_factor_quadrature=t_start_factor_quadrature,
        t_start_factor=t_start_factor,
    )
    eval_points = xp.linspace(0.0, 2 * math.pi, 10, endpoint=False, device=device, dtype=dtype)
    expected = xp.astype(example_simple_answer(eval_points), xp.result_type(dtype, 1j))
    actual = interpolant(eval_points)
    assert xp.all(xpx.isclose(actual, expected))
