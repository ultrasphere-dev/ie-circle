import numpy as np
import pytest
import sympy

from ie_circle._shape import CircleShape, KressShape, SympyShape


@pytest.mark.parametrize(
    "shape_cls, sympy_args, shape_args",
    [
        (
            CircleShape,
            lambda t: (1.5 * sympy.cos(t), 1.5 * sympy.sin(t)),
            {"rho": 1.5},
        ),
        (
            KressShape,
            lambda t: (
                sympy.cos(t) + 0.65 * sympy.cos(2 * t) - 0.65,
                1.5 * sympy.sin(t),
            ),
            {},
        ),
    ],
)
def test_sympy_shape_consistency(shape_cls, sympy_args, shape_args):
    # Setup SympyShape
    t_sym = sympy.Symbol("t")
    x_expr, y_expr = sympy_args(t_sym)
    sympy_shape = SympyShape(x_expr, y_expr, t_sym)

    # Setup Original Shape
    original_shape = shape_cls(**shape_args)

    # Test points
    t_vals = np.linspace(0, 2 * np.pi, 20)

    # Compare x
    # pytest.approx handles numpy arrays
    assert sympy_shape.x(t_vals) == pytest.approx(original_shape.x(t_vals))

    # Compare dx
    assert sympy_shape.dx(t_vals) == pytest.approx(original_shape.dx(t_vals))

    # Compare ddx
    assert sympy_shape.ddx(t_vals) == pytest.approx(original_shape.ddx(t_vals))
