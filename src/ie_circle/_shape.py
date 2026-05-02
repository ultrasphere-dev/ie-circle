from collections.abc import Callable
from typing import Any, Protocol

import attrs
import sympy
from array_api._2024_12 import Array
from array_api_compat import array_namespace


class Shape(Protocol):
    def x(self, t: Array, /) -> Array: ...

    def dx(self, t: Array, /) -> Array: ...

    def ddx(self, t: Array, /) -> Array: ...


def jacobian(shape: Shape, t: Array, /) -> Array:
    xp = array_namespace(t)
    return xp.sqrt(xp.sum(shape.dx(t) ** 2, axis=-1))


@attrs.define(frozen=True)
class CircleShape(Shape):
    """Circle of radius ``rho`` centered at the origin."""

    rho: float

    def x(self, t: Array, /) -> Array:
        xp = array_namespace(t)
        return xp.stack([self.rho * xp.cos(t), self.rho * xp.sin(t)], axis=-1)

    def dx(self, t: Array, /) -> Array:
        xp = array_namespace(t)
        return xp.stack([-self.rho * xp.sin(t), self.rho * xp.cos(t)], axis=-1)

    def ddx(self, t: Array, /) -> Array:
        xp = array_namespace(t)
        return xp.stack([-self.rho * xp.cos(t), -self.rho * xp.sin(t)], axis=-1)


@attrs.define(frozen=True)
class KressShape(Shape):
    """
    Shape of x(t) = (cos(t) + 0.65 cos(2t) - 0.65, 1.5 sin(t)).

    References
    ----------
    Kress, R. (1991). Boundary integral equations in
    time-harmonic acoustic scattering.
    Mathematical and Computer Modelling, 15(3), 229--243.
    https://doi.org/10.1016/0895-7177(91)90068-I

    """

    def x(self, t: Array, /) -> Array:
        xp = array_namespace(t)
        return xp.stack(
            [xp.cos(t) + 0.65 * xp.cos(2 * t) - 0.65, 1.5 * xp.sin(t)],
            axis=-1,
        )

    def dx(self, t: Array, /) -> Array:
        xp = array_namespace(t)
        return xp.stack(
            [-xp.sin(t) - 0.65 * 2 * xp.sin(2 * t), 1.5 * xp.cos(t)],
            axis=-1,
        )

    def ddx(self, t: Array, /) -> Array:
        xp = array_namespace(t)
        return xp.stack(
            [-xp.cos(t) - 0.65 * 2 * 2 * xp.cos(2 * t), -1.5 * xp.sin(t)],
            axis=-1,
        )


@attrs.define(frozen=True)
class SympyShape(Shape):
    """
    Shape defined by SymPy expressions.

    The derivatives are computed at initialization.
    """

    x_expr: sympy.Expr
    y_expr: sympy.Expr
    t_symbol: sympy.Symbol = attrs.field()

    _x_func: Callable[..., Any] = attrs.field(init=False, repr=False)
    _dx_func: Callable[..., Any] = attrs.field(init=False, repr=False)
    _ddx_func: Callable[..., Any] = attrs.field(init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        # Calculate derivatives
        dx_expr_x = sympy.diff(self.x_expr, self.t_symbol)
        dx_expr_y = sympy.diff(self.y_expr, self.t_symbol)
        ddx_expr_x = sympy.diff(dx_expr_x, self.t_symbol)
        ddx_expr_y = sympy.diff(dx_expr_y, self.t_symbol)

        # Lambdify
        object.__setattr__(
            self,
            "_x_func",
            sympy.lambdify(self.t_symbol, [self.x_expr, self.y_expr], modules="numpy"),
        )
        object.__setattr__(
            self,
            "_dx_func",
            sympy.lambdify(self.t_symbol, [dx_expr_x, dx_expr_y], modules="numpy"),
        )
        object.__setattr__(
            self,
            "_ddx_func",
            sympy.lambdify(self.t_symbol, [ddx_expr_x, ddx_expr_y], modules="numpy"),
        )

    def _eval(self, func: Callable[..., Any], t: Array) -> Array:
        xp = array_namespace(t)
        # func returns a list of results (usually numpy arrays or scalars)
        res = func(t)
        # Convert to array and broadcast if necessary
        v0 = xp.asarray(res[0])
        v1 = xp.asarray(res[1])
        if v0.ndim == 0:
            v0 = xp.broadcast_to(v0, t.shape)
        if v1.ndim == 0:
            v1 = xp.broadcast_to(v1, t.shape)
        return xp.stack([v0, v1], axis=-1)

    def x(self, t: Array, /) -> Array:
        return self._eval(self._x_func, t)

    def dx(self, t: Array, /) -> Array:
        return self._eval(self._dx_func, t)

    def ddx(self, t: Array, /) -> Array:
        return self._eval(self._ddx_func, t)
