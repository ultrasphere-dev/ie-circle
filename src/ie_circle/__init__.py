__version__ = "0.2.1"
from ._bie import (
    ArrayFunction,
    KernelFunction,
    Kernels,
    NystromInterpolant,
    QuadratureType,
    nystrom,
    nystrom_lhs,
    nystrom_rhs,
)
from ._example import example_13_19, example_13_19_answer
from ._fourier_integral import (
    cot_power_fourier_integral_coefficients,
    harmonic_number,
    log_cot_power_fourier_integral_coefficients,
)
from ._quadrature import (
    PowerQuadratureRule,
    QuadratureRule,
    cot_power_quadrature,
    fourier_coeff_to_quadrature,
    garrick_wittich_quadrature,
    kussmaul_martensen_kress_quadrature,
    log_cot_power_quadrature,
    shift_quadrature_singularity,
    trapezoidal_quadrature,
)
from ._shape import CircleShape, KressShape, Shape, SympyShape

__all__ = [
    "ArrayFunction",
    "CircleShape",
    "KernelFunction",
    "Kernels",
    "KressShape",
    "NystromInterpolant",
    "PowerQuadratureRule",
    "QuadratureRule",
    "QuadratureType",
    "Shape",
    "SympyShape",
    "cot_power_fourier_integral_coefficients",
    "cot_power_quadrature",
    "example_13_19",
    "example_13_19_answer",
    "fourier_coeff_to_quadrature",
    "garrick_wittich_quadrature",
    "harmonic_number",
    "kussmaul_martensen_kress_quadrature",
    "log_cot_power_fourier_integral_coefficients",
    "log_cot_power_fourier_integral_coefficients",
    "log_cot_power_quadrature",
    "nystrom",
    "nystrom_lhs",
    "nystrom_rhs",
    "shift_quadrature_singularity",
    "trapezoidal_quadrature",
]
