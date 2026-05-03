---
name: array-api
description: Conventions that MUST be followed when implementing array API compatible functions and their tests.
---

# Conventions for implementing array API compatible functions and their tests

- **Think (not plan) about the formulation and the way to implement mathematically beautifully a LOT beforehand, ask questions to the user or internet, then finally write code.** Besides being a intelligent assistant, you have to admit that you are **extremely bad at numerical** programming due to the lack of such dataset (clean pure-Python numerical code is very rare). Think about the math formulation, the way to implement it, then at the **very final stage**, write code.

## Function implementation

- This repository is about numerical analysis. You need to make very sure about the math formulation before implementing the code.
- If documentation about the formulation is provided, always read it first, to very make sure the functions are implemented mathematically correctly and beautifully.
- All methods should be array API compatible.
  - **Function: name** Name functions without `compute_`, `calculate_` suffix (e.g. `prime_number(i: int, /)`, not `compute_prime_number`, `p_i`).
  - **Arguments: name**: The function name variable names in the code should not the same as the variable name math formulation (e.g. `n`) but very readable (e.g. `n_iterations`). In the variable description in the docstring, the corresponding variable name in the math formulation should be mentioned:

    ```text
    n_iterations : Array
                The number of iterations $n$ of shape (...,).
    ```

    Try not to add the word "corresponding to" because it is redundant.

  - **Output shape depending on integer output**: If output shape depends on the input argument, it should match the input integer when possible. For example, to compute $0, 1, ...$, set argument name to `n_*_end` (`n_end` if it is obvious) and return `0, 1, ..., n_*_end - 1`, so that the output shape is `(n_*_end,)` and it is intuitive, matches `range(n_end)` convention and `np.arange(n_end)` syntax.
  - **GUFunc compatibility**: If an array is passed to function, the function should be GUFunc-compatible, i.e. the function should only remove / append extra dimensions from the LAST dimensions of the input / output arrays, e.g. `(..., a, b) -> (..., c, d, e)`.
  - **Arguments: arrays** If array is passed to function, use `array_api.latest.Array` as the array type hint, and use `array_api_compat.array_namespace()` to get the array API namespace `xp`. `array_api_compat.array_namespace()` should be ideally called with all input arrays / evaluated function as arguments IF POSSIBLE, e.g. `xp = array_api_compat.array_namespace(x, y, z)` to validate the input arrays are on the same array API and device. The arguments may also contain None, Python scalars, useful when arguments are optional.
  - **Arguments: no arrays -> special kw-only arguments** If NO array is passed to function, add `xp`, `device`, `dtype` as an required keyword-only argument with type hint `array_api.latest.ArrayNamespace`, `Any`, `Any` respectively. Do not add these arguments if an array is passed to function.
  - **Arguments: function -> GUFunc-compatible** If the function needs function arguments, assume that to be also GUFunc-compatible. The argument description should end with `of (..., a, b) -> (..., c, d, e)`. Don't add any word in the following sentence: "GUFunc-compatible vectorized function from array to array", just explain the mathematical meaning of the function and its shape convention.
  - **Docstring: function output-dependent shape**: When the shape is variable (depending on function etc.) one can do `(..., ...(f))` where `f` implies the function but may replaced with something more suitable. `(..., *something)` is also possible but less preferred, yet sometimes it might be more suitable.
  - **Docstring**: The docstring should be Numpydoc style.

    ```python
    from array_api.latest import Array, ArrayNamespace

    def func(x: Array, power: Array) -> Array:
        """
        Computes $x^p$.

        Extended description of function.

        Parameters
        ----------
        x : Array
            The base array of shape (...,).
        y : Array
            The power to which $x$ is raised $p$ of shape (...,).

        Returns
        -------
        Array
            $x^p$ of shape (...,).
        """
    ```

  - **Docstring: shape**: The docstring should mention the shape of the input / output arrays and the argument description should end with `of shape (..., a, b)`.
  - **Importing Numpy allowed only for constants**: Never import `numpy` directly, unless for constants like `np.pi` for context when `xp` is not available.
  - **Type promotion**: Understand Type promotion rules, i.e. float64 + complex64 -> complex128. Mixed integer and floating-point type promotion rules are not specified, but we assume that for every floating (including complex) dtype x, x + (int type) -> x.
  - **Type promotion: no wrapping Python scalars**: Avoid wrapping `int` arrays, Python scalars with `xp.asarray()` but use them directly (because it is redundant). The exception is when you need to divide int by int (in this case you only need to wrap one of them).
  - **Type promotion: avoid conversion as much as possible and make it inline**:Avoid creating variables for `int` version, `float` version, `complex` version of the same array as much as possible.
  - **Type promotion**: The type can be converted by `xp.astype(x, dtype, /)`.
  - **Avoid float when possible**: Avoid expressing integer as float. `1` instead of `1.0` whenever possible.
  - **Type promotion: Scipy -> input cpu, output asarray**: As an exception, if Scipy functions are needed (e.g. `scipy.special.yv`), do `xp.asarray(yv(xp.asarray(x, device="cpu")), device=x.device, dtype=x.dtype)`. (Do not specify dtype in the inner `asarray`). Note that every array has property `device` (including NumPy >= 2.0), you don't need `getattr`.
  - **Expand dimensions using []**: When expanding dimensions, prefer something like `x[(...,) + (None,) * n + (slice(None),) * m]` or `x[(slice(None),) * m + (None,) * n + (...,)]` over `xp.reshape()`. Avoid creating "expanded version` and "non-expanded version" of the same array, unless both of them are frequently used.
  - **Type promotion: no complex -> float (terrible undetectable bug)** Do not `asarray(x, dtype=dtype)` if `x` is complex dtype and `dtype` is float. This sometimes happens when `dtype` is an variable (trying to make function that is any-float compatible e.g. float32 -> float32 / complex64, complex64 -> complex64, float64 -> float64 / complex128, complex128 -> complex128). It will be equivalent to `xp.real(x)` which may cause severe numerical issues. Instead do `xp.asarray(x, dtype=xp.promote_types(x.dtype, xp.promote_types(dtype, complex)))`.

## Tests

- Tests should be also array API compatible.
  - In `tests/conftest.py`, there are fixtures named `xp: ArrayNamespace`, `device: Any`, `dtype: Any`. Any test function must use these fixtures as arguments, and create arrays (i.e. `zeros()`) within the test function.
  - If there is an array passed as fixture / parameter to the test function. wrap it with `xp.asarray(..., device=device, dtype=dtype)` at the beginning of the test function. If it is a scalar, never wrap it but use it directly.
  - Parameterize tests using `pytest.mark.parametrize`.
  - Do not try to read the contents of `tests/conftest.py`.
  - To run python commands, use `uv run python`, `uv run pytest`, etc. Never run `python` directly. You may run `uv run pytest` on your own.
