"""Utilities for iterating and analyzing the logistic map.

This module provides helpers for exploring the logistic map defined by
``f_r(x) = r * x * (1 - x)`` for control parameter ``r`` in ``[0, 4]`` and
state ``x`` in ``[0, 1]``. Functions are provided to iterate orbits, discard
transients, compute derivatives, locate fixed points, and estimate Lyapunov
exponents across a sweep of ``r`` values.
"""
from __future__ import annotations

import math

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


def logistic_map(x: float, r: float) -> float:
    """Compute a single iteration of the logistic map.

    Parameters
    ----------
    x:
        Current state of the system. Typical analyses consider ``x`` in the
        unit interval ``[0, 1]``.
    r:
        Control parameter of the logistic map. Dynamical phenomena of interest
        generally occur for ``r`` in ``[0, 4]``.

    Returns
    -------
    float
        The next value ``x_{n+1} = r * x_n * (1 - x_n)``.
    """

    return r * x * (1.0 - x)


def logistic_map_derivative(x: float, r: float) -> float:
    """Compute the derivative ``f'(x)`` of the logistic map.

    Parameters
    ----------
    x:
        Point at which to evaluate the derivative.
    r:
        Control parameter of the logistic map.

    Returns
    -------
    float
        The derivative ``f'(x) = r * (1 - 2 * x)``.
    """

    return r * (1.0 - 2.0 * x)


@dataclass(frozen=True)
class FixedPoint:
    """Representation of a logistic map fixed point and its stability.

    Attributes
    ----------
    x:
        Fixed point value satisfying ``f_r(x) = x``.
    derivative:
        Value of ``f'(x)`` evaluated at the fixed point.
    stable:
        ``True`` when ``|f'(x)| < 1`` indicating linear stability.
    """

    x: float
    derivative: float
    stable: bool


def iterate_logistic_map(x0: float, r: float, steps: int) -> list[float]:
    """Iterate the logistic map for a fixed number of steps.

    Parameters
    ----------
    x0:
        Initial condition ``x_0`` in ``[0, 1]``.
    r:
        Control parameter ``r`` in ``[0, 4]``.
    steps:
        Number of iterations to perform. Must be non-negative.

    Returns
    -------
    list of float
        Sequence ``[x_0, x_1, ..., x_steps]`` produced by repeated iteration of
        the logistic map.
    """

    if steps < 0:
        raise ValueError("steps must be non-negative")

    orbit: list[float] = [x0]
    x = x0
    for _ in range(steps):
        x = logistic_map(x, r)
        orbit.append(x)
    return orbit


def logistic_orbit(x0: float, r: float, length: int, discard: int = 100) -> list[float]:
    """Compute a long-run orbit after discarding transient iterations.

    Parameters
    ----------
    x0:
        Initial condition ``x_0`` in ``[0, 1]``.
    r:
        Control parameter ``r`` in ``[0, 4]``.
    length:
        Number of post-transient points to return. Must be non-negative.
    discard:
        Number of initial transient iterations to omit. Must be non-negative.

    Returns
    -------
    list of float
        Sequence of length ``length`` containing the orbit after discarding the
        first ``discard`` transient iterations.
    """

    if length < 0 or discard < 0:
        raise ValueError("length and discard must be non-negative")

    x = x0
    for _ in range(discard):
        x = logistic_map(x, r)

    orbit: list[float] = []
    for _ in range(length):
        x = logistic_map(x, r)
        orbit.append(x)
    return orbit


def fixed_points(r: float) -> tuple[FixedPoint, ...]:
    """Return analytically known fixed points and their stability.

    Parameters
    ----------
    r:
        Control parameter ``r`` in ``[0, 4]`` (excluding ``r = 0`` for the
        non-trivial fixed point).

    Returns
    -------
    tuple of :class:`FixedPoint`
        Analytic fixed points ``x = 0`` and ``x = 1 - 1 / r`` (when defined)
        together with derivative values and stability classification based on
        ``|f'(x)| < 1``.
    """

    points = []

    # Trivial fixed point x = 0 for all r.
    derivative_0 = logistic_map_derivative(0.0, r)
    points.append(FixedPoint(x=0.0, derivative=derivative_0, stable=abs(derivative_0) < 1))

    if r != 0.0:
        x_non_trivial = 1.0 - 1.0 / r
        derivative_non_trivial = logistic_map_derivative(x_non_trivial, r)
        points.append(
            FixedPoint(
                x=x_non_trivial,
                derivative=derivative_non_trivial,
                stable=abs(derivative_non_trivial) < 1,
            )
        )

    return tuple(points)


def fixed_points_numeric(
    r: float,
    guesses: Sequence[float],
    *,
    tol: float = 1e-12,
    max_iter: int = 1000,
) -> tuple[FixedPoint, ...]:
    """Approximate fixed points using Newton's method.

    Parameters
    ----------
    r:
        Control parameter ``r`` in ``[0, 4]``.
    guesses:
        Sequence of initial guesses for Newton's method. Distinct guesses can
        recover multiple fixed points.
    tol:
        Convergence tolerance on successive iterates. Defaults to ``1e-12``.
    max_iter:
        Maximum number of Newton iterations to perform per guess. Defaults to
        ``1000``.

    Returns
    -------
    tuple of :class:`FixedPoint`
        Approximate fixed points found for the supplied guesses. Duplicates
        (within ``tol``) are filtered out. Each fixed point is annotated with
        derivative and stability as in :func:`fixed_points`.
    """

    def newton_step(x: float) -> float:
        fx = logistic_map(x, r) - x
        dfx = logistic_map_derivative(x, r) - 1.0
        if dfx == 0:
            raise ZeroDivisionError("Derivative vanished during Newton iteration")
        return x - fx / dfx

    roots: list[FixedPoint] = []
    for guess in guesses:
        x = guess
        for _ in range(max_iter):
            x_next = newton_step(x)
            if abs(x_next - x) < tol:
                derivative = logistic_map_derivative(x_next, r)
                candidate = FixedPoint(x=x_next, derivative=derivative, stable=abs(derivative) < 1)
                if not any(abs(candidate.x - root.x) < tol for root in roots):
                    roots.append(candidate)
                break
            x = x_next
        else:
            raise RuntimeError("Newton iteration failed to converge")

    return tuple(roots)


def lyapunov_exponent(x0: float, r: float, steps: int, discard: int = 100) -> float:
    """Estimate the Lyapunov exponent for a single ``r`` value.

    Parameters
    ----------
    x0:
        Initial condition ``x_0`` in ``[0, 1]``.
    r:
        Control parameter ``r`` in ``[0, 4]``.
    steps:
        Number of post-transient iterations to include in the average. Must be
        positive for a meaningful estimate.
    discard:
        Number of initial iterations to discard as transients. Must be
        non-negative.

    Returns
    -------
    float
        Estimated Lyapunov exponent computed as the average of
        ``log|f'(x_n)|`` over the long-run orbit.
    """

    if steps <= 0:
        raise ValueError("steps must be positive")
    if discard < 0:
        raise ValueError("discard must be non-negative")

    x = x0
    for _ in range(discard):
        x = logistic_map(x, r)

    total = 0.0
    for _ in range(steps):
        derivative = logistic_map_derivative(x, r)
        if derivative == 0.0:
            # Avoid log(0) by treating derivative approaching zero as a large
            # negative contribution.
            return float("-inf")
        total += math.log(abs(derivative))
        x = logistic_map(x, r)
    return total / steps


def lyapunov_exponent_sweep(
    r_values: Iterable[float],
    x0: float,
    steps: int,
    discard: int = 100,
) -> list[tuple[float, float]]:
    """Estimate Lyapunov exponents for a collection of ``r`` values.

    Parameters
    ----------
    r_values:
        Iterable of control parameter values for which to estimate Lyapunov
        exponents.
    x0:
        Shared initial condition ``x_0`` used for each ``r``.
    steps:
        Number of post-transient iterations used in each exponent estimate.
    discard:
        Number of iterations to discard for transients in each computation.

    Returns
    -------
    list of tuple (float, float)
        Pairs ``(r, lambda)`` where ``lambda`` is the estimated Lyapunov
        exponent for parameter ``r``. The result is suitable for plotting
        bifurcation diagrams or Lyapunov exponent curves.
    """

    results: list[tuple[float, float]] = []
    for r in r_values:
        exponent = lyapunov_exponent(x0=x0, r=r, steps=steps, discard=discard)
        results.append((r, exponent))
    return results
