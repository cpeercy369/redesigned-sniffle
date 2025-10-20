"""Cobweb model with logistic demand and linear supply.

This module provides a minimal deterministic cobweb model in which firms form
adaptive expectations about future prices. Demand is modeled as an S-shaped
logistic curve and supply is linear in the expected price. The core parameters
have direct economic interpretations:

* ``max_demand`` is the quantity demanded when prices approach zero.
* ``demand_midpoint`` is the price at which demand falls to half of its maximum.
* ``demand_slope`` controls how quickly consumers reduce purchases as prices
  rise; higher values imply a steeper demand curve.
* ``supply_intercept`` is the quantity supplied if producers expect the price to
  be zero (e.g., reflecting fixed output commitments).
* ``supply_slope`` captures the responsiveness of supply to expected prices.

Agents update price expectations with speed :math:`\alpha \in [0, 1]` according
to the adaptive expectations rule

.. math::

   p_{t+1}^e = p_t^e + \alpha (p_t - p_t^e),

where ``alpha`` is the expectations speed parameter supplied to the public API.
Producers choose output based on their expectation ``p_t^e`` and the market price
``p_t`` clears demand and supply. Market clearing and the expectation update
jointly determine a one-dimensional price map, which we expose alongside
utilities for simulation, fixed point analysis, stability classification, and
Lyapunov exponent estimation.

The implementation assumes that the logistic demand function is strictly
monotone decreasing, ensuring a unique market-clearing price for any admissible
quantity. No stochastic shocks are included, so all chaotic behavior arises
purely from deterministic expectation dynamics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class CobwebModel:
    """Deterministic cobweb model with logistic demand and linear supply.

    Parameters
    ----------
    max_demand:
        Maximum quantity demanded as prices approach zero. Must be positive.
    demand_midpoint:
        Price at which demand equals ``max_demand / 2``.
    demand_slope:
        Positive slope parameter controlling how sharply demand falls with
        price.
    supply_intercept:
        Quantity supplied when the expected price is zero.
    supply_slope:
        Positive slope parameter describing how supply responds to expected
        prices.
    """

    max_demand: float
    demand_midpoint: float
    demand_slope: float
    supply_intercept: float
    supply_slope: float

    def __post_init__(self) -> None:
        if self.max_demand <= 0:
            raise ValueError("max_demand must be positive")
        if self.demand_slope <= 0:
            raise ValueError("demand_slope must be positive")
        if self.supply_slope <= 0:
            raise ValueError("supply_slope must be positive")

    # ------------------------------------------------------------------
    # Demand and supply primitives
    # ------------------------------------------------------------------
    def demand(self, price: float) -> float:
        """Return quantity demanded at ``price`` using an S-shaped curve."""

        exponent = self.demand_slope * (price - self.demand_midpoint)
        return self.max_demand / (1.0 + math.exp(exponent))

    def supply(self, expected_price: float) -> float:
        """Return quantity supplied when the expected price is given."""

        return self.supply_intercept + self.supply_slope * expected_price

    def inverse_demand(self, quantity: float) -> float:
        """Return the price consistent with the given quantity demanded.

        The logistic functional form implies a closed-form inverse for admissible
        quantities in the open interval ``(0, max_demand)``.
        """

        if not 0 < quantity < self.max_demand:
            raise ValueError(
                "quantity must lie strictly between 0 and max_demand for a "
                "well-defined inverse demand"
            )

        ratio = self.max_demand / quantity - 1.0
        return self.demand_midpoint + math.log(ratio) / self.demand_slope

    # ------------------------------------------------------------------
    # Core price map
    # ------------------------------------------------------------------
    def expected_price_given_actual(self, price: float) -> float:
        """Recover the expectation that would justify the observed price."""

        demand = self.demand(price)
        return (demand - self.supply_intercept) / self.supply_slope

    def price_map(self, price: float, alpha: float) -> float:
        """Advance the model one period and return the new market price.

        Parameters
        ----------
        price:
            Observed market price at time ``t``.
        alpha:
            Expectations speed. ``alpha = 1`` corresponds to naive expectations
            (full adjustment), while ``alpha = 0`` implies static expectations.

        Returns
        -------
        float
            The market-clearing price at time ``t + 1``.
        """

        expected_price = self.expected_price_given_actual(price)
        updated_expectation = expected_price + alpha * (price - expected_price)
        quantity_next = self.supply(updated_expectation)
        return self.inverse_demand(quantity_next)

    # ------------------------------------------------------------------
    # Iteration utilities
    # ------------------------------------------------------------------
    def iterate(
        self,
        initial_price: float,
        alpha: float,
        steps: int,
        discard: int = 0,
    ) -> List[float]:
        """Iterate the price map, discarding an optional transient prefix."""

        if steps < 0 or discard < 0:
            raise ValueError("steps and discard must be non-negative")

        price = initial_price
        for _ in range(discard):
            price = self.price_map(price, alpha)

        trajectory: List[float] = []
        for _ in range(steps):
            price = self.price_map(price, alpha)
            trajectory.append(price)
        return trajectory

    def derivative(
        self, price: float, alpha: float, h: float = 1e-6
    ) -> float:
        """Estimate the derivative of the price map via central differences."""

        if h <= 0:
            raise ValueError("finite difference step h must be positive")

        price_plus = self.price_map(price + h, alpha)
        price_minus = self.price_map(price - h, alpha)
        return (price_plus - price_minus) / (2.0 * h)

    def solve_fixed_point(
        self,
        alpha: float,
        initial_guess: float,
        tol: float = 1e-10,
        max_iter: int = 100,
    ) -> float:
        """Find a price fixed point using Newton's method.

        The routine solves ``price_map(p, alpha) = p`` starting from
        ``initial_guess``. Convergence is declared when both the update and the
        residual fall below ``tol``.
        """

        if tol <= 0:
            raise ValueError("tol must be positive")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")

        price = initial_guess
        for _ in range(max_iter):
            residual = self.price_map(price, alpha) - price
            if abs(residual) < tol:
                return price

            derivative = self.derivative(price, alpha)
            slope = derivative - 1.0
            if abs(slope) < 1e-12:
                raise RuntimeError("derivative too close to unity for Newton step")

            update = residual / slope
            price_next = price - update
            if abs(price_next - price) < tol:
                return price_next
            price = price_next

        raise RuntimeError("Newton iteration failed to converge")

    def classify_stability(
        self,
        alpha: float,
        fixed_point: float,
        tol: float = 1e-8,
        h: float = 1e-6,
    ) -> str:
        """Classify a fixed point as stable, unstable, or neutral."""

        derivative = self.derivative(fixed_point, alpha, h)
        magnitude = abs(derivative)
        if abs(magnitude - 1.0) <= tol:
            return "neutral"
        return "stable" if magnitude < 1.0 else "unstable"

    # ------------------------------------------------------------------
    # Lyapunov exponent estimation
    # ------------------------------------------------------------------
    def lyapunov_exponent(
        self,
        alpha: float,
        initial_price: float,
        steps: int,
        discard: int = 100,
        h: float = 1e-6,
    ) -> float:
        """Estimate the Lyapunov exponent for a single ``alpha`` value."""

        if steps <= 0:
            raise ValueError("steps must be positive")
        if discard < 0:
            raise ValueError("discard must be non-negative")

        price = initial_price
        for _ in range(discard):
            price = self.price_map(price, alpha)

        accumulator = 0.0
        for _ in range(steps):
            derivative = abs(self.derivative(price, alpha, h))
            if derivative <= 0:
                # Treat zero derivatives as contributing large negative growth.
                return float("-inf")
            accumulator += math.log(derivative)
            price = self.price_map(price, alpha)

        return accumulator / steps

    def lyapunov_spectrum(
        self,
        alphas: Sequence[float],
        initial_price: float,
        steps: int,
        discard: int = 100,
        h: float = 1e-6,
    ) -> List[Tuple[float, float]]:
        """Estimate Lyapunov exponents across a range of ``alpha`` values."""

        return [
            (alpha, self.lyapunov_exponent(alpha, initial_price, steps, discard, h))
            for alpha in alphas
        ]

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def iterate_with_derivatives(
        self,
        initial_price: float,
        alpha: float,
        steps: int,
        discard: int = 0,
        h: float = 1e-6,
    ) -> List[Tuple[float, float]]:
        """Return pairs of prices and local derivatives after a transient."""

        price = initial_price
        for _ in range(discard):
            price = self.price_map(price, alpha)

        trajectory: List[Tuple[float, float]] = []
        for _ in range(steps):
            price = self.price_map(price, alpha)
            derivative = self.derivative(price, alpha, h)
            trajectory.append((price, derivative))
        return trajectory


def lyapunov_grid(
    model: CobwebModel,
    alphas: Iterable[float],
    initial_price: float,
    steps: int,
    discard: int = 100,
    h: float = 1e-6,
) -> List[Tuple[float, float]]:
    """Convenience wrapper mirroring logistic-map tooling.

    Parameters mirror :meth:`CobwebModel.lyapunov_spectrum` but accept any
    iterable of ``alpha`` values, allowing easy integration with NumPy linspace
    grids without materializing them in advance.
    """

    return [
        (alpha, model.lyapunov_exponent(alpha, initial_price, steps, discard, h))
        for alpha in alphas
    ]
