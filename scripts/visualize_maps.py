"""Visualization utilities for one-dimensional iterated maps.

This module can be used as a script to generate bifurcation diagrams,
Lyapunov exponent plots, and cobweb diagrams for several canonical maps.

Example usage
-------------
$ python -m scripts.visualize_maps --output-dir outputs
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


MapFunction = Callable[[float, float], float]
DerivativeFunction = Callable[[float, float], float]


@dataclass(frozen=True)
class MapModel:
    """Container describing a one-dimensional map and plotting defaults."""

    name: str
    map_function: MapFunction
    derivative_function: DerivativeFunction
    parameter_range: Tuple[float, float]
    parameter_label: str
    initial_condition: float
    domain: Tuple[float, float]
    cobweb_alphas: Sequence[float]
    cobweb_iterations: int
    description: str


# ---------------------------------------------------------------------------
# Map definitions
# ---------------------------------------------------------------------------

def logistic_map(alpha: float, x: float) -> float:
    """Logistic map x_{n+1} = α x_n (1 - x_n)."""

    return alpha * x * (1.0 - x)


def logistic_map_derivative(alpha: float, x: float) -> float:
    """Derivative of the logistic map with respect to x."""

    return alpha * (1.0 - 2.0 * x)


def quadratic_map(alpha: float, x: float) -> float:
    """Quadratic map (Feigenbaum map) x_{n+1} = 1 - α x_n^2."""

    return 1.0 - alpha * x * x


def quadratic_map_derivative(alpha: float, x: float) -> float:
    """Derivative of the quadratic map with respect to x."""

    return -2.0 * alpha * x


MODELS: Tuple[MapModel, ...] = (
    MapModel(
        name="logistic",
        map_function=logistic_map,
        derivative_function=logistic_map_derivative,
        parameter_range=(2.5, 4.0),
        parameter_label=r"$\\alpha$",
        initial_condition=0.5,
        domain=(0.0, 1.0),
        cobweb_alphas=(2.8, 3.3, 3.9),
        cobweb_iterations=50,
        description="Logistic map $x_{n+1} = \\alpha x_n (1-x_n)$",
    ),
    MapModel(
        name="quadratic",
        map_function=quadratic_map,
        derivative_function=quadratic_map_derivative,
        parameter_range=(1.0, 2.0),
        parameter_label=r"$\\alpha$",
        initial_condition=0.0,
        domain=(-1.2, 1.2),
        cobweb_alphas=(1.2, 1.401155, 1.8),
        cobweb_iterations=80,
        description="Quadratic map $x_{n+1} = 1 - \\alpha x_n^2$",
    ),
)


# ---------------------------------------------------------------------------
# Core numerical routines
# ---------------------------------------------------------------------------

def iterate_map(
    map_function: MapFunction,
    alpha: float,
    initial_value: float,
    steps: int,
) -> np.ndarray:
    """Iterate ``map_function`` for ``steps`` iterations."""

    values = np.empty(steps, dtype=float)
    x = initial_value
    for i in range(steps):
        x = map_function(alpha, x)
        values[i] = x
    return values


def compute_bifurcation(
    model: MapModel,
    alpha_values: np.ndarray,
    burn_in: int,
    sample_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute long-run orbit samples for the bifurcation diagram."""

    all_alphas: List[float] = []
    orbit_samples: List[float] = []

    total_steps = burn_in + sample_points
    for alpha in alpha_values:
        trajectory = iterate_map(
            model.map_function,
            alpha,
            model.initial_condition,
            total_steps,
        )
        samples = trajectory[burn_in:]
        all_alphas.extend([alpha] * len(samples))
        orbit_samples.extend(samples)

    return np.asarray(all_alphas), np.asarray(orbit_samples)


def compute_lyapunov(
    model: MapModel,
    alpha_values: np.ndarray,
    burn_in: int,
    sample_points: int,
) -> np.ndarray:
    """Estimate the Lyapunov exponent for each parameter value."""

    exponents = np.empty_like(alpha_values)
    total_steps = burn_in + sample_points

    for idx, alpha in enumerate(alpha_values):
        x = model.initial_condition
        lyapunov_sum = 0.0
        count = 0
        for step in range(total_steps):
            x = model.map_function(alpha, x)
            derivative = model.derivative_function(alpha, x)
            if step >= burn_in:
                # Prevent numerical blow-ups when derivative is extremely small.
                abs_derivative = max(abs(derivative), 1e-12)
                lyapunov_sum += math.log(abs_derivative)
                count += 1
        exponents[idx] = lyapunov_sum / max(count, 1)

    return exponents


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_bifurcation(
    ax: matplotlib.axes.Axes,
    model: MapModel,
    alpha_values: np.ndarray,
    orbit_parameters: np.ndarray,
    orbit_samples: np.ndarray,
) -> None:
    """Plot a standard bifurcation diagram."""

    ax.scatter(
        orbit_parameters,
        orbit_samples,
        s=0.1,
        alpha=0.7,
        color="#1f77b4",
        linewidths=0,
    )
    ax.set_title(f"Bifurcation diagram — {model.description}")
    ax.set_xlabel(model.parameter_label)
    ax.set_ylabel("Long-run orbit values")
    ax.set_xlim(alpha_values[0], alpha_values[-1])
    ax.set_ylim(*model.domain)


def plot_lyapunov(
    ax: matplotlib.axes.Axes,
    model: MapModel,
    alpha_values: np.ndarray,
    exponents: np.ndarray,
) -> None:
    """Plot Lyapunov exponent curve and highlight zero crossings."""

    ax.plot(alpha_values, exponents, color="#d62728", label="Lyapunov exponent")
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", label="Zero line")

    sign_changes = np.where(np.diff(np.sign(exponents)) != 0)[0]
    for idx in sign_changes:
        a0, a1 = alpha_values[idx], alpha_values[idx + 1]
        l0, l1 = exponents[idx], exponents[idx + 1]
        if l1 == l0:
            crossing_alpha = a0
        else:
            crossing_alpha = a0 - l0 * (a1 - a0) / (l1 - l0)
        ax.scatter(
            crossing_alpha,
            0.0,
            color="#2ca02c",
            s=30,
            zorder=5,
            label="Chaos onset" if idx == sign_changes[0] else None,
        )
        ax.axvline(crossing_alpha, color="#2ca02c", linestyle=":", linewidth=0.8)

    ax.set_title(f"Lyapunov exponent — {model.description}")
    ax.set_xlabel(model.parameter_label)
    ax.set_ylabel(r"Lyapunov exponent $\\lambda$")
    ax.set_xlim(alpha_values[0], alpha_values[-1])
    ax.legend(loc="best")


def compute_cobweb_points(
    map_function: MapFunction,
    alpha: float,
    x0: float,
    iterations: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute points for a cobweb diagram."""

    xs = [x0]
    ys = [0.0]
    x = x0
    for _ in range(iterations):
        y = map_function(alpha, x)
        xs.extend([x, x])
        ys.extend([y, y])
        x = y
        xs.append(x)
        ys.append(x)
    return np.asarray(xs), np.asarray(ys)


def plot_cobwebs(
    fig: matplotlib.figure.Figure,
    model: MapModel,
    alpha_values: Sequence[float],
    iterations: int,
    initial_condition: float,
) -> None:
    """Draw cobweb diagrams for the provided ``alpha_values``."""

    domain = np.linspace(*model.domain, 500)
    map_curve = model.map_function

    for idx, alpha in enumerate(alpha_values):
        ax = fig.add_subplot(1, len(alpha_values), idx + 1)
        ax.plot(domain, [map_curve(alpha, x) for x in domain], color="#1f77b4", label="Map")
        ax.plot(domain, domain, color="black", linestyle="--", linewidth=0.8, label="y=x")
        cobweb_xs, cobweb_ys = compute_cobweb_points(
            map_curve,
            alpha,
            initial_condition,
            iterations,
        )
        ax.plot(cobweb_xs, cobweb_ys, color="#ff7f0e", linewidth=0.8)
        ax.set_title(fr"$\\alpha = {alpha:.4g}$")
        ax.set_xlim(*model.domain)
        ax.set_ylim(*model.domain)
        if idx == 0:
            ax.set_ylabel("Next iterate")
        ax.set_xlabel("Current iterate")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle(f"Cobweb diagrams — {model.description}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate visualisations for classic one-dimensional maps.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where plots will be saved (created if missing).",
    )
    parser.add_argument(
        "--alpha-steps",
        type=int,
        default=1200,
        help="Number of parameter samples for the bifurcation and Lyapunov plots.",
    )
    parser.add_argument(
        "--burn-in",
        type=int,
        default=400,
        help="Number of transient iterations discarded before sampling orbits.",
    )
    parser.add_argument(
        "--sample-points",
        type=int,
        default=400,
        help="Number of orbit points recorded per parameter value.",
    )
    parser.add_argument(
        "--cobweb-iters",
        type=int,
        default=None,
        help="Override the default number of iterations used for cobweb diagrams.",
    )
    parser.add_argument(
        "--initial",
        type=float,
        default=None,
        help="Override the default initial condition for all computations.",
    )
    return parser


def ensure_directory(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_figure(fig: matplotlib.figure.Figure, output_dir: str, filename: str) -> str:
    ensure_directory(output_dir)
    full_path = os.path.join(output_dir, filename)
    fig.savefig(full_path, dpi=300)
    plt.close(fig)
    return full_path


def main(args: Sequence[str] | None = None) -> None:
    parser = build_argument_parser()
    options = parser.parse_args(args=args)

    alpha_steps = max(10, options.alpha_steps)
    burn_in = max(0, options.burn_in)
    sample_points = max(1, options.sample_points)

    for model in MODELS:
        alpha_values = np.linspace(
            model.parameter_range[0], model.parameter_range[1], alpha_steps
        )
        initial_condition = (
            options.initial if options.initial is not None else model.initial_condition
        )
        cobweb_iterations = (
            options.cobweb_iters if options.cobweb_iters is not None else model.cobweb_iterations
        )

        # Bifurcation diagram
        orbit_params, orbit_samples = compute_bifurcation(
            model,
            alpha_values,
            burn_in,
            sample_points,
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_bifurcation(ax, model, alpha_values, orbit_params, orbit_samples)
        save_figure(
            fig,
            options.output_dir,
            f"{model.name}_bifurcation.png",
        )

        # Lyapunov exponents
        exponents = compute_lyapunov(
            model,
            alpha_values,
            burn_in,
            sample_points,
        )
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_lyapunov(ax, model, alpha_values, exponents)
        save_figure(
            fig,
            options.output_dir,
            f"{model.name}_lyapunov.png",
        )

        # Cobweb diagrams
        fig = plt.figure(figsize=(12, 4))
        plot_cobwebs(
            fig,
            model,
            model.cobweb_alphas,
            cobweb_iterations,
            initial_condition,
        )
        save_figure(
            fig,
            options.output_dir,
            f"{model.name}_cobweb.png",
        )


if __name__ == "__main__":
    main()
