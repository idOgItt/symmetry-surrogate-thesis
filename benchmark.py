"""Periodic diffusion, candidate transformations, and the linear surrogate."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Problem:
    name: str
    diffusion_y: float
    diffusion_x: float


PROBLEMS = (
    Problem("isotropic", 0.65, 0.65),
    Problem("weak_anisotropy", 0.55, 0.75),
    Problem("strong_anisotropy", 0.2, 1.1),
)


@dataclass(frozen=True)
class Transform:
    name: str
    turns: int
    reflected: bool = False

    @property
    def swaps_axes(self):
        return self.turns % 2 == 1

    def apply(self, fields):
        if self.reflected:
            fields = np.flip(fields, axis=-1)
        return np.rot90(fields, k=self.turns, axes=(-2, -1))


# D4 without the identity, which always commutes with a surrogate.
TRANSFORMS = (
    Transform("rot90", 1),
    Transform("rot180", 2),
    Transform("rot270", 3),
    Transform("flip_x", 0, True),
    Transform("flip_y", 2, True),
    Transform("flip_diag", 1, True),
    Transform("flip_antidiag", 3, True),
)


def frequency_laplacian(grid):
    """Eigenvalues of the positive one-dimensional periodic Laplacian."""
    return 4 * np.sin(np.pi * np.fft.fftfreq(grid)) ** 2


def random_fields(rng, count, grid):
    wave = frequency_laplacian(grid)
    laplacian = wave[:, None] + wave[None, :]
    spectral_filter = (1 + laplacian) ** -1
    white_noise = rng.standard_normal((count, grid, grid))
    spectrum = np.fft.fft2(white_noise, axes=(-2, -1))
    fields = np.fft.ifft2(spectrum * spectral_filter, axes=(-2, -1)).real
    return fields / np.sqrt(np.mean(spectral_filter ** 2))


def diffuse(fields, problem, time=1.0):
    """Exact time advance for the spatially discrete heat equation."""
    wave = frequency_laplacian(fields.shape[-1])
    generator = problem.diffusion_y * wave[:, None] + problem.diffusion_x * wave[None, :]
    spectrum = np.fft.fft2(fields, axes=(-2, -1))
    return np.fft.ifft2(spectrum * np.exp(-time * generator), axes=(-2, -1)).real


def is_symmetry(problem, transform):
    return not transform.swaps_axes or problem.diffusion_y == problem.diffusion_x


@dataclass(frozen=True)
class LinearSurrogate:
    weight: np.ndarray

    def __call__(self, fields):
        return (fields.reshape(len(fields), -1) @ self.weight).reshape(fields.shape)


def periodic_laplacian(grid):
    """Matrix whose quadratic form sums squared nearest-neighbor differences."""
    basis = np.eye(grid * grid).reshape(-1, grid, grid)
    laplacian = 4 * basis
    for axis in (1, 2):
        laplacian -= np.roll(basis, 1, axis=axis)
        laplacian -= np.roll(basis, -1, axis=axis)
    return laplacian.reshape(grid * grid, -1)


class SurrogateFit:
    """Solve least squares with L2 and output-gradient penalties on the weights."""

    def __init__(self, inputs, output_eigenbasis):
        self.count = len(inputs)
        self.left, self.singular, self.right = np.linalg.svd(
            inputs.reshape(self.count, -1), full_matrices=False
        )
        values, self.output_modes = output_eigenbasis
        self.output_frequencies = np.maximum(values, 0.0)

    def targets(self, outputs):
        return self.left.T @ outputs.reshape(self.count, -1) @ self.output_modes

    def solve(self, targets, ridge=0.0, smooth=0.0):
        if ridge == smooth == 0.0:
            # Match the minimum-norm solution of np.linalg.lstsq.
            cutoff = np.finfo(float).eps * max(self.count, self.right.shape[1])
            inverse = np.divide(
                1.0, self.singular, out=np.zeros_like(self.singular),
                where=self.singular > cutoff * self.singular[0],
            )
            weights = inverse[:, None] * targets
        else:
            singular = self.singular[:, None]
            numerator = singular * targets / self.count
            denominator = singular ** 2 / self.count + ridge + smooth * self.output_frequencies
            # Discard numerical nullspace directions when smoothness alone
            # leaves the constant output mode unpenalized.
            cutoff = np.finfo(float).eps * max(self.count, self.right.shape[1])
            has_input_signal = singular > cutoff * self.singular[0]
            weights = np.divide(
                numerator, denominator, out=np.zeros_like(numerator),
                where=has_input_signal & (denominator > 0),
            )
        return LinearSurrogate(self.right.T @ weights @ self.output_modes.T)
