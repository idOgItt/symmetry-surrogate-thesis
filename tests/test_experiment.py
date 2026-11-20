import unittest
from functools import partial

import numpy as np

from benchmark import (
    PROBLEMS, TRANSFORMS, SurrogateFit, diffuse, is_symmetry,
    periodic_laplacian, random_fields,
)
from metrics import commutator_errors, detection_metrics


class DiffusionTest(unittest.TestCase):
    def test_diagonal_transform_names(self):
        field = np.arange(9).reshape(3, 3)
        by_name = {transform.name: transform for transform in TRANSFORMS}
        np.testing.assert_array_equal(by_name["flip_diag"].apply(field), field.T)
        np.testing.assert_array_equal(by_name["flip_antidiag"].apply(field), field.T[::-1, ::-1])

    def test_fourier_mode(self):
        grid = 7
        fields = np.broadcast_to(np.cos(2 * np.pi * np.arange(grid) / grid), (1, grid, grid))
        problem = PROBLEMS[1]
        attenuation = np.exp(-4 * problem.diffusion_x * np.sin(np.pi / grid) ** 2)
        np.testing.assert_allclose(diffuse(fields, problem), attenuation * fields, atol=1e-14)

    def test_analytic_symmetry_labels_match_reference(self):
        fields = random_fields(np.random.default_rng(18), 41, 9)
        for problem in PROBLEMS:
            reference = partial(diffuse, problem=problem)
            scores = commutator_errors(reference, fields, reference(fields), TRANSFORMS)
            for transform in TRANSFORMS:
                with self.subTest(problem=problem.name, transform=transform.name):
                    if is_symmetry(problem, transform):
                        self.assertLess(scores[transform.name], 1e-13)
                    else:
                        self.assertGreater(scores[transform.name], 0.05)


class SurrogateTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(951)
        self.inputs = rng.normal(size=(6, 3, 3))
        self.outputs = rng.normal(size=(6, 3, 3))
        self.laplacian = periodic_laplacian(3)
        self.fit = SurrogateFit(self.inputs, np.linalg.eigh(self.laplacian))
        self.targets = self.fit.targets(self.outputs)
        self.x = self.inputs.reshape(6, 9)
        self.y = self.outputs.reshape(6, 9)

    def test_minimum_norm_baseline(self):
        expected, *_ = np.linalg.lstsq(self.x, self.y, rcond=None)
        np.testing.assert_allclose(self.fit.solve(self.targets).weight, expected, atol=1e-13)

    def test_regularized_normal_equations(self):
        for ridge, smooth in ((0.3, 0.0), (0.0, 0.2), (0.1, 0.2)):
            with self.subTest(ridge=ridge, smooth=smooth):
                weight = self.fit.solve(self.targets, ridge, smooth).weight
                gradient = self.x.T @ (self.x @ weight - self.y) / len(self.x)
                gradient += ridge * weight + smooth * weight @ self.laplacian
                np.testing.assert_allclose(gradient, 0, atol=1e-13)
                self.assertGreater(np.linalg.norm(weight - self.fit.solve(self.targets).weight), 0.01)


class DetectionTest(unittest.TestCase):
    def test_threshold_and_auc_with_a_tie(self):
        scores = [0.1, 0.2, 0.2, 0.4]
        metrics = detection_metrics(scores, [1, 1, 0, 0], threshold=0.2)
        self.assertEqual(metrics["roc_auc"], 0.875)
        self.assertEqual(metrics["f1"], 0.8)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["false_negatives"], 0)
        self.assertEqual(metrics["margin"], 0.0)


if __name__ == "__main__":
    unittest.main()
