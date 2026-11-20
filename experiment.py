"""Run the paired experiment and export its results."""

import argparse
import csv
import json
import platform
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

from benchmark import (
    PROBLEMS, TRANSFORMS, SurrogateFit, diffuse, is_symmetry,
    periodic_laplacian, random_fields,
)
from metrics import commutator_errors, detection_metrics, relative_error


@dataclass(frozen=True)
class Config:
    grid: int = 9
    train_sizes: tuple[int, ...] = (24, 96)
    noise_levels: tuple[float, ...] = (0.0, 0.2)
    strengths: tuple[float, ...] = (0.003, 0.03, 0.3)
    seeds: tuple[int, ...] = tuple(range(20))
    test_size: int = 256
    threshold: float = 0.2


def variants(strengths):
    yield "none", 0.0, 0.0
    for strength in strengths:
        yield "ridge", strength, 0.0
        # Scale by the spectral upper bound of the grid Laplacian.
        yield "smooth", 0.0, strength / 8


def run(config):
    run_rows, score_rows = [], []
    output_eigenbasis = np.linalg.eigh(periodic_laplacian(config.grid))
    for seed in config.seeds:
        train_rng, test_rng, noise_rng = (
            np.random.default_rng(stream) for stream in np.random.SeedSequence(seed).spawn(3)
        )
        train = random_fields(train_rng, max(config.train_sizes), config.grid)
        test = random_fields(test_rng, config.test_size, config.grid)
        noise = noise_rng.standard_normal(train.shape)
        problems = []
        for problem in PROBLEMS:
            reference = partial(diffuse, problem=problem)
            clean_train, clean_test = reference(train), reference(test)
            oracle = commutator_errors(reference, test, clean_test, TRANSFORMS)
            problems.append((problem, clean_train, clean_test, oracle))

        for count in config.train_sizes:
            fit = SurrogateFit(train[:count], output_eigenbasis)
            for sigma in config.noise_levels:
                prepared = []
                for problem, clean_train, clean_test, oracle in problems:
                    targets = clean_train[:count]
                    targets = targets + sigma * np.sqrt(np.mean(targets ** 2)) * noise[:count]
                    prepared.append((problem, fit.targets(targets), clean_test, oracle))

                for method, ridge, smooth in variants(config.strengths):
                    strength = ridge + 8 * smooth
                    key = dict(seed=seed, train_size=count, noise=sigma, method=method, strength=strength)
                    residuals, labels, test_errors = [], [], []
                    for problem, targets, clean_test, oracle in prepared:
                        surrogate = fit.solve(targets, ridge, smooth)
                        prediction = surrogate(test)
                        test_errors.append(relative_error(prediction, clean_test))
                        scores = commutator_errors(surrogate, test, prediction, TRANSFORMS)
                        for transform in TRANSFORMS:
                            label = is_symmetry(problem, transform)
                            residual = scores[transform.name]
                            residuals.append(residual)
                            labels.append(label)
                            score_rows.append(dict(
                                key, problem=problem.name, transform=transform.name,
                                is_symmetry=int(label), commutator=residual,
                                reference_commutator=oracle[transform.name],
                            ))

                    run_rows.append(dict(
                        key, relative_l2=float(np.mean(test_errors)),
                        **{f"relative_l2_{p.name}": v for p, v in zip(PROBLEMS, test_errors)},
                        **detection_metrics(residuals, labels, config.threshold),
                    ))
    return run_rows, score_rows


def summarize(rows, seeds):
    groups = {}
    baseline = {}
    for row in rows:
        regime = row["train_size"], row["noise"]
        variant = row["method"], row["strength"]
        groups.setdefault(regime + variant, []).append(row)
        if row["method"] == "none":
            baseline[regime + (row["seed"],)] = row

    # Resample the same seeds for each paired comparison.
    indices = np.random.default_rng(618).integers(0, len(seeds), size=(4000, len(seeds)))
    summary = []
    for (count, sigma, method, strength), runs in groups.items():
        entry = dict(train_size=count, noise=sigma, method=method, strength=strength, n=len(runs))
        for metric in ("relative_l2", "roc_auc", "f1", "margin", "false_positives", "false_negatives"):
            values = np.array([row[metric] for row in runs])
            controls = np.array([baseline[count, sigma, row["seed"]][metric] for row in runs])
            delta = values - controls
            low, high = np.quantile(delta[indices].mean(axis=1), [0.025, 0.975])
            entry.update({
                f"{metric}_mean": float(values.mean()),
                f"{metric}_std": float(values.std(ddof=1)),
                f"{metric}_delta": float(delta.mean()),
                f"{metric}_delta_ci_low": float(low),
                f"{metric}_delta_ci_high": float(high),
            })
        summary.append(entry)
    return summary


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    from dataclasses import asdict

    import matplotlib

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="override the default output directory")
    parser.add_argument("--quick", action="store_true", help="use only two seeds and a smaller test set")
    args = parser.parse_args()
    config = Config(seeds=(0, 1), test_size=64) if args.quick else Config()
    output = args.out or Path("results/quick" if args.quick else "results/full")
    output.mkdir(parents=True, exist_ok=True)

    from plot_results import plot_overview

    runs, scores = run(config)
    summary = summarize(runs, config.seeds)
    write_csv(output / "runs.csv", runs)
    write_csv(output / "symmetry_scores.csv", scores)
    write_csv(output / "summary.csv", summary)
    metadata = dict(
        config=asdict(config), problems=[asdict(problem) for problem in PROBLEMS],
        python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__,
    )
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    plot_overview(summary, config, output / "overview.png")
    print(f"{len(runs)} trials, {len(scores)} candidate scores, output: {output.resolve()}")


if __name__ == "__main__":
    main()
