"""Operator accuracy and symmetry detection metrics."""

import numpy as np


def relative_error(prediction, reference):
    scale = np.linalg.norm(reference.ravel())
    if scale <= np.finfo(float).tiny:
        raise ValueError("Relative error requires a nonzero reference")
    return float(np.linalg.norm((prediction - reference).ravel()) / scale)


def commutator_errors(operator, inputs, prediction, transforms):
    # All candidate actions preserve the Euclidean norm of prediction.
    return {
        transform.name: relative_error(
            operator(transform.apply(inputs)), transform.apply(prediction)
        )
        for transform in transforms
    }


def detection_metrics(residuals, labels, threshold):
    residuals = np.asarray(residuals)
    labels = np.asarray(labels, dtype=bool)
    positives, negatives = residuals[labels], residuals[~labels]
    if not len(positives) or not len(negatives):
        raise ValueError("Detection metrics require both positive and negative candidates")
    differences = negatives[None, :] - positives[:, None]
    auc = np.mean((differences > 0) + 0.5 * (differences == 0))
    predicted = residuals <= threshold
    tp = np.count_nonzero(predicted & labels)
    fp = np.count_nonzero(predicted & ~labels)
    fn = np.count_nonzero(~predicted & labels)
    return {
        "roc_auc": float(auc),
        "f1": float(2 * tp / (2 * tp + fp + fn)),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "margin": float(negatives.min() - positives.max()),
    }
