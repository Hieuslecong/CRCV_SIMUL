from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from lightgbm import LGBMClassifier

from .counterfactual_errors import CounterfactualConfig, simulate_corruption
from .removal_policy import sample_keep_remove_training


@dataclass(frozen=True)
class V5201PolicyTrainingConfig:
    natural_max_keep: int = 1400
    natural_max_remove: int = 450
    natural_boundary_keep_fraction: float = 0.60
    synthetic_max_keep: int = 260
    synthetic_max_remove: int = 120
    synthetic_boundary_keep_fraction: float = 0.70
    synthetic_probability_lift_min: float = 0.01
    synthetic_probability_lift_max: float = 0.12
    n_estimators: int = 160
    learning_rate: float = 0.04
    num_leaves: int = 15
    max_depth: int = 5
    min_child_samples: int = 30
    reg_lambda: float = 0.5


def counterfactual_training_rows(record, real_probability, base_threshold: float,
                                 index: int, seed: int = 1337,
                                 simulator_config: CounterfactualConfig | None = None,
                                 config: V5201PolicyTrainingConfig | None = None):
    """Build canonical V5.20.1 KEEP/REMOVE rows from one FIT record.

    Counterfactual over-segmentation is overlaid on the real Base mask. Synthetic
    pixels receive a range of plausible probabilities instead of a single fixed
    confidence artifact. GT is used only to construct training targets.
    """
    cfg = config or V5201PolicyTrainingConfig()
    sim_cfg = simulator_config or CounterfactualConfig()
    image = record["image"]
    gt = record["gt"]
    real_probability = np.asarray(real_probability, np.float32)
    real_base = real_probability >= float(base_threshold)
    Xs, ys = [], []

    X, y, _ = sample_keep_remove_training(
        image, real_probability, real_base, gt,
        max_keep=cfg.natural_max_keep,
        max_remove=cfg.natural_max_remove,
        boundary_keep_fraction=cfg.natural_boundary_keep_fraction,
        seed=seed + index,
    )
    if len(y):
        Xs.append(X); ys.append(y)

    for j, op in enumerate(["width_dilate", "side_spur", "isolated_blob", "false_bridge"]):
        corr, _ = simulate_corruption(gt, op, seed=seed + 1000*index + 17*j, config=sim_cfg)
        added = corr & ~gt
        if not added.any():
            continue
        cbase = real_base | added
        cprob = real_probability.copy()
        rng = np.random.default_rng(seed + 3000*index + j)
        lift = np.clip(
            base_threshold + rng.uniform(cfg.synthetic_probability_lift_min,
                                         cfg.synthetic_probability_lift_max,
                                         size=int(added.sum())),
            0, 0.995,
        ).astype(np.float32)
        cprob[added] = np.maximum(cprob[added], lift)
        X, y, _ = sample_keep_remove_training(
            image, cprob, cbase, gt,
            max_keep=cfg.synthetic_max_keep,
            max_remove=cfg.synthetic_max_remove,
            boundary_keep_fraction=cfg.synthetic_boundary_keep_fraction,
            seed=seed + 2000*index + j,
        )
        if len(y):
            Xs.append(X); ys.append(y)
    return Xs, ys


def train_pixel_remove_policy(fit_records, fit_probabilities: dict, base_threshold: float,
                              seed: int = 1337,
                              simulator_config: CounterfactualConfig | None = None,
                              config: V5201PolicyTrainingConfig | None = None):
    """Canonical reproducible V5.20.1 pixel-candidate learner.

    No class_weight='balanced' is used: KEEP-heavy sampling already controls the
    target distribution and prevents the former double-overweighting of REMOVE.
    """
    cfg = config or V5201PolicyTrainingConfig()
    Xs, ys = [], []
    for i, record in enumerate(fit_records):
        a, b = counterfactual_training_rows(record, fit_probabilities[record["name"]],
                                            base_threshold, i, seed,
                                            simulator_config, cfg)
        Xs += a; ys += b
    if not Xs:
        raise ValueError("no policy training rows generated")
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    if len(np.unique(y)) < 2:
        raise ValueError("KEEP/REMOVE training requires both classes")
    clf = LGBMClassifier(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth,
        min_child_samples=cfg.min_child_samples,
        reg_lambda=cfg.reg_lambda,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )
    clf.fit(X, y)
    return clf, {
        "n": int(len(y)),
        "keep": int((y == 0).sum()),
        "remove": int((y == 1).sum()),
        "remove_fraction": float((y == 1).mean()),
    }
