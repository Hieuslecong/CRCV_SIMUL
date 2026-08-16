import numpy as np
import pytest
import torch

from crcv52.relational_v55 import (
    CRCVV55RelationalBlock,
    build_component_view,
    build_relation_views,
    same_source_rank_loss,
    top1_margin_abstention,
)
from crcv52.runtime_v55 import CRCVV55Block, StructuralSafetyGate, V55RuntimeConfig
from crcv52.sim_prior import SimulationGeometryPrior, fit_simulation_prior


def _prior():
    seqs = []
    for k in range(20):
        x = np.linspace(0, 30, 31, dtype=np.float32)
        y = 0.3 * np.sin(np.linspace(0, 2.0 + 0.03 * k, 31)).astype(np.float32)
        seqs.append(np.c_[x, y])
    return SimulationGeometryPrior(fit_simulation_prior(seqs))


def _toy_record():
    h = w = 64
    image = np.ones((h, w, 3), np.float32) * 0.7
    image[31:34, 5:59] = 0.2
    prob = np.full((h, w), 0.05, np.float32)
    base = np.zeros((h, w), bool)
    base[32, 5:26] = 1
    base[32, 38:59] = 1
    prob[base] = 0.9
    return {"image": image, "prob": prob, "base": base, "threshold": 0.4}


def test_simulation_prior_prefers_smooth_crack_like_path():
    prior = _prior()
    smooth = np.c_[
        np.linspace(0, 30, 31),
        0.3 * np.sin(np.linspace(0, 2, 31)),
    ].astype(np.float32)
    zigzag = np.asarray([[i, 5 * (i % 2)] for i in range(31)], np.float32)
    assert prior.score_polyline(smooth) > 0.5
    assert prior.score_polyline(smooth) > prior.score_polyline(zigzag) + 0.3


def test_relation_views_are_gt_free_and_have_expected_shape():
    rec = _toy_record()
    add = np.zeros_like(rec["base"])
    add[32, 26:38] = 1
    cand = {
        "family": "v54_ridge_continue",
        "add": add,
        "path_yx": [(32, x) for x in range(25, 39)],
        "source_yx": (32, 25),
        "length": 12,
        "score": 0.8,
        "source_score": 0.7,
        "mean_ridge": 0.8,
    }
    sv, pv, dv, meta = build_relation_views(rec, cand, sim_score=0.9)
    assert sv.shape == pv.shape == dv.shape == (8, 33, 33)
    assert meta.shape == (13,)
    assert "gt" not in rec


def test_v55_model_is_lightweight_and_has_three_recovery_heads():
    model = CRCVV55RelationalBlock()
    params = sum(p.numel() for p in model.parameters())
    assert params < 250_000
    x = torch.randn(2, 8, 33, 33)
    meta = torch.randn(2, 13)
    out = model.forward_recovery(x, x, x, meta)
    assert set(out) == {"same_crack_logit", "path_valid_logit", "continuity_logit"}
    assert all(v.shape == (2,) for v in out.values())


def test_same_source_ranking_rewards_correct_order():
    labels = torch.tensor([1, 0, 0, 1, 0])
    groups = torch.tensor([0, 0, 0, 1, 1])
    good = torch.tensor([3.0, -1.0, -2.0, 2.0, -1.0])
    bad = torch.tensor([-1.0, 3.0, 2.0, -1.0, 2.0])
    assert same_source_rank_loss(good, labels, groups) < same_source_rank_loss(
        bad, labels, groups
    )


def test_top1_margin_abstention_rejects_ambiguous_source():
    out = top1_margin_abstention(
        np.array([0.90, 0.40, 0.80, 0.75]),
        np.array([0, 0, 1, 1]),
        absolute_threshold=0.80,
        margin_threshold=0.20,
    )
    by = {z.group_id: z for z in out}
    assert by[0].accepted_index == 0
    assert by[1].accepted_index is None


def test_component_suppression_view_uses_shared_eight_channel_schema():
    rec = _toy_record()
    comp = np.zeros_like(rec["base"])
    comp[31:34, 5:26] = 1
    view, feat = build_component_view(rec, comp, sim_score=0.8)
    assert view.shape == (8, 33, 33)
    assert feat.shape == (12,)


def test_runtime_is_exactly_fail_closed_when_heads_unqualified():
    cfg = V55RuntimeConfig(
        proposal_qualified=True,
        relation_verifier_qualified=False,
        suppression_qualified=False,
        recovery_enabled=False,
        suppression_enabled=False,
        width_enabled=False,
        joint_training_enabled=False,
        runtime_policy="FAIL_CLOSED_BASE_ONLY",
        max_add_fraction=0.02,
        max_remove_fraction=0.01,
        max_cc_increase=0,
    )
    base = np.zeros((32, 32), np.uint8)
    base[16, 5:20] = 1
    refined, meta = CRCVV55Block(cfg).refine(base)
    assert np.array_equal(refined, base)
    assert not meta["recovery_applied"]
    assert not meta["suppression_applied"]


def test_runtime_refuses_unqualified_enabled_recovery():
    cfg = V55RuntimeConfig(
        proposal_qualified=True,
        relation_verifier_qualified=False,
        suppression_qualified=False,
        recovery_enabled=True,
        suppression_enabled=False,
        width_enabled=False,
        joint_training_enabled=False,
        runtime_policy="FAIL_CLOSED_BASE_ONLY",
        max_add_fraction=0.02,
        max_remove_fraction=0.01,
        max_cc_increase=0,
    )
    with pytest.raises(RuntimeError):
        CRCVV55Block(cfg, recovery_fn=lambda r: np.zeros((8, 8), bool))


def test_structural_safety_rejects_isolated_addition():
    base = np.zeros((32, 32), bool)
    base[16, 5:20] = 1
    isolated = np.zeros_like(base)
    isolated[2:4, 2:4] = 1
    gate = StructuralSafetyGate(
        max_add_fraction=0.5, max_remove_fraction=0.5, max_cc_increase=0
    )
    refined, meta = gate.validate(base, add=isolated)
    assert not meta["safety_pass"]
    assert "isolated_add" in meta["reasons"]
    assert np.array_equal(refined.astype(bool), base)
