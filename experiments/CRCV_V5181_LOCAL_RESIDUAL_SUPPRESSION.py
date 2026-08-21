from __future__ import annotations

import json
import pickle
import importlib.util
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np
from lightgbm import LGBMRegressor

ROOT = Path('/mnt/data')
OUT = ROOT / 'v5181_run'
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ['TinyUNet','FastSCNNLite','BiSeNetTiny','MobileNetV3SmallSeg','DSUNetLite']
PRIMARY = ['precision','recall','f1','miou']
SEEDS = [1337, 2027, 31415]

spec = importlib.util.spec_from_file_location('v518', ROOT / 'CRCV_V518_WIDTH_SUPPRESSION_MULTIMODEL_SMOKE.py')
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

AUTH_THRESHOLDS = np.asarray([.015,.020,.025,.030,.035,.040,.045,.050,.055,.060], np.float32)
MAX_REGION_FRAC = 0.025
MAX_TOTAL_REMOVE_FRAC = 0.030
MIN_REGION_PIXELS = 2
MAX_ADD_ACTIONS = 3
SUPPRESSION_RECALL_FLOOR = -0.005
FULL_RECALL_FLOOR = 0.0


def _metrics_delta(c0, c1):
    b = v.v.metrics(c0)
    q = v.v.metrics(c1)
    return {k: float(q[k] - b[k]) for k in PRIMARY}, b, q


def local_residual_suppress(base: np.ndarray, authenticity: np.ndarray, threshold: float,
                            max_region_frac: float = MAX_REGION_FRAC,
                            max_total_frac: float = MAX_TOTAL_REMOVE_FRAC,
                            min_region_pixels: int = MIN_REGION_PIXELS):
    base = np.asarray(base, bool)
    auth = np.asarray(authenticity, np.float32)
    if base.shape != auth.shape:
        raise ValueError('base/authenticity shape mismatch')
    risk = base & np.isfinite(auth) & (auth < float(threshold))
    n, lab = cv2.connectedComponents(risk.astype(np.uint8), 8)
    out = base.copy()
    removed = np.zeros_like(base, bool)
    total = 0
    max_region = max(1, int(np.floor(max_region_frac * base.size)))
    max_total = max(1, int(np.floor(max_total_frac * base.size)))
    for cid in range(1, n):
        region = lab == cid
        area = int(region.sum())
        if area < int(min_region_pixels):
            continue
        if area > max_region or total + area > max_total:
            continue
        region &= out
        area = int(region.sum())
        if area < int(min_region_pixels):
            continue
        out[region] = False
        removed[region] = True
        total += area
    return out, removed


def _add_scores_by_image(add_rows, add_scores):
    by = defaultdict(list)
    for row, score in zip(add_rows, add_scores):
        by[row['image']].append((row, float(score)))
    return by


def apply_add_actions(mask: np.ndarray, rows_scores, add_threshold: float,
                      max_add: int = MAX_ADD_ACTIONS):
    out = np.asarray(mask, bool).copy()
    used_sources = set()
    accepted = []
    for row, s in sorted(rows_scores, key=lambda z: -z[1]):
        if s < add_threshold:
            break
        src = tuple(row['source'])
        sy, sx = map(int, src)
        if src in used_sources or not (0 <= sy < out.shape[0] and 0 <= sx < out.shape[1]):
            continue
        y0, y1 = max(0, sy - 2), min(out.shape[0], sy + 3)
        x0, x1 = max(0, sx - 2), min(out.shape[1], sx + 3)
        if not out[y0:y1, x0:x1].any():
            continue
        add = v.unpack(row['mask_pack'], row['shape']) & ~out
        if int(add.sum()) < 2:
            continue
        dil = cv2.dilate(add.astype(np.uint8), np.ones((3,3), np.uint8), 1).astype(bool)
        _, labs = cv2.connectedComponents(out.astype(np.uint8), 8)
        touched = set(labs[dil & out].tolist()) - {0}
        if len(touched) > 2:
            continue
        out |= add
        used_sources.add(src)
        accepted.append(add)
        if len(accepted) >= max_add:
            break
    return out, accepted


def build_auth_cache(records, authenticity_model):
    cache = {}
    for model_name in MODELS:
        for split in ['cal','val']:
            for rec in records[model_name][split]:
                cache[(model_name, split, rec['name'])] = v.authenticity_map(rec, authenticity_model)
    return cache


def evaluate_setting(records, add_rows_by_model, add_scores_by_model, auth_cache,
                     split: str, auth_threshold: float, add_threshold: float,
                     do_remove: bool = True, do_add: bool = True,
                     return_images: bool = False):
    per_model = {}
    images = {}
    for model_name in MODELS:
        add_by_image = _add_scores_by_image(add_rows_by_model[model_name], add_scores_by_model[model_name])
        c0 = np.zeros(4, np.int64)
        c1 = np.zeros(4, np.int64)
        removed_pixels = 0
        added_pixels = 0
        remove_regions = 0
        add_actions = 0
        per_image = {}
        for rec in records[model_name][split]:
            base = np.asarray(rec['base'], bool)
            if do_remove:
                auth = auth_cache[(model_name, split, rec['name'])]
                current, removed = local_residual_suppress(base, auth, auth_threshold)
                removed_pixels += int(removed.sum())
                if removed.any():
                    remove_regions += int(cv2.connectedComponents(removed.astype(np.uint8),8)[0] - 1)
            else:
                current = base.copy()
                removed = np.zeros_like(base, bool)
            before_add = current.copy()
            accepted = []
            if do_add:
                current, accepted = apply_add_actions(current, add_by_image[rec['name']], add_threshold)
                add_actions += len(accepted)
                added_pixels += int((current & ~before_add).sum())
            gt = rec['gt'] > .5
            c0 += v.v.confusion(base, gt)
            c1 += v.v.confusion(current, gt)
            if return_images:
                per_image[rec['name']] = {
                    'final': current,
                    'removed': removed,
                    'added': current & ~before_add,
                }
        delta, bmet, qmet = _metrics_delta(c0, c1)
        per_model[model_name] = {
            'base': bmet,
            'refined': qmet,
            'delta': delta,
            'removed_pixels': int(removed_pixels),
            'added_pixels': int(added_pixels),
            'remove_regions': int(remove_regions),
            'add_actions': int(add_actions),
        }
        if return_images:
            images[model_name] = per_image
    if return_images:
        return per_model, images
    return per_model


def mean_delta(per_model):
    return {k: float(np.mean([per_model[n]['delta'][k] for n in MODELS])) for k in PRIMARY}


def summary(per_model):
    return {
        k: {
            'mean': float(np.mean([per_model[n]['delta'][k] for n in MODELS])),
            'std_across_backbones': float(np.std([per_model[n]['delta'][k] for n in MODELS])),
            'positive_models': int(sum(per_model[n]['delta'][k] > 0 for n in MODELS)),
            'ge_0p5pp': int(sum(per_model[n]['delta'][k] >= .005 for n in MODELS)),
        }
        for k in PRIMARY
    }


def calibrate(records, add_rows_by_model, add_scores_by_model, auth_cache):
    all_scores = np.concatenate([add_scores_by_model[n] for n in MODELS])
    add_grid = np.unique(np.r_[np.inf, np.quantile(all_scores, np.linspace(.35,.995,14))])
    best = None
    trace = []
    for auth_t in AUTH_THRESHOLDS:
        rem = evaluate_setting(records, add_rows_by_model, add_scores_by_model, auth_cache,
                               'cal', float(auth_t), np.inf, do_remove=True, do_add=False)
        rem_safe = all(
            rem[n]['delta']['recall'] >= SUPPRESSION_RECALL_FLOOR
            and rem[n]['delta']['f1'] >= 0
            and rem[n]['delta']['miou'] >= 0
            for n in MODELS
        )
        if not rem_safe:
            continue
        for add_t in add_grid:
            full = evaluate_setting(records, add_rows_by_model, add_scores_by_model, auth_cache,
                                    'cal', float(auth_t), float(add_t), True, True)
            feasible = all(
                full[n]['delta']['recall'] >= FULL_RECALL_FLOOR
                and full[n]['delta']['f1'] >= 0
                and full[n]['delta']['miou'] >= 0
                for n in MODELS
            )
            md = mean_delta(full)
            score = md['f1'] + 1.35*md['miou'] + .12*md['recall'] + .10*md['precision']
            action_penalty = sum(full[n]['add_actions'] + full[n]['remove_regions'] for n in MODELS) * 5e-6
            score -= action_penalty
            key = (int(feasible), float(score), -float(auth_t))
            trace.append({'auth_threshold':float(auth_t),'add_threshold':float(add_t),
                          'feasible':bool(feasible),'mean_delta':md})
            if best is None or key > best[0]:
                best = (key, float(auth_t), float(add_t), rem, full)
    if best is None or best[0][0] != 1:
        return np.inf, np.inf, None, None, trace
    return best[1], best[2], best[3], best[4], trace


def fit_add_regressor(fit_add, seed):
    X = np.stack([r['feat'] for r in fit_add])
    Y = np.asarray([v.utility_add(r['delta']) for r in fit_add], np.float32)
    sw = np.asarray([v.model_weight(r) * (1.25 if y < 0 else 1.0) for r, y in zip(fit_add, Y)], np.float32)
    model = LGBMRegressor(
        n_estimators=260, learning_rate=.027, num_leaves=15, min_child_samples=12,
        reg_lambda=10, reg_alpha=.8, subsample=.85, subsample_freq=1,
        colsample_bytree=.85, random_state=int(seed), verbosity=-1, n_jobs=4,
    )
    model.fit(X, Y, sample_weight=sw)
    return model


def run_seed(seed, records, add_variants, auth_model, auth_cache):
    fit_add = sum([add_variants[n]['fit'] for n in MODELS], [])
    add_model = fit_add_regressor(fit_add, seed)
    add_rows_cal = {n:add_variants[n]['cal'] for n in MODELS}
    add_rows_val = {n:add_variants[n]['val'] for n in MODELS}
    add_scores_cal = {n:v.score(add_model, add_rows_cal[n]) for n in MODELS}
    add_scores_val = {n:v.score(add_model, add_rows_val[n]) for n in MODELS}
    auth_t, add_t, cal_rem, cal_full, trace = calibrate(records, add_rows_cal, add_scores_cal, auth_cache)
    val_full, val_images = evaluate_setting(records, add_rows_val, add_scores_val, auth_cache,
                                            'val', auth_t, add_t, True, True, True)
    val_rem = evaluate_setting(records, add_rows_val, add_scores_val, auth_cache,
                               'val', auth_t, np.inf, True, False)
    val_add = evaluate_setting(records, add_rows_val, add_scores_val, auth_cache,
                               'val', np.inf, add_t, False, True)
    return {
        'seed':int(seed),
        'thresholds':{'authenticity':float(auth_t),'add':float(add_t)},
        'cal_suppression_only':cal_rem,
        'cal_full':cal_full,
        'val_suppression_only':val_rem,
        'val_recovery_only':val_add,
        'val_full':val_full,
        'summary':summary(val_full),
        'images':val_images,
        'calibration_trace':trace,
    }


def regression_checks(records, auth_model, auth_cache, seed_result):
    checks = {}
    checks['splits_only_fit_cal_val'] = all(set(records[n]) == {'fit','cal','val'} for n in MODELS)
    checks['no_test_key'] = all('test' not in records[n] and 'final' not in records[n] for n in MODELS)
    sample = records[MODELS[0]]['val'][0]
    auth = auth_cache[(MODELS[0], 'val', sample['name'])]
    a,_ = local_residual_suppress(sample['base'], auth, seed_result['thresholds']['authenticity'])
    mutated = dict(sample)
    mutated['gt'] = 1.0 - np.asarray(sample['gt'])
    b,_ = local_residual_suppress(mutated['base'], auth, seed_result['thresholds']['authenticity'])
    checks['local_residual_gt_invariant'] = bool(np.array_equal(a,b))
    subset_ok = True
    checked = 0
    for n in MODELS:
        for rec in records[n]['val'][:4]:
            out, removed = local_residual_suppress(rec['base'], auth_cache[(n,'val',rec['name'])], seed_result['thresholds']['authenticity'])
            subset_ok &= bool(np.all(~removed | np.asarray(rec['base'],bool)))
            subset_ok &= bool(np.all(~out | np.asarray(rec['base'],bool)))
            checked += 1
    checks['removed_subset_base'] = bool(subset_ok)
    checks['records_checked'] = int(checked)
    checks['all_finite_thresholds'] = bool(np.isfinite(seed_result['thresholds']['authenticity']) and np.isfinite(seed_result['thresholds']['add']))
    checks['full_f1_positive_5of5'] = bool(seed_result['summary']['f1']['positive_models'] == 5)
    checks['full_miou_positive_5of5'] = bool(seed_result['summary']['miou']['positive_models'] == 5)
    checks['full_recall_positive_5of5'] = bool(seed_result['summary']['recall']['positive_models'] == 5)
    checks['all_pass'] = bool(all(vv for kk,vv in checks.items() if kk != 'records_checked'))
    return checks


def strip_images(result):
    z = dict(result)
    z.pop('images', None)
    z.pop('calibration_trace', None)
    return z


def main():
    data = pickle.load(open(ROOT/'v517_clean_run'/'v517_banks.pkl','rb'))
    records = data['records']
    for n in MODELS:
        assert set(records[n]) == {'fit','cal','val'}
    add_variants_all = pickle.load(open(ROOT/'v518_run'/'add_variants_partial.pkl','rb'))
    add_variants = add_variants_all['learned']
    auth_model = pickle.load(open(ROOT/'v518_run'/'foreground_authenticity.pkl','rb'))
    auth_cache = build_auth_cache(records, auth_model)

    seed_results = []
    first_images = None
    first_trace = None
    for seed in SEEDS:
        print('RUN seed', seed, flush=True)
        rr = run_seed(seed, records, add_variants, auth_model, auth_cache)
        print(' thresholds', rr['thresholds'], 'summary', rr['summary'], flush=True)
        if first_images is None:
            first_images = rr['images']
            first_trace = rr['calibration_trace']
        seed_results.append(strip_images(rr))

    agg_seed = {}
    for k in PRIMARY:
        vals = [x['summary'][k]['mean'] for x in seed_results]
        agg_seed[k] = {'mean':float(np.mean(vals)), 'std':float(np.std(vals)),
                       'min':float(np.min(vals)), 'max':float(np.max(vals))}

    prev = json.loads((ROOT/'v518_run'/'CRCV_V518_ROBUST_CAL_RESULTS.json').read_text())['A3_rgb_ribbon']
    result = {
        'protocol':{
            'version':'V5.18.1-local-residual-suppression',
            'resolution':160,
            'models':MODELS,
            'final_test':'SEALED_NOT_READ',
            'selection':'CAL-only; suppression-only per-backbone Recall loss >= -0.5pp; full per-backbone Recall/F1/mIoU >= 0',
            'runtime_gt':'FORBIDDEN',
            'suppression':'pooled foreground-authenticity map -> very-low-auth connected residual regions inside Frozen Base',
            'recovery':'frozen V5.18 learned RGB ribbon action bank',
            'action_scorer_seeds':SEEDS,
        },
        'seed_results':seed_results,
        'seed_aggregate':agg_seed,
        'previous_v518_robust_reference':{
            'summary':prev['summary'],
            'thresholds':prev['thresholds'],
        },
    }
    checks = regression_checks(records, auth_model, auth_cache, {**seed_results[0], 'images':first_images})
    result['regression_checks'] = checks
    (OUT/'CRCV_V5181_LOCAL_RESIDUAL_RESULTS.json').write_text(json.dumps(result, indent=2))
    (OUT/'CRCV_V5181_CALIBRATION_TRACE_SEED1337.json').write_text(json.dumps(first_trace, indent=2))
    with open(OUT/'CRCV_V5181_VAL_IMAGES_SEED1337.pkl','wb') as f:
        pickle.dump(first_images, f, pickle.HIGHEST_PROTOCOL)
    (OUT/'CRCV_V5181_REGRESSION_CHECKS.json').write_text(json.dumps(checks, indent=2))
    print('DONE', json.dumps(agg_seed, indent=2), 'checks', checks, flush=True)

if __name__ == '__main__':
    main()
