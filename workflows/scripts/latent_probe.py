"""Supervised probe: predict clinical targets from VAE μ vs PCA (same dimension).

Uses outer train / test from proc partitions; inner CV for hyperparameters on train only.
Probe settings come from config YAML (``latent_probe:``). Clinical rows are keyed by ``--clin_id_column``;
that column must match sample ids in expression and in ``*_partitions.pt``.

Regression targets are z-scored (``StandardScaler`` on *y*, train-only per inner-CV fold and on full outer train for refit); reported metrics use the original *y* scale.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import warnings

import numpy as np
import yaml
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    r2_score,
)
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR

from amlvae.data.clin_cond import load_clin_cond
from amlvae.models.VAE import VAE

_CLASSIFICATION = "classification"
_REGRESSION = "regression"

_SUPPORTED_MODELS = frozenset(
    {"logistic", "svc", "svr", "rf", "gbrt", "ridge", "elasticnet"}
)
_CLASS_METRICS = frozenset({"auroc", "accuracy", "f1_macro"})
_REG_METRICS = frozenset({"rmse", "mae", "r2"})


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--proc", required=True, help="proc dir with expr + partitions")
    p.add_argument("--model_path", required=True, help="trained VAE checkpoint")
    p.add_argument("--dataset", default="mds")
    p.add_argument("--out", required=True, help="output CSV path")
    p.add_argument("--clin_path", required=True)
    p.add_argument(
        "--clin_id_column",
        default="MLL ID",
        help="Clinical table column whose values equal expression index / partition train|val|test ids",
    )
    p.add_argument(
        "--config_yaml",
        required=True,
        help="Workflow config.yaml; uses the latent_probe: mapping (targets, inner_cv_folds, …)",
    )
    return p.parse_args()


def load_latent_probe_cfg(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        full = yaml.safe_load(f)
    if not isinstance(full, dict):
        raise ValueError(f"config_yaml must parse to a mapping, got {type(full)}")
    block = full.get("latent_probe")
    if not isinstance(block, dict):
        raise ValueError("config_yaml must contain a 'latent_probe:' mapping")
    return block


def load_expression(proc: str, dataset: str):
    data = pd.read_csv(f"{proc}/{dataset}_expr.csv")
    data = data.set_index(data.columns[0])
    parts = torch.load(f"{proc}/{dataset}_partitions.pt", weights_only=False)
    train_ids = np.asarray(parts["train_ids"])
    val_ids = np.asarray(parts["val_ids"])
    test_ids = np.asarray(parts["test_ids"])
    X_train = torch.tensor(data.loc[train_ids].values, dtype=torch.float32)
    X_val = torch.tensor(data.loc[val_ids].values, dtype=torch.float32)
    X_test = torch.tensor(data.loc[test_ids].values, dtype=torch.float32)
    return data, X_train, X_val, X_test, train_ids, val_ids, test_ids


def load_clinical(path: str, id_col: str) -> pd.DataFrame:
    if path.endswith(".xlsx"):
        clin = pd.read_excel(path, sheet_name=0)
    elif path.endswith(".csv"):
        clin = pd.read_csv(path)
    else:
        raise ValueError("clin_path must be .xlsx or .csv")
    if id_col not in clin.columns:
        raise KeyError(f"clin_id_column {id_col!r} not in clinical columns: {list(clin.columns)}")
    clin = clin.drop_duplicates(subset=[id_col], keep="first")
    return clin.set_index(id_col)


def _vae_mu(
    model: VAE,
    X: torch.Tensor,
    device: torch.device,
    C: torch.Tensor | None = None,
) -> np.ndarray:
    with torch.no_grad():
        x_cond = C.to(device) if C is not None else None
        return model.encode(X.to(device), x_cond=x_cond)[0].cpu().numpy()


def fit_transform_pca(
    X_train: np.ndarray,
    X_val: np.ndarray | None,
    X_test: np.ndarray,
    n_components: int,
    scope: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return PCA latent for train and test rows (fit scope matches eval.py when train_only)."""
    if scope == "train_only":
        pca = PCA(n_components=n_components)
        pca.fit(X_train)
        return pca.transform(X_train), pca.transform(X_test)
    if X_val is None:
        raise ValueError("train_val scope requires validation expression matrix")
    pca = PCA(n_components=n_components)
    pca.fit(np.vstack([X_train, X_val]))
    return pca.transform(X_train), pca.transform(X_test)


def _needs_proba(metric: str, other_metrics: list[str]) -> bool:
    if metric == "auroc":
        return True
    return any(m == "auroc" for m in other_metrics)


def _build_estimator_and_grid(
    model_name: str,
    task: str,
    random_state: int,
    need_proba: bool,
) -> tuple[object, dict]:
    if model_name not in _SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model_name!r}; use {_SUPPORTED_MODELS}")

    if task == _CLASSIFICATION:
        if model_name == "logistic":
            est = LogisticRegression(
                max_iter=10000,
                random_state=random_state,
                class_weight="balanced",
            )
            grid = {
                "C": [0.01, 0.1, 1.0, 10.0],
                "penalty": ["l2"],
                "solver": ["lbfgs"],
            }
            return est, grid
        if model_name == "svc":
            est = SVC(
                kernel="rbf",
                class_weight="balanced",
                random_state=random_state,
                probability=need_proba,
            )
            grid = {
                "C": [0.1, 1.0, 10.0],
                "gamma": ["scale", 0.01, 0.001],
            }
            return est, grid
        if model_name == "rf":
            est = RandomForestClassifier(
                random_state=random_state, class_weight="balanced"
            )
            grid = {"n_estimators": [100, 200], "max_depth": [None, 12]}
            return est, grid
        if model_name == "gbrt":
            est = GradientBoostingClassifier(random_state=random_state)
            grid = {"learning_rate": [0.05, 0.1], "max_depth": [3, 5], "n_estimators": [100]}
            return est, grid
        raise ValueError(f"Model {model_name} not implemented for classification")

    # regression
    if model_name == "ridge":
        est = Ridge()
        grid = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
        return est, grid
    if model_name == "elasticnet":
        est = ElasticNet(random_state=random_state, max_iter=10000)
        grid = {
            "alpha": [0.01, 0.1, 1.0],
            "l1_ratio": [0.2, 0.5, 0.8],
        }
        return est, grid
    if model_name == "rf":
        est = RandomForestRegressor(random_state=random_state)
        grid = {"n_estimators": [100, 200], "max_depth": [None, 12]}
        return est, grid
    if model_name == "gbrt":
        est = GradientBoostingRegressor(random_state=random_state)
        grid = {"learning_rate": [0.05, 0.1], "max_depth": [3, 5], "n_estimators": [100]}
        return est, grid
    if model_name == "svr":
        est = SVR(kernel="rbf")
        grid = {"C": [0.1, 1.0, 10.0], "gamma": ["scale", 0.01]}
        return est, grid
    if model_name == "logistic":
        raise ValueError("logistic is not valid for regression; use ridge, elasticnet, rf, gbrt, or svr")

    raise ValueError(f"Model {model_name} not implemented for regression")


def _gridsearch_scoring(metric: str, task: str, n_classes: int) -> str:
    if task == _CLASSIFICATION:
        if metric not in _CLASS_METRICS:
            raise ValueError(f"Unknown classification metric {metric!r}")
        if metric == "auroc":
            return "roc_auc_ovr" if n_classes > 2 else "roc_auc"
        if metric == "accuracy":
            return "accuracy"
        if metric == "f1_macro":
            return "f1_macro"
    else:
        if metric not in _REG_METRICS:
            raise ValueError(f"Unknown regression metric {metric!r}")
        if metric == "rmse":
            return "neg_root_mean_squared_error"
        if metric == "mae":
            return "neg_mean_absolute_error"
        if metric == "r2":
            return "r2"
    raise ValueError(metric)


def _wrap_regression_estimator(est) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(regressor=est, transformer=StandardScaler())


def _prefix_param_grid(grid: dict, prefix: str) -> dict:
    return {f"{prefix}{k}": v for k, v in grid.items()}


def _cv_mean_in_report_space(gs: GridSearchCV, metric: str, task: str) -> float:
    """Convert GridSearchCV best negative scores to user-facing scale (positive RMSE, etc.)."""
    best = gs.cv_results_["mean_test_score"][gs.best_index_]
    if task == _REGRESSION and metric == "rmse":
        return float(-best)
    if task == _REGRESSION and metric == "mae":
        return float(-best)
    return float(best)


def _proba_class_labels(est) -> np.ndarray:
    """Classes in predictor order (same columns as predict_proba); supports Pipeline."""
    if hasattr(est, "classes_"):
        return np.asarray(est.classes_)
    if hasattr(est, "named_steps"):
        for _, step in reversed(list(est.named_steps.items())):
            if hasattr(step, "classes_"):
                return np.asarray(step.classes_)
    raise ValueError("Estimator has no classes_; cannot score AUROC")


def _score_test_primary(
    est, X_te: np.ndarray, y_te: np.ndarray, metric: str, task: str, n_classes: int
) -> float:
    if task == _CLASSIFICATION:
        if metric == "accuracy":
            return float(accuracy_score(y_te, est.predict(X_te)))
        if metric == "f1_macro":
            return float(f1_score(y_te, est.predict(X_te), average="macro"))
        if metric == "auroc":
            if n_classes < 2:
                return float("nan")
            if not hasattr(est, "predict_proba"):
                raise ValueError(
                    "metric auroc requires predict_proba; use svc with probability or logistic/rf/gbrt"
                )
            if len(np.unique(y_te)) < 2:
                return float("nan")
            proba = est.predict_proba(X_te)
            labels = _proba_class_labels(est)
            if labels.shape[0] == 2:
                return float(roc_auc_score(y_te, proba[:, 1]))
            return float(
                roc_auc_score(
                    y_te,
                    proba,
                    labels=labels,
                    multi_class="ovr",
                    average="weighted",
                )
            )
    else:
        pred = est.predict(X_te)
        if metric == "rmse":
            return float(np.sqrt(mean_squared_error(y_te, pred)))
        if metric == "mae":
            return float(mean_absolute_error(y_te, pred))
        if metric == "r2":
            return float(r2_score(y_te, pred))
    raise ValueError(metric)


def _compute_test_metric(
    est,
    X_te: np.ndarray,
    y_te: np.ndarray,
    token: str,
    task: str,
    n_classes: int,
) -> float:
    """Shared evaluator for primary and other_metrics tokens."""
    if task == _CLASSIFICATION:
        if token == "accuracy":
            return float(accuracy_score(y_te, est.predict(X_te)))
        if token == "f1_macro":
            return float(f1_score(y_te, est.predict(X_te), average="macro"))
        if token == "auroc":
            return _score_test_primary(est, X_te, y_te, "auroc", task, n_classes)
    else:
        pred = est.predict(X_te)
        if token == "rmse":
            return float(np.sqrt(mean_squared_error(y_te, pred)))
        if token == "mae":
            return float(mean_absolute_error(y_te, pred))
        if token == "r2":
            return float(r2_score(y_te, pred))
    raise ValueError(f"Cannot compute test metric {token!r} for task {task}")


def _safe_stratified_kfold(n_splits: int, n_samples: int, y: np.ndarray, random_state: int):
    """Stratified K-fold for classification only; n_splits must be <= smallest class count."""
    _, counts = np.unique(y, return_counts=True)
    min_c = int(counts.min()) if len(counts) else 0
    if min_c < 2:
        raise ValueError(
            "Not enough samples per class for stratified CV (need at least 2 samples in "
            "every retained class). Drop rare labels upstream or lower inner_cv_folds."
        )
    k = min(n_splits, min_c, max(2, n_samples // 2))
    if k < 2:
        raise ValueError(
            f"inner_cv_folds too large for {n_samples} samples / class balance (min class {min_c})"
        )
    if k < n_splits:
        warnings.warn(f"Reducing inner_cv_folds from {n_splits} to {k} for stratified CV")
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)


def _final_estimator_name(best) -> str:
    if hasattr(best, "regressor_"):
        return type(best.regressor_).__name__
    return type(best).__name__


def run_probe_for_target(
    Z_train: np.ndarray,
    Z_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    spec: dict,
    other_metrics: list[str],
    inner_cv_folds: int,
    random_state: int,
    n_jobs: int,
) -> dict:
    task = str(spec["type"]).strip().lower()
    model_name = str(spec["model"]).strip().lower()
    metric = str(spec["metric"]).strip().lower()

    if task not in (_CLASSIFICATION, _REGRESSION):
        raise ValueError(f"type must be classification or regression, got {task!r}")

    n_classes = len(np.unique(y_train)) if task == _CLASSIFICATION else 0
    if task == _CLASSIFICATION and n_classes < 2:
        raise ValueError("Classification requires at least 2 classes in training after filtering")

    need_proba = _needs_proba(metric, other_metrics)
    if task == _CLASSIFICATION and need_proba and model_name == "svc":
        need_proba = True

    est, grid = _build_estimator_and_grid(model_name, task, random_state, need_proba)
    if task == _REGRESSION:
        est = _wrap_regression_estimator(est)
        grid = _prefix_param_grid(grid, "regressor__")

    if task == _CLASSIFICATION:
        if metric not in _CLASS_METRICS:
            raise ValueError(f"metric {metric!r} invalid for classification")
        cv = _safe_stratified_kfold(
            inner_cv_folds, len(y_train), y_train, random_state
        )
    else:
        if metric not in _REG_METRICS:
            raise ValueError(f"metric {metric!r} invalid for regression")
        # Regression uses KFold only (never stratified by label).
        n = len(y_train)
        k = min(max(2, inner_cv_folds), n)
        if k < inner_cv_folds:
            warnings.warn(
                f"Using inner_cv_folds={k} for regression (n_train={n}; KFold requires n_splits <= n)."
            )
        cv = KFold(n_splits=k, shuffle=True, random_state=random_state)

    scoring = _gridsearch_scoring(metric, task, n_classes)
    gs = GridSearchCV(
        est,
        grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
    )
    gs.fit(Z_train, y_train)

    cv_mean = _cv_mean_in_report_space(gs, metric, task)
    test_primary = _score_test_primary(
        gs.best_estimator_, Z_test, y_test, metric, task, n_classes
    )

    row = {
        "cv_score_mean": cv_mean,
        "test_score": test_primary,
        "best_params": json.dumps(gs.best_params_),
        "fitted_estimator": _final_estimator_name(gs.best_estimator_),
    }

    om = [m.strip().lower() for m in other_metrics if m.strip().lower() != metric]
    for tok in om:
        if task == _CLASSIFICATION and tok not in _CLASS_METRICS:
            continue
        if task == _REGRESSION and tok not in _REG_METRICS:
            continue
        try:
            row[f"test_{tok}"] = _compute_test_metric(
                gs.best_estimator_, Z_test, y_test, tok, task, n_classes
            )
        except ValueError:
            continue
    return row


def main():
    print()
    print("---------------------------------------------")
    print("Latent probe: VAE vs PCA supervised comparison")
    print("---------------------------------------------")
    args = get_args()
    print(args)
    print("---------------------------------------------")

    lp_cfg = load_latent_probe_cfg(args.config_yaml)

    targets_spec = lp_cfg.get("targets") or {}
    if not isinstance(targets_spec, dict) or not targets_spec:
        raise ValueError("config latent_probe.targets must be a non-empty mapping")

    other_metrics = lp_cfg.get("other_metrics") or []
    if not isinstance(other_metrics, list):
        raise ValueError("config latent_probe.other_metrics must be a list")

    inner_cv_folds = int(lp_cfg.get("inner_cv_folds", 5))
    random_state = int(lp_cfg.get("random_state", 0))
    pca_fit_scope = lp_cfg.get("pca_fit_scope", "train_only")
    if pca_fit_scope not in ("train_only", "train_val"):
        raise ValueError("pca_fit_scope must be train_only or train_val")
    n_jobs = int(lp_cfg.get("n_jobs", 1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, X_train, X_val, X_test, train_ids, val_ids, test_ids = load_expression(
        args.proc, args.dataset
    )

    model = VAE.load(args.model_path, map_location=device).to(device).eval()
    n_latent = model.latent_dim

    C_train = C_test = None
    if model.conditional_dim > 0:
        C_train = load_clin_cond(args.proc, args.dataset, train_ids)
        C_test = load_clin_cond(args.proc, args.dataset, test_ids)

    X_tr_np = X_train.reshape(len(X_train), -1).numpy()
    X_va_np = X_val.reshape(len(X_val), -1).numpy()
    X_te_np = X_test.reshape(len(X_test), -1).numpy()

    Z_vae_train = _vae_mu(model, X_train, device, C_train)
    Z_vae_test = _vae_mu(model, X_test, device, C_test)

    pca_arg_val = X_va_np if pca_fit_scope == "train_val" else None
    Z_pca_train, Z_pca_test = fit_transform_pca(
        X_tr_np,
        pca_arg_val,
        X_te_np,
        n_components=n_latent,
        scope=pca_fit_scope,
    )

    clin = load_clinical(args.clin_path, args.clin_id_column)

    rows = []
    for target_name, spec in targets_spec.items():
        if not isinstance(spec, dict):
            warnings.warn(f"Target {target_name!r}: spec must be a mapping; skipping")
            continue
        for key in ("type", "model", "metric"):
            if key not in spec:
                raise KeyError(f"Target {target_name!r} missing required key {key!r}")
        spec_type = str(spec["type"]).strip().lower()
        spec_model = str(spec["model"]).strip().lower()
        spec_metric = str(spec["metric"]).strip().lower()
        if target_name not in clin.columns:
            warnings.warn(f"Target {target_name!r} not in clinical table; skipping")
            continue
        for representation, Z_tr, Z_te in (
            ("vae", Z_vae_train, Z_vae_test),
            ("pca", Z_pca_train, Z_pca_test),
        ):
            task = spec_type
            if task == _CLASSIFICATION:
                s_tr = clin.reindex(train_ids)[target_name]
                mask_tr = s_tr.notna() & (s_tr.astype(str).str.strip() != "")
                s_te = clin.reindex(test_ids)[target_name]
                mask_te = s_te.notna() & (s_te.astype(str).str.strip() != "")

                y_tr_raw = s_tr[mask_tr].astype(str).values
                Z_tr_sub = Z_tr[np.flatnonzero(mask_tr.values)]
                # Drop classes too small for stratified K-fold (each class needs >= n_splits samples).
                min_class_n = max(2, int(inner_cv_folds))
                cnt = Counter(y_tr_raw)
                valid_labels = {lab for lab, c in cnt.items() if c >= min_class_n}
                if len(valid_labels) < 2:
                    warnings.warn(
                        f"Skipping {target_name!r} / {representation}: fewer than 2 classes with "
                        f">= {min_class_n} training samples (inner_cv_folds={inner_cv_folds}). "
                        f"Drop or merge rare labels, or lower inner_cv_folds."
                    )
                    continue
                keep_mc = np.array([lab in valid_labels for lab in y_tr_raw])
                n_drop = int((~keep_mc).sum())
                if n_drop:
                    examples = sorted(set(y_tr_raw[~keep_mc]))[:5]
                    warnings.warn(
                        f"{target_name} / {representation}: dropped {n_drop} train rows in classes "
                        f"with <{min_class_n} samples (e.g. {examples})."
                    )
                Z_tr_sub = Z_tr_sub[keep_mc]
                y_tr_raw = y_tr_raw[keep_mc]

                y_te_raw = s_te[mask_te].astype(str).values

                le = LabelEncoder()
                y_train = le.fit_transform(y_tr_raw)
                known = set(le.classes_)
                te_keep = np.array([lab in known for lab in y_te_raw], dtype=bool)
                if te_keep.sum() == 0:
                    warnings.warn(
                        f"Skipping {target_name!r} / {representation}: no test labels seen in train"
                    )
                    continue
                y_test = le.transform(y_te_raw[te_keep])
                Z_te_sub = Z_te[np.flatnonzero(mask_te.values)[te_keep]]
            else:
                y_num_tr = pd.to_numeric(
                    clin.reindex(train_ids)[target_name], errors="coerce"
                )
                y_num_te = pd.to_numeric(
                    clin.reindex(test_ids)[target_name], errors="coerce"
                )
                mask_tr = y_num_tr.notna().values
                mask_te = y_num_te.notna().values
                Z_tr_sub = Z_tr[mask_tr]
                Z_te_sub = Z_te[mask_te]
                y_train = y_num_tr[mask_tr].astype(float).values
                y_test = y_num_te[mask_te].astype(float).values

            # Minimum data: 2 train rows to run CV; >=1 test for evaluation.
            if len(y_train) < 2 or len(y_test) < 1:
                warnings.warn(
                    f"Skipping {target_name!r} / {representation}: need n_train>=2 and n_test>=1; "
                    f"got n_train={len(y_train)}, n_test={len(y_test)}. "
                    f"For regression: ensure clin_id_column matches expression / partition ids ({args.clin_id_column!r}) "
                    f"and values are numeric. For classification: check non-empty labels."
                )
                continue
            try:
                probe = run_probe_for_target(
                    Z_tr_sub,
                    Z_te_sub,
                    y_train,
                    y_test,
                    {
                        "type": spec_type,
                        "model": spec_model,
                        "metric": spec_metric,
                    },
                    other_metrics,
                    inner_cv_folds,
                    random_state,
                    n_jobs,
                )
            except Exception as e:
                warnings.warn(f"{target_name} / {representation}: {e}")
                continue
            row = {
                "representation": representation,
                "target": target_name,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "n_latent": n_latent,
                "pca_fit_scope": pca_fit_scope,
                **probe,
            }
            row["type"] = spec_type
            row["model"] = spec_model
            row["metric"] = spec_metric
            rows.append(row)

    out_df = pd.DataFrame(rows)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(out_df)
    print("---------------------------------------------")
    print("latent_probe complete.")
    print()


if __name__ == "__main__":
    main()
