"""Ray-free hyperparameter tuning driver.

Runs a single-process, serial hyperopt (TPE) search over the AMLVAE search
space. Trials are executed in-process via the :class:`Trainer` callable, so
there's no Ray raylet / dashboard / runtime-env-agent and nothing to segfault
on HPC compute nodes.

Output schema (``amlvae_tune_results.csv``) is unchanged so downstream tools
(``amlvae.utils.tune_parsing``, the ``train`` rule, etc.) keep working.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from functools import partial

import numpy as np
import pandas as pd
import torch
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, tpe

from amlvae.train.Trainer import Trainer


####################################################################################
####################################################################################
__SEARCH_SPACE__ = {
    "lr"          : hp.loguniform("lr", -12, -2),
    "l2"          : hp.choice("l2", [0, 1e-6, 1e-2]),
    "n_hidden"    : hp.choice("n_hidden", [256, 512, 1024, 2048]),
    "n_layers"    : hp.choice("n_layers", [1, 2, 3, 4]),
    "batch_size"  : hp.choice("batch_size", [32, 64, 128, 256]),
    "norm"        : hp.choice("norm", ["batch", "layer", "none"]),
    "variational" : hp.choice("variational", [True]),
    "anneal"      : hp.choice("anneal", [True, False]),
    "dropout"     : hp.uniform("dropout", 0, 0.5),
    "nonlin"      : hp.choice("nonlin", ["elu", "gelu"]),
    "beta"        : hp.choice("beta", [1.0]),
    "free_bits"   : hp.choice("free_bits", [0.0, 0.5, 2.0]),
}
####################################################################################
####################################################################################


# (csv column, sign-for-minimisation). sign = +1 means "lower is better";
# sign = -1 means "higher is better" and hyperopt will minimise -metric.
_METRIC_MODES = {
    "recon_mse": ("val_recon_mse", +1),
    "nll":      ("val_nll",       +1),
    "elbo":     ("val_elbo",      +1),
    "kld":      ("val_kld",       +1),
    "r2":       ("val_r2",        -1),
}


def get_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--proc', type=str, required=True, help='path to proc dir')
    ap.add_argument('--out', type=str, required=True, help='path to output dir')
    ap.add_argument('--target_metric', type=str, default='recon_mse',
                    choices=list(_METRIC_MODES.keys()),
                    help='target metric for tuning')
    ap.add_argument('--num_samples', type=int, default=100)
    ap.add_argument('--patience', type=int, default=1000)
    ap.add_argument('--gpus', type=int, default=1, help='ignored; kept for CLI compat')
    ap.add_argument('--cpus', type=int, default=10, help='ignored; kept for CLI compat')
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--n_latent', type=int, default=12)
    ap.add_argument('--dataset_name', type=str, default='aml')
    ap.add_argument('--seed', type=int, default=0)
    # Legacy/ignored flags so existing Snakefile invocations keep working:
    ap.add_argument('--ray_temp_dir', type=str, default=None, help=argparse.SUPPRESS)
    ap.add_argument('--ray_local_mode', type=str, default=None, help=argparse.SUPPRESS)
    return ap.parse_args()


_FAIL_LOSS = float('inf')


def _metric_short(metric_col: str) -> str:
    """`val_recon_mse` -> `recon_mse`. Prefer removeprefix over replace
    so we don't accidentally strip `val_` mid-string."""
    return metric_col.removeprefix('val_')


def _run_trial(config, trainer, metric_col, metric_sign, trial_counter, base_seed):
    idx = trial_counter['n']
    trial_counter['n'] += 1
    t0 = time.time()

    # Per-trial torch seed so the same hyperopt config + seed reproduces
    # bit-identically across runs. Each trial gets a distinct seed so we
    # don't artificially correlate trials.
    seed = (base_seed + idx) & 0x7FFFFFFF
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"\n[tune] trial {idx} (seed={seed}): {config}", flush=True)
    try:
        trainer(config)
    except Exception as exc:  # pragma: no cover
        print(f"[tune] trial {idx} FAILED after {time.time() - t0:.1f}s: {exc}",
              flush=True)
        traceback.print_exc()
        return {
            'status': STATUS_FAIL,
            'loss': _FAIL_LOSS,
            'config': config,
            'error': repr(exc),
            'wall_time_s': time.time() - t0,
        }

    metrics = trainer.best_val_metrics or {}
    val = metrics.get(_metric_short(metric_col))
    if val is None or not np.isfinite(val):
        print(f"[tune] trial {idx} produced no valid {metric_col}; "
              f"metrics={metrics}", flush=True)
        return {
            'status': STATUS_FAIL,
            'loss': _FAIL_LOSS,
            'config': config,
            'error': f'non-finite {metric_col}: {val}',
            'wall_time_s': time.time() - t0,
        }

    loss = metric_sign * float(val)
    print(f"[tune] trial {idx} done in {time.time() - t0:.1f}s: "
          f"{metric_col}={val:.4f}, n_epochs={trainer.n_epochs}",
          flush=True)

    return {
        'status': STATUS_OK,
        'loss': loss,
        'config': config,
        'best_val_metrics': metrics,
        'n_epochs': trainer.n_epochs,
        'best_epoch': trainer.best_epoch,
        'wall_time_s': time.time() - t0,
        'seed': seed,
    }


def _flatten_trials(trials, metric_col):
    rows = []
    for t in trials.trials:
        res = t.get('result') or {}
        if res.get('status') != STATUS_OK:
            continue
        cfg = res.get('config', {})
        m = res.get('best_val_metrics', {}) or {}
        rows.append({
            'val_recon_mse': m.get('recon_mse'),
            'val_nll':      m.get('nll'),
            'val_kld':      m.get('kld'),
            'val_elbo':     m.get('elbo'),
            'val_r2':       m.get('r2'),
            'batch_size':  cfg.get('batch_size'),
            'beta':        cfg.get('beta'),
            'free_bits':   cfg.get('free_bits'),
            'dropout':     cfg.get('dropout'),
            'l2':          cfg.get('l2'),
            'lr':          cfg.get('lr'),
            'n_hidden':    cfg.get('n_hidden'),
            'n_latent':    cfg.get('n_latent'),
            'n_layers':    cfg.get('n_layers'),
            'nonlin':      cfg.get('nonlin'),
            'norm':        cfg.get('norm'),
            'variational': cfg.get('variational'),
            'anneal':      cfg.get('anneal'),
            'n_epochs':    res.get('n_epochs'),
            'best_epoch':  res.get('best_epoch'),
            'wall_time_s': res.get('wall_time_s'),
            'trial_id':    t.get('tid'),
            'seed':        res.get('seed'),
            'path':        '',  # preserved for tune_parsing compatibility
        })
    df = pd.DataFrame(rows)
    if not df.empty and metric_col in df.columns:
        ascending = True  # minimised metrics all use ascending sort
        if metric_col == 'val_r2':
            ascending = False
        df = df.sort_values(by=metric_col, ascending=ascending).reset_index(drop=True)
    return df


def main():
    print()
    print('---------------------------------------------')
    print('AML-VAE: Hyperparameter tuning (ray-free)')
    print('---------------------------------------------')
    print()
    args = get_args()
    print('arguments:', args)
    if args.ray_temp_dir is not None or args.ray_local_mode is not None:
        print('[tune] note: --ray_temp_dir / --ray_local_mode are ignored '
              '(Ray is no longer used).')
    print('---------------------------------------------')

    os.makedirs(args.out, exist_ok=True)

    space = dict(__SEARCH_SPACE__)
    space['n_latent'] = hp.choice('n_latent', [args.n_latent])

    metric_col, metric_sign = _METRIC_MODES[args.target_metric]
    if args.target_metric in {'recon_mse', 'nll', 'elbo'}:
        selection_key = args.target_metric
    else:
        # `kld` and `r2` are not valid within-trial selection metrics
        # (KLD-only minimisation collapses the posterior; R2 needs to be
        # *maximised*, which the Trainer's "lower is better" loop can't do).
        # Fall back to recon_mse and warn loudly so the inconsistency is
        # visible in the run log.
        selection_key = 'recon_mse'
        print(
            f"[tune] WARNING: --target_metric={args.target_metric!r} cannot be "
            f"used as the within-trial best-epoch selector; falling back to "
            f"'recon_mse'. Each trial reports {metric_col} from the "
            f"recon_mse-best epoch, which is a slightly biased estimate.",
            flush=True,
        )

    trainer = Trainer(
        root=args.proc,
        checkpoint=False,
        report_to_tune=False,
        epochs=args.epochs,
        verbose=False,
        patience=args.patience,
        dataset_name=args.dataset_name,
        model_selection_metric=selection_key,
    )

    trials = Trials()
    trial_counter = {'n': 0}

    objective = partial(
        _run_trial,
        trainer=trainer,
        metric_col=metric_col,
        metric_sign=metric_sign,
        trial_counter=trial_counter,
        base_seed=args.seed,
    )

    try:
        fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=args.num_samples,
            trials=trials,
            rstate=np.random.default_rng(args.seed),
            show_progressbar=False,
        )
    except KeyboardInterrupt:
        print('[tune] interrupted; writing partial results.', flush=True)

    df = _flatten_trials(trials, metric_col)
    out_csv = os.path.join(args.out, 'amlvae_tune_results.csv')
    df.to_csv(out_csv, index=False)

    print(f"[tune] wrote {len(df)} successful trials to {out_csv}")
    if len(df):
        best = df.iloc[0]
        print(f"[tune] best {metric_col} = {best[metric_col]:.4f} "
              f"(trial_id={best['trial_id']})")
    print('---------------------------------------------')


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
