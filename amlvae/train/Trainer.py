import math
import os
import tempfile

import pandas as pd
import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset

from amlvae.data.clin_cond import load_clin_cond
from amlvae.models.VAE import VAE
from amlvae.train.losses import adversarial_mse

# Ray is only used when the caller explicitly opts into `report_to_tune=True`.
# Import lazily so the Trainer works on nodes where Ray is broken.
try:
    from ray import tune as _ray_tune  # noqa: F401
    try:
        from ray.tune import Checkpoint as _RayCheckpoint
    except Exception:
        from ray.train import Checkpoint as _RayCheckpoint
    _RAY_AVAILABLE = True
except Exception:
    _ray_tune = None
    _RayCheckpoint = None
    _RAY_AVAILABLE = False


_VALID_SELECTION_METRICS = {"recon_mse", "nll", "elbo"}


class Trainer:
    """VAE trainer. Supports hyperparameter sweeps via Ray Tune.

    Parameters
    ----------
    root : str
        Directory containing `<dataset_name>_expr.csv` and
        `<dataset_name>_partitions.pt` (both produced by `proc.py`).
    dataset_name : str
        Dataset prefix used in the proc outputs.
    use_conditional : bool
        If True, concatenate clinical covariates into encoder/decoder.
    use_adversarial : bool
        If True, train GRL adversarial head with MSE on covariates.
    lambda_adv : float
        Weight on adversarial MSE loss.
    checkpoint : bool
        If True, save per-epoch Ray-style checkpoints during `__call__`.
        Only consulted when `report_to_tune=True`.
    report_to_tune : bool
        If True, call `ray.tune.report(...)` each epoch. Default False; the
        standard flow is now a Ray-free hyperopt loop that reads the best
        metrics off `self` after `__call__` returns.
    log_every : int
        Emit a Ray checkpoint every `log_every` epochs when both
        `checkpoint=True` and `report_to_tune=True`.
    epochs : int
        Hard upper bound on training epochs.
    verbose : bool
        If True, print per-epoch metrics.
    patience : int
        Early-stopping patience on the selection metric.
    return_best_model : bool
        If True, `__call__` returns the best (by selection metric) model.
    model_selection_metric : {"recon_mse", "nll", "elbo"}
        Which validation metric to track for best-model selection.
    """

    def __init__(
        self,
        root,
        dataset_name="aml",
        use_conditional=False,
        use_adversarial=False,
        lambda_adv=1.0,
        checkpoint=False,
        report_to_tune=False,
        log_every=250,
        epochs=500,
        verbose=False,
        patience=100,
        return_best_model=False,
        model_selection_metric="recon_mse",
        metrics_log_path=None,
    ):
        if model_selection_metric not in _VALID_SELECTION_METRICS:
            raise ValueError(
                f"model_selection_metric must be one of {_VALID_SELECTION_METRICS}; "
                f"got {model_selection_metric!r}"
            )

        self.root = root
        self.dataset_name = dataset_name
        self.use_conditional = use_conditional
        self.use_adversarial = use_adversarial
        self.lambda_adv = float(lambda_adv)

        data = pd.read_csv(f"{root}/{dataset_name}_expr.csv")
        data = data.set_index(data.columns[0])
        partitions = torch.load(
            f"{root}/{dataset_name}_partitions.pt", weights_only=False
        )
        self._partitions = partitions

        self.X_train = torch.tensor(
            data.loc[partitions["train_ids"], :].values, dtype=torch.float32
        )
        self.X_val = torch.tensor(
            data.loc[partitions["val_ids"], :].values, dtype=torch.float32
        )
        self.X_test = torch.tensor(
            data.loc[partitions["test_ids"], :].values, dtype=torch.float32
        )

        self.C_train = None
        self.C_val = None
        self.C_test = None
        self._clin_dim = 0

        if use_conditional or use_adversarial:
            self.C_train = load_clin_cond(
                root, dataset_name, partitions["train_ids"]
            )
            self.C_val = load_clin_cond(
                root, dataset_name, partitions["val_ids"]
            )
            self.C_test = load_clin_cond(
                root, dataset_name, partitions["test_ids"]
            )
            self._clin_dim = self.C_train.shape[1]

        self.checkpoint = checkpoint
        self.report_to_tune = report_to_tune
        self.epochs = epochs
        self.log_every = log_every
        self.verbose = verbose
        self.patience = patience
        self.return_best_model = return_best_model
        self.model_selection_metric = model_selection_metric
        self.metrics_log_path = metrics_log_path

        # populated by __call__; read by the caller (e.g. hyperopt loop)
        self.best_val_metrics = None
        self.best_epoch = None
        self.n_epochs = None

    def _x_cond_for(self, partition):
        if not self.use_conditional:
            return None
        if partition == "train":
            return self.C_train
        if partition == "val":
            return self.C_val
        if partition == "test":
            return self.C_test
        raise ValueError(f"Unknown partition: {partition}")

    def _y_adv_for(self, partition):
        if not self.use_adversarial:
            return None
        return self._x_cond_for(partition)

    def train_epoch(self, model, optim, loader, device, beta, free_bits=0.0):
        model.train()
        use_clin = self.use_conditional or self.use_adversarial

        for batch in loader:
            if use_clin:
                x_batch, c_batch = batch[0], batch[1]
                c_batch = c_batch.to(device, non_blocking=True)
            else:
                x_batch = batch[0]

            x_batch = x_batch.to(device, non_blocking=True)
            x_cond = c_batch if self.use_conditional else None
            y_adv = c_batch if self.use_adversarial else None

            optim.zero_grad()
            out = model(x_batch, x_cond=x_cond)
            losses = model.loss(x_batch, beta=beta, free_bits=free_bits, **out)
            total = losses["loss"]
            if self.use_adversarial:
                total = total + self.lambda_adv * adversarial_mse(
                    out["adv_pred"], y_adv
                )
            total.backward()
            optim.step()

    @torch.no_grad()
    def eval(self, model, device, partition="val", batch_size=1024):
        if partition == "train":
            X = self.X_train
        elif partition == "val":
            X = self.X_val
        elif partition == "test":
            X = self.X_test
        else:
            raise ValueError(f"Unknown partition: {partition}")

        C = self._x_cond_for(partition)
        C_adv = self._y_adv_for(partition)

        model.eval()

        nll_sum = 0.0
        kld_sum = 0.0
        mse_sum = 0.0
        adv_sum = 0.0
        n = 0
        xhat_parts = []
        for start in range(0, X.size(0), batch_size):
            x = X[start : start + batch_size].to(device, non_blocking=True)
            x_cond = None
            y_adv = None
            if C is not None:
                x_cond = C[start : start + batch_size].to(
                    device, non_blocking=True
                )
            if C_adv is not None:
                y_adv = C_adv[start : start + batch_size].to(
                    device, non_blocking=True
                )

            out = model(x, x_cond=x_cond)
            losses = model.loss(x, beta=1.0, free_bits=0.0, **out)
            bs = x.size(0)
            nll_sum += losses["nll"].item() * bs
            kld_sum += losses["kld_raw"].item() * bs
            mse_sum += losses["recon_mse"].item() * bs
            if self.use_adversarial:
                adv_sum += adversarial_mse(out["adv_pred"], y_adv).item() * bs
            n += bs
            xhat_parts.append(out["xhat"].detach().cpu())

        nll = nll_sum / max(n, 1)
        kld = kld_sum / max(n, 1)
        recon_mse = mse_sum / max(n, 1)
        elbo = nll + kld
        adv_loss = adv_sum / max(n, 1) if self.use_adversarial else 0.0

        xhat = torch.cat(xhat_parts, dim=0).numpy()
        r2 = r2_score(X.cpu().numpy(), xhat, multioutput="variance_weighted")

        metrics = {
            "recon_mse": recon_mse,
            "nll": nll,
            "kld": kld,
            "elbo": elbo,
            "r2": r2,
        }
        if self.use_adversarial:
            metrics["adv_loss"] = adv_loss
        return metrics

    def __call__(self, config):
        conditional_dim = self._clin_dim if self.use_conditional else 0
        advesarial_dim = self._clin_dim if self.use_adversarial else 0

        model_kwargs = {
            "input_dim": self.X_train.size(1),
            "hidden_dim": config["n_hidden"],
            "n_layers": config["n_layers"],
            "latent_dim": config["n_latent"],
            "norm": config["norm"],
            "variational": config["variational"],
            "dropout": config["dropout"],
            "nonlin": config["nonlin"],
            "conditional_dim": conditional_dim,
            "advesarial_dim": advesarial_dim,
            "grl_alpha": float(config.get("grl_alpha", 1.0)),
        }

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = VAE(**model_kwargs).to(device)
        optim = torch.optim.Adam(
            model.parameters(), lr=config["lr"], weight_decay=config["l2"]
        )

        if self.use_conditional or self.use_adversarial:
            dataset = TensorDataset(self.X_train, self.C_train)
        else:
            dataset = TensorDataset(self.X_train)
        loader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            drop_last=False,
        )

        free_bits = float(config.get("free_bits", 0.0))
        selection_key = self.model_selection_metric

        best_metric = float("inf")
        patience_count = 0
        best_model = None
        best_val_metrics = None
        best_epoch = -1
        metrics_log = []

        T = max(int(self.epochs * 0.75), 1)
        for epoch in range(self.epochs):
            if config.get("anneal", False):
                if epoch < T:
                    fraction = epoch / T
                    beta = 0.5 * (1 - math.cos(fraction * math.pi)) * config["beta"]
                else:
                    beta = config["beta"]
            else:
                beta = config["beta"]

            self.train_epoch(model, optim, loader, device, beta, free_bits=free_bits)
            val = self.eval(model, device, partition="val")

            metrics_log.append({
                "epoch": epoch,
                "beta": beta,
                **{f"val_{k}": v for k, v in val.items()},
            })

            current = val[selection_key]
            if current < best_metric:
                best_metric = current
                best_model = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                best_val_metrics = dict(val)
                best_epoch = epoch
                patience_count = 0
            else:
                patience_count += 1

            if self.report_to_tune:
                if not _RAY_AVAILABLE:
                    raise RuntimeError(
                        "report_to_tune=True requires `ray` to be importable."
                    )
                with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                    ckpt = None
                    if self.checkpoint and (epoch + 1) % self.log_every == 0:
                        torch.save(
                            model.state_dict(),
                            os.path.join(temp_checkpoint_dir, "model.pth"),
                        )
                        ckpt = _RayCheckpoint.from_directory(temp_checkpoint_dir)
                    report = {
                        "val_recon_mse": val["recon_mse"],
                        "val_nll": val["nll"],
                        "val_kld": val["kld"],
                        "val_elbo": val["elbo"],
                        "val_r2": val["r2"],
                        "beta": beta,
                    }
                    if self.use_adversarial:
                        report["val_adv_loss"] = val["adv_loss"]
                    _ray_tune.report(report, checkpoint=ckpt)

            if self.verbose:
                msg = (
                    f"epoch {epoch}: recon_mse {val['recon_mse']:.4f} "
                    f"nll {val['nll']:.2f} kld {val['kld']:.2f} "
                    f"elbo {val['elbo']:.2f} r2 {val['r2']:.3f} beta {beta:.2e}"
                )
                if self.use_adversarial:
                    msg += f" adv {val['adv_loss']:.4f}"
                print(msg, end="\r")

            if patience_count > self.patience:
                break

        if best_model is not None:
            model.load_state_dict(best_model)

        if self.metrics_log_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(self.metrics_log_path)), exist_ok=True)
            pd.DataFrame(metrics_log).to_csv(self.metrics_log_path, index=False)

        self.best_val_metrics = best_val_metrics
        self.best_epoch = best_epoch
        self.n_epochs = epoch + 1 if metrics_log else 0

        if self.return_best_model:
            return model
        return
