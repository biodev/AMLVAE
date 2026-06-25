"""Run ARACNe-AP (threshold -> bootstraps -> consolidate) on bulk expression."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd

DEFAULT_JAR = Path(
    "/home/exacloud/gscratch/mcweeney_lab/evans/ARACNe-AP/dist/aracne.jar"
)


def _run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}):\n"
            f"{' '.join(cmd)}\n\nstdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )
    return result


def write_aracne_inputs(
    expr_df: pd.DataFrame,
    tfs: pd.Series | list[str],
    input_dir: Path | str,
) -> tuple[Path, Path]:
    """Write ARACNe-AP expression matrix and TF list.

    Parameters
    ----------
    expr_df
        Genes x samples expression matrix (index = gene symbols).
    tfs
        Regulator gene symbols (one per line in output).
    input_dir
        Directory for ``matrix.txt`` and ``tfs.txt``.

    Returns
    -------
    (expr_path, tf_path)
    """
    input_dir = Path(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    expr_path = input_dir / "matrix.txt"
    tf_path = input_dir / "tfs.txt"

    expr_out = expr_df.copy().fillna(0.0)
    expr_out.insert(0, "gene", expr_out.index.astype(str))
    expr_out.to_csv(expr_path, sep="\t", index=False)

    regulators = pd.Series(tfs).dropna().astype(str).unique()
    pd.Series(regulators).to_csv(tf_path, index=False, header=False)

    return expr_path, tf_path


def calculate_threshold(
    jar: Path | str,
    expr_path: Path | str,
    tf_path: Path | str,
    out_dir: Path | str,
    pvalue: float = 1e-8,
    seed: int = 1,
    xmx: str = "16G",
) -> None:
    """Compute MI threshold (--calculateThreshold)."""
    cmd = [
        "java",
        f"-Xmx{xmx}",
        "-jar",
        str(jar),
        "-e",
        str(expr_path),
        "-o",
        str(out_dir),
        "--tfs",
        str(tf_path),
        "--pvalue",
        str(pvalue),
        "--seed",
        str(seed),
        "--calculateThreshold",
    ]
    _run_cmd(cmd)


def run_bootstrap(
    jar: Path | str,
    expr_path: Path | str,
    tf_path: Path | str,
    out_dir: Path | str,
    pvalue: float = 1e-8,
    seed: int = 1,
    threads: int = 8,
    xmx: str = "16G",
) -> None:
    """Run one ARACNe bootstrap (seed = bootstrap index)."""
    cmd = [
        "java",
        f"-Xmx{xmx}",
        "-jar",
        str(jar),
        "-e",
        str(expr_path),
        "-o",
        str(out_dir),
        "--tfs",
        str(tf_path),
        "--pvalue",
        str(pvalue),
        "--seed",
        str(seed),
        "--threads",
        str(threads),
    ]
    _run_cmd(cmd)


def consolidate(
    jar: Path | str,
    out_dir: Path | str,
    nobonferroni: bool = False,
    xmx: str = "16G",
) -> Path:
    """Consolidate bootstrap networks into ``network.txt``."""
    cmd = [
        "java",
        f"-Xmx{xmx}",
        "-jar",
        str(jar),
        "-o",
        str(out_dir),
        "--consolidate",
    ]
    if nobonferroni:
        cmd.append("--nobonferroni")
    _run_cmd(cmd)
    network_path = Path(out_dir) / "network.txt"
    if not network_path.exists():
        raise FileNotFoundError(f"Expected consolidated network at {network_path}")
    return network_path


def run_aracne_pipeline(
    expr_df: pd.DataFrame,
    tfs: pd.Series | list[str],
    out_dir: Path | str,
    jar: Path | str = DEFAULT_JAR,
    n_bootstraps: int = 100,
    pvalue: float = 1e-8,
    threads: int = 8,
    xmx: str = "16G",
    force: bool = False,
    nobonferroni: bool = False,
) -> Path:
    """Run full ARACNe-AP pipeline: threshold, bootstraps, consolidate.

    Returns path to ``network.txt``.
    """
    out_dir = Path(out_dir)
    input_dir = out_dir / "inputs"
    network_path = out_dir / "network.txt"

    if network_path.exists() and not force:
        print(f"Using existing network: {network_path}")
        return network_path

    if out_dir.exists() and force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    expr_path, tf_path = write_aracne_inputs(expr_df, tfs, input_dir)

    print("Step 1/3: calculating MI threshold...")
    calculate_threshold(jar, expr_path, tf_path, out_dir, pvalue=pvalue, xmx=xmx)

    print(f"Step 2/3: running {n_bootstraps} bootstraps...")
    for i in range(1, n_bootstraps + 1):
        print(f"  bootstrap {i}/{n_bootstraps}", end="\r")
        run_bootstrap(
            jar,
            expr_path,
            tf_path,
            out_dir,
            pvalue=pvalue,
            seed=i,
            threads=threads,
            xmx=xmx,
        )
    print()

    print("Step 3/3: consolidating bootstraps...")
    return consolidate(jar, out_dir, nobonferroni=nobonferroni, xmx=xmx)
