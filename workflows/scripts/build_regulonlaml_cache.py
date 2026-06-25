#!/usr/bin/env python3
"""One-time build of regulonlaml parquet cache for pyviper (requires rdata)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import rdata
from pyviper import Interactome

RDA_URL = (
    "https://github.com/federicogiorgi/aracne.networks/raw/master/data/regulonlaml.rda"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args()

    out_dir = args.repo_root / "data" / "regulons"
    out_dir.mkdir(parents=True, exist_ok=True)
    rda_path = out_dir / "regulonlaml.rda"
    cache_path = out_dir / "regulonlaml_human_symbol.parquet.gzip"

    if not rda_path.exists():
        import urllib.request

        print(f"Downloading {RDA_URL}")
        urllib.request.urlretrieve(RDA_URL, rda_path)

    parsed = rdata.parser.parse_file(str(rda_path))
    regulon = rdata.conversion.convert(parsed)["regulonlaml"]
    rows: list[tuple[str, str, float, float]] = []
    for reg_id, entry in regulon.items():
        tfmode = entry["tfmode"]
        for tgt, mor, likelihood in zip(
            tfmode.coords["dim_0"].values, tfmode.values, entry["likelihood"]
        ):
            rows.append((str(reg_id), str(tgt), float(mor), float(likelihood)))

    interactome = Interactome(
        "regulonlaml", pd.DataFrame(rows, columns=["regulator", "target", "mor", "likelihood"])
    )
    interactome.translate_regulators("human_symbol", verbose=False)
    interactome.translate_targets("human_symbol", verbose=False)
    interactome.net_table.to_parquet(cache_path, compression="gzip", index=False)
    print(f"Wrote {cache_path} ({interactome.net_table.shape[0]:,} edges)")


if __name__ == "__main__":
    main()
