# Script that takes a directory, loads all
#
#
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_splits", type=int, default=1)
    parser.add_argument("--target_dir", type=str, required=True)
    args = parser.parse_args()

    NUM_SPLITS = args.num_splits
    TARGET_DIR = args.target_dir

    dfs = []
    total_rows = 0
    for parquet_file in Path(TARGET_DIR).glob("*.parquet"):
        dfs.append(
            pd.read_parquet(
                parquet_file,
                engine="pyarrow",
            )
        )
        total_rows += dfs[-1].shape[0]

    if len(dfs) == 0:
        raise ValueError("No Files Detected at Target Directory")

    gpu_dfs = np.array_split(pd.concat(dfs), NUM_SPLITS)

    assert len(gpu_dfs) == NUM_SPLITS
    assert sum(df.shape[0] for df in gpu_dfs) == total_rows

    logger.info(f"Loaded {len(gpu_dfs)} parquet files with {total_rows:,} rows")

    print(gpu_dfs[0].head())
    columns_str = "\n".join(gpu_dfs[0].columns)
    print(columns_str)
