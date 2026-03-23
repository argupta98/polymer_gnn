#!/usr/bin/env python3
"""
Generate polymer candidate sequences from polymer_combinations.parquet.

For each composition row (monomers + molar distribution), generates one
without-replacement sequence sample at a fixed degree of polymerization (DP).

Output: directory of parquet files, each with columns 'ID' and 'sequence'.
ID format: {numeric_poly_id}_{sample_number}  e.g. "42_1"

Usage:
    python generate_candidates.py \\
        --compositions antimicrobial_gnn/shoshana_polymers/polymer_combinations.parquet \\
        --output_dir candidate_sequences/ \\
        [--dp 70] \\
        [--n_samples 1] \\
        [--chunk_size 500000] \\
        [--n_workers 4] \\
        [--seed 42]
"""

from __future__ import annotations

import argparse
import ast
import logging
import os
import random
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(funcName)s:%(lineno)d - %(message)s",
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
)

# ---------------------------------------------------------------------------
# Sequence generation
# ---------------------------------------------------------------------------

def _saferound(values: list[float], target: int) -> list[int]:
    """
    Round a list of floats so they sum exactly to target.
    Uses floor + distribute remainders to the largest fractional parts.
    """
    floats = np.array(values, dtype=float) * target
    floors = np.floor(floats).astype(int)
    remainder = target - floors.sum()
    fracs = floats - floors
    # Assign the remainder to the indices with the largest fractional parts
    top = np.argsort(fracs)[::-1][:remainder]
    floors[top] += 1
    return floors.tolist()


def generate_sequence(monomers: list[str], mol_dist: list[float], dp: int) -> str:
    """
    Generate a single without-replacement polymer sequence.

    Computes the integer count of each monomer (via saferound), builds the
    full monomer list, shuffles, and concatenates.
    """
    counts = _saferound(mol_dist, dp)
    pool: list[str] = []
    for mon, cnt in zip(monomers, counts):
        pool.extend([mon] * int(cnt))
    np.random.shuffle(pool)
    return "".join(pool)


def _parse_list_col(val):
    """Parse a stringified Python list column back to a Python list."""
    if isinstance(val, list):
        return val
    return ast.literal_eval(val)


# ---------------------------------------------------------------------------
# Chunk worker (runs in subprocess)
# ---------------------------------------------------------------------------

def _process_chunk(args: tuple) -> pd.DataFrame:
    chunk_df, dp, n_samples, seed_offset = args

    if seed_offset is not None:
        np.random.seed(seed_offset)
        random.seed(seed_offset)

    records = []
    for row in chunk_df.itertuples(index=False):
        raw_id = str(row.ID)
        numeric_id = raw_id.replace("ID", "")

        monomers = _parse_list_col(row.monomers)
        mol_dist = _parse_list_col(row.mol_distribution)

        for s in range(1, n_samples + 1):
            seq = generate_sequence(monomers, mol_dist, dp)
            records.append({"ID": f"{numeric_id}_{s}", "sequence": seq})

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="Generate candidate polymer sequences")
    parser.add_argument(
        "--compositions",
        type=str,
        default="antimicrobial_gnn/shoshana_polymers/polymer_combinations.parquet",
        help="Path to polymer_combinations.parquet",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="candidate_sequences",
        help="Directory to write output parquet files",
    )
    parser.add_argument("--dp", type=int, default=70, help="Degree of polymerization (default 70)")
    parser.add_argument("--n_samples", type=int, default=1, help="Sequences to generate per composition row (default 1)")
    parser.add_argument("--chunk_size", type=int, default=500_000, help="Rows per output parquet file (default 500,000)")
    parser.add_argument("--n_workers", type=int, default=max(1, os.cpu_count() - 1), help="Parallel workers (default: nCPU-1)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    return parser.parse_args()


def main():
    args = get_args()

    compositions_path = Path(args.compositions)
    if not compositions_path.exists():
        raise FileNotFoundError(f"Compositions file not found: {compositions_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading compositions from {compositions_path} ...")
    df = pd.read_parquet(compositions_path, engine="pyarrow", columns=["ID", "monomers", "mol_distribution"])
    total = len(df)
    logger.info(f"  {total:,} compositions loaded")
    logger.info(f"  DP={args.dp}, n_samples={args.n_samples}, chunk_size={args.chunk_size:,}, workers={args.n_workers}")

    # Split into chunks
    chunks = [df.iloc[i : i + args.chunk_size] for i in range(0, total, args.chunk_size)]
    logger.info(f"  {len(chunks)} chunks to process")

    chunk_args = [
        (chunk.reset_index(drop=True), args.dp, args.n_samples, args.seed + idx)
        for idx, chunk in enumerate(chunks)
    ]

    t0 = time.time()
    total_written = 0

    with Pool(processes=args.n_workers) as pool:
        for file_idx, result_df in enumerate(pool.imap(_process_chunk, chunk_args)):
            out_path = output_dir / f"candidates_{file_idx:04d}.parquet"
            result_df.to_parquet(out_path, engine="pyarrow", index=False)
            total_written += len(result_df)
            elapsed = time.time() - t0
            rate = total_written / elapsed if elapsed > 0 else 0
            logger.info(
                f"  [{file_idx+1}/{len(chunks)}] wrote {len(result_df):,} rows → {out_path.name} "
                f"| total {total_written:,} | {rate:,.0f} seq/s"
            )

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"Done. {total_written:,} sequences written to {output_dir}/")
    logger.info(f"Elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"Files: {len(list(output_dir.glob('*.parquet')))}")


if __name__ == "__main__":
    main()
