"""
Create the large sampling space of polymer sequence from source compositions.
Provided compositions of (monA:0.2, monB:0.6, monC:0.2) generate sequences
of monAmonAmonBmonCmonCmonC

As of September 17th, we only replicate 1 sequence per composition.
- memory issues make larger replicates challenging
"""

from polymerization import sample
from pathlib import Path
import pandas as pd
import logging
from logging import getLogger
import os
from concurrent.futures import ProcessPoolExecutor
import time
import uuid
import json

DEBUG = os.environ.get("DEBUG", "false").lower() == "true"

logger = getLogger(__name__)
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
stream_handler = logging.StreamHandler()
# Format with line numbers
# Format with function calling name
fmt = logging.Formatter(
    "%(asctime)s %(lineno)d %(levelname)s - %(funcName)s - %(message)s"
)
stream_handler.setFormatter(fmt)
logger.addHandler(stream_handler)

root_dir = Path(__file__).parent.parent


def generate_samples_worker(
    dfs: list[pd.DataFrame], number_of_replicates=10, save_dir=None
):
    """
    Generates samples for a single polymer composition
    Args:
        dfs: list of pd.Dataframe each with unique monomers formulation
        number_of_replicates: number of replicates to generate
        save_dir: directory to save samples to

    Writes generated samples to disk in the format:
    000000000001.txt
    N.txt

    Each line in the file is a sample
    """

    def generate_samples(row):
        ID = row["ID"]
        monomers = row["monomers"]
        mol_dist = row["mol_distribution"]
        logger.debug(f"ID {ID}")
        logger.debug("monomers", monomers)
        logger.debug("mol_dist", mol_dist, type(mol_dist))

        samples = sample(
            monomers=monomers,
            mol_dist=mol_dist,
            DP=70,
            sampling_method="wo_replacement",
            n=number_of_replicates,
            batch=True,
            encoded=False,
        ).samples
        assert len(samples) == number_of_replicates

        return samples

    number_of_rows = sum(len(d) for d in dfs)

    # Create the sequence for each polymer composition
    # Iterate over each polymer formulation
    for i, df in enumerate(dfs):
        assert df.monomers.astype(str).nunique() == 1

        sequences = df.apply(generate_samples, axis=1)
        df["sequence"] = sequences
        df = df.explode("sequence")

        # Create a unique ID for each sample
        # Each duplicate polymer formulation has several compositions
        # We use the duplicate indices created by .explode to create a unique ID
        df = df.assign(
            ID=lambda x: x.ID.str.replace("ID", "").str.zfill(15)
            + "_"
            + x.groupby(level=0).cumcount().add(1).astype(str).str.zfill(5),
        )
        logger.debug("head")
        logger.debug(df.head(10))
        logger.debug("tail")
        logger.debug(df.tail(10))

        dfs[i] = df  # Contains each composition with added replicates

    assert (
        sum(len(d) for d in dfs) == number_of_rows * number_of_replicates
    ), f"Number of rows mismatch: {sum(len(d) for d_ in dfs for d in d_)} != {number_of_rows}"

    ##############################
    # Save these samples to disk, split by MAXROWS
    ##############################

    logger.info("-" * 100)
    logger.info(f"Number of samples: {df.shape[0]}")

    sample_path = root_dir / save_dir
    sample_path.mkdir(parents=True, exist_ok=True)
    group_idx_path = sample_path / "group_map.jsonl"

    # Save each 5 million rows to disk
    # Only split if there are 5 million more
    # If there is remainder keep it in the last file
    row_target = 5_000_000 if not DEBUG else 2500

    # Constrain to 5 million rows per file
    total_rows = sum(len(d) for d in dfs)
    dfs_to_save = []
    groups = []
    current_rows = 0
    saved_rows = 0
    for i, df in enumerate(dfs):
        groups.extend(df.ID.tolist())
        df.drop(columns=["monomers"])
        dfs_to_save.append(df)
        current_rows += df.shape[0]

        # Save the dataframe when we reach the row target or when we are at the
        # last group
        if (dfs_to_save and current_rows >= row_target) or i == len(dfs) - 1:
            df_save = pd.concat(dfs_to_save, ignore_index=True)

            save_id = uuid.uuid4().hex + f"_rows_{len(df_save)}"
            save_parquet_path = sample_path / f"{save_id}.parquet"

            df_save.to_parquet(save_parquet_path, index=False)

            groups = {"filename": save_id, "groups": groups}
            with group_idx_path.open("a") as f:
                f.write(json.dumps(groups) + "\n")

            saved_rows += len(df_save)

            dfs_to_save = []
            groups = []
            current_rows = 0

    assert saved_rows == total_rows, f"{saved_rows} != {total_rows}"


def split_df(df_targets: pd.DataFrame, max_workers: int) -> list[list[pd.DataFrame]]:
    """
    Splits a dataframe into a list of dataframes with a total number of rows
    split equally among the processes.
    Will groupby on unique monomers formulation and then create list of dataframes
    that equally split the dataframe into the number of processes.
    Will not split a group, so number of rows per processor may not be equal.

    If groups are too large we may want to consider splitting it (TODO).

    Args:
        df_targets: pandas dataframe with columns ID, monomers, mol_distribution
        max_workers: number of processes to split into
    Returns:
        list of list of dataframes
        each element in the list is a list of dataframes each for a single process
    """

    samples_per_process = len(df_targets) // max_workers
    logger.info(f"Splitting into {max_workers} processes")
    logger.info(f"minimum_samples_per_process: {samples_per_process}")

    # Groupby on unique monomers formulation and then create list of dataframes
    df_targets["grouper_col"] = df_targets["monomers"].astype(str)
    df_by_formulation = df_targets.groupby("grouper_col")
    max_length = max(len(group) for name, group in df_by_formulation)
    min_length = min(len(group) for name, group in df_by_formulation)
    logger.info(f"Max length: {max_length}")
    logger.info(f"Min length: {min_length}")

    # Split into manageable number of rows per job
    df_targets_split: list[pd.DataFrame] = []
    job_group: list[pd.DataFrame] = []
    no_samples = 0
    no_groups = 0
    for name, group in df_by_formulation:
        # Add unique formulation to the job group
        no_samples += len(group)
        no_groups += 1

        group = group.drop(columns=["grouper_col"])
        job_group.append(group)

        # if we have enough samples to fill a job
        # then add the job to the list of jobs
        if no_samples >= samples_per_process:
            df_targets_split.append(job_group)
            job_group = []
            no_samples = 0

    if job_group:
        df_targets_split.append(job_group)

    # Double check that the number of samples is correct
    total_samples = sum([len(d) for df in df_targets_split for d in df])
    assert total_samples == df_targets.shape[0]
    assert len(df_targets_split) <= max_workers

    logger.info(f"{len(df_targets_split)} jobs to process")
    logger.info(f"{no_groups} groups in dataset")

    return df_targets_split


if __name__ == "__main__":
    source_data_path = root_dir / "shoshana_polymers" / "polymer_combinations.parquet"

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--number_of_replicates",
        type=int,
        default=1,
        help="Number of replicates to generate per polymer composition",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="samples_1replicate",
        help="Directory to save samples to",
    )

    args = parser.parse_args()

    NUMBER_OF_REPLICATES = args.number_of_replicates
    SAVE_DIR = args.save_dir
    if Path(SAVE_DIR).exists():
        yn = input(f"{SAVE_DIR} already exists. Continue? [y/n]")
        if yn != "y":
            exit()

    logger.info("Starting Polymer Sequence generation")
    logger.info(f"Number of replicates: {NUMBER_OF_REPLICATES}")

    ##########################
    # Prepare Data
    logger.info(f"Reading data from {source_data_path}")
    TARGET_COLUMNS = ["ID", "mol_distribution", "monomers"]
    df_targets = pd.read_parquet(
        source_data_path,
        columns=TARGET_COLUMNS,
    )
    if DEBUG:
        df_targets = df_targets.head(100_000)
    total_samples = df_targets.shape[0]

    logger.info(
        f"Number of samples generated: { total_samples * NUMBER_OF_REPLICATES}")

    # check if we want to continue
    yn = input("Continue? [y/n]")
    if yn != "y":
        exit()

    logger.info(f"Read {df_targets.shape[0]} rows")
    logger.info(f"Selected df with columns {df_targets.columns}")

    ##########################
    # Parallelize the process

    # Keep on cpu available
    max_workers = os.cpu_count() - 1 if not DEBUG else 1

    # Split df into the number of processes
    df_targets_split = split_df(df_targets, max_workers)
    total_rows_after_split = sum(len(d) for d_ in df_targets_split for d in d_)

    logger.info(
        f"split_df returned {len(df_targets_split)} jobs, each with {[len(d) for d in df_targets_split]} groups, and {total_rows_after_split} total rows"
    )
    assert (
        total_rows_after_split == total_samples
    ), f"{total_rows_after_split} != {total_samples}"

    # Kick off individual processes
    start = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for dfs in df_targets_split:
            logger.info(
                f"Processing {len(dfs)} groups with a total of {sum(len(d) for d in dfs)} samples"
            )
            futures.append(
                executor.submit(
                    generate_samples_worker,
                    dfs,
                    NUMBER_OF_REPLICATES,
                    SAVE_DIR,
                )
            )

        # Wait for all futures to complete
        for future in futures:
            future.result()

    end = time.time()
    elapsed = end - start
    logger.info(f"Elapsed time: {elapsed}")
    logger.info(f"Time per sample: {elapsed / df_targets.shape[0]}")
    logger.info(
        f"Estimated time for 7Million samples: {elapsed / total_samples * 7e6 / 3600} hours"
    )
