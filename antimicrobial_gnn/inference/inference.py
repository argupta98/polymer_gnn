import argparse
import datetime
import time
import dgl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pandas as pd
import json
import gc

from .utils.preprocess_data import Preprocess
from .utils.infer import Inference
from .utils.load_data import get_dataloader

from pathlib import Path
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from logging import getLogger
import logging
import os
import numpy as np

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

DEBUG = os.environ.get("DEBUG", "false").lower() == "true"

# Logger with stream handler
logger = getLogger(__name__)
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))
fmt = logging.Formatter(
    "%(asctime)s %(name)s [%(levelname)s] %(filename)s:%(lineno)d %(funcName)s(): %(message)s"
)
sh = logging.StreamHandler()
sh.setFormatter(fmt)
logger.addHandler(sh)


# Set soure directories for data
SCALERS = "scalers.pkl"
DATALOADER_NUM_WORKERS = 0

# Max number of samples per save iteration
MAX_SAMPLES_PER_PROCESS = 50_000 if not DEBUG else 20_000
# Samples to send to GPU at a time
INFERENCE_BATCH_SIZE = 4096


def load_data(target_dir: str, num_gpus: int) -> list[pd.DataFrame]:
    #######
    # Load Data
    # Has to support multiple parquet files in a directory
    # concat dfs into dfs with at least max rows
    if DEBUG:
        df = pd.read_csv("shoshana_polymers/round1/uniform.csv")
        df = df.rename(columns={"ID": "poly_ID", "Unnamed: 0": "ID"})
        df["ID"] = df["ID"].map(lambda x: str(x + 1) + "_1")
        return [df]

    dfs = []
    total_rows = 0
    for parquet_file in Path(target_dir).glob("*.parquet"):
        dfs.append(
            pd.read_parquet(
                parquet_file,
                engine="pyarrow",
            )
        )
        total_rows += dfs[-1].shape[0]

    if len(dfs) == 0:
        raise ValueError("No Files Detected at Target Directory")

    gpu_dfs = np.array_split(pd.concat(dfs), num_gpus)

    assert len(gpu_dfs) == num_gpus
    assert sum(df.shape[0] for df in gpu_dfs) == total_rows

    logger.info(f"Loaded {len(gpu_dfs)} parquet files with {total_rows:,} rows")

    return gpu_dfs


def preprocessing_worker(
    df_polymers,
    SCALERS,
    MODEL,
) -> list[tuple[str, dgl.DGLGraph]]:
    run_preprocess = Preprocess(
        df_polymers=df_polymers,
        SCALERS=SCALERS,
        MODEL=hyperparameters["model"],
    )

    # Data is a list of tuples (ID, dgl_graph)
    data: list[tuple[str, dgl.DGLGraph]] = run_preprocess.get_dgl_graphs()

    return data


def prediction_pipeline(
    gpu_df: pd.DataFrame,
    gpu_idx: int,
    hyperparameters: dict,
    model_path: str,
    save_dir: str,
):
    # Setup results dump for this processing pipeline
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = save_dir / f"results_{timestamp}_{gpu_idx}.txt"

    with open(file_name, mode="w") as file:
        file.write("ID,pred\n")

    logger.info(f"Starting inference on GPU {gpu_idx}")

    start_time = time.time()

    process_preds = []
    estimated_iterations = len(gpu_df) // MAX_SAMPLES_PER_PROCESS + 1

    #########
    # Instantiate model and put on device
    logger.info(f"Running inference on GPU {gpu_idx}")
    start = time.time()
    model = Inference(
        GPU=gpu_idx,
        HYPERPARAMETERS=hyperparameters,
        MODEL_PATH=model_path,
    )
    logger.info(
        f"{time.time() - start}sec Instantiating Inference Pipeline GPU {gpu_idx}"
    )

    ########
    # Start parallelized inference
    # We HAVE to split, otherwise there isnt enough memory to process all the graphs
    # TODO: Ideally this is a all wrapped in a dataloader
    for i in range(0, len(gpu_df), MAX_SAMPLES_PER_PROCESS):
        df = gpu_df[i : i + MAX_SAMPLES_PER_PROCESS]
        logger.info(f"Starting iteration {i} with {len(df)} samples.")

        start_time = time.time()

        # Grab dgl graphs from parquet files
        logger.info(f"Grabbing dgl graphs from parquet files for GPU {gpu_idx}")

        data: list[tuple[str, dgl.DGLGraph]] = preprocessing_worker(
            df_polymers=df,
            SCALERS=SCALERS,
            MODEL=hyperparameters["model"],
        )

        elapsed_time = time.time() - start_time
        logger.info(
            f"Finished grabbing dgl graphs from parquet files for GPU {gpu_idx}"
        )
        logger.info(f"Elapsed time: {elapsed_time:.2f} seconds for GPU {gpu_idx}")

        # Create DataLoader
        infer_loader: DataLoader = get_dataloader(
            DATA=data,
            BATCH_SIZE=INFERENCE_BATCH_SIZE,
            NUM_WORKERS=DATALOADER_NUM_WORKERS,
        )

        logger.info(f"Created DataLoader for GPU {gpu_idx}")
        logger.info(
            f"Number of batches in dataloader: {len(infer_loader)} in GPU {gpu_idx}"
        )

        # Run Inference
        start = time.time()
        preds = model.predict(dataloader=infer_loader)
        logger.info(
            f"""{time.time() - start:.2f}sec for Inference for GPU {
                gpu_idx
            } iteration: {i}/{estimated_iterations}"""
        )

        # Write predictions to file
        with open(file_name, mode="a+") as file:
            for ID, pred in preds:
                file.write(f"{ID},{pred}\n")

    return process_preds


def start_inference(
    model_path: str,
    gpu_dfs: list[pd.DataFrame],
    model_hyperparameters: dict,
    results_dir: str,
):
    # Dispatch inference to each gpu
    with ProcessPoolExecutor() as executor:
        futures = []
        for i, df in enumerate(gpu_dfs):
            futures.append(
                executor.submit(
                    prediction_pipeline,
                    model_path=model_path,
                    gpu_df=df if not DEBUG else pd.concat([df] * 20),
                    gpu_idx=i,
                    hyperparameters=model_hyperparameters,
                    save_dir=results_dir,
                )
            )
            if DEBUG and i == 1:
                logger.info("DEBUG MODE")
                logger.info(f"Processed Num rows: {len(df)}")
                break

        # Wait for all futures to complete
        preds = []
        for future in as_completed(futures):
            future.result()

    return preds


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model directory",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="Path to directory with parquet files",
    )

    parser.add_argument(
        "--num_gpus",
        type=int,
        default=torch.cuda.device_count(),
        help="Number of GPUs to use",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    model_path = Path(args.model_path)
    target_dir = args.target_dir
    num_gpus = args.num_gpus

    results_dir = Path("inference_results_round_multiclass_round3/") / model_path.stem
    if results_dir.exists():
        logger.warning(f"Inference_results exists")
        shutil.rmtree(results_dir)
        logger.warning(f"inference_results removed: {results_dir.exists()}")
    results_dir.mkdir(parents=True)

    logger.info(f"Processing model: {model_path}")
    logger.info(f"Processing target_dir: {target_dir}")
    logger.info(f"Results will be saved to: {results_dir}")

    # Each list in the list contains a dataframe with rows for each GPU
    gpu_dfs: list[list[pd.DataFrame]] = load_data(
        target_dir=target_dir, num_gpus=num_gpus
    )

    logger.info(f"Loaded {len(gpu_dfs)} parquet files")

    # Run inference in multiple processes
    hyperparameters = json.load((model_path / "configure.json").open())

    _ = start_inference(
        model_path=str(model_path),
        gpu_dfs=gpu_dfs,
        model_hyperparameters=hyperparameters,
        results_dir=results_dir,
    )

    if DEBUG:
        """Check against training set"""
        results_dfs = []
        for file in Path("inference_results").glob("*.txt"):
            results_dfs.append(
                pd.read_csv("results.txt").sort_values(by=["ID"]).reset_index(drop=True)
            )
        infer = pd.concat(results_dfs)

        from_training = pd.read_csv(
            "inference/model/val_model_on_infer_set/results.txt"
        )[["ID", "y_pred"]].rename(columns={"y_pred": "pred"})

        from_training["ID"] = from_training["ID"].str.replace("SID", "") + "_1"
        from_training = from_training.sort_values(by=["ID"]).reset_index(drop=True)

        assert len(infer) == len(from_training), (
            f"Received infer: {len(infer)}\tfrom_training {len(from_training)}"
        )
        assert infer["ID"].equals(from_training["ID"])

        logger.info("--- Infer Head ---")
        logger.info(f"infer head: {infer.head()}")
        logger.info("-" * 100)
        logger.info("--- From Training Head ---")
        logger.info(f"from_training head: {from_training.head()}")
        logger.info("-" * 100)

        logger.info(
            f"Largest difference: \
            {max(infer['pred'] - from_training['pred'])}"
        )

        infer["diff"] = infer["pred"] - from_training["pred"]
        print("--- Difference Results ---")
        print(infer["diff"].value_counts().sort_index(ascending=False).head(10))
        print(infer["diff"].describe())
        fig, ax = plt.subplots(1, 1)
        infer["diff"].plot.hist(ax=ax)
        fig.savefig("tmp.png")
