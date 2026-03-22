# get all txt files into a dataframe from inference_results dir
# use pathlib
# txt files are comma separated
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import logging

import os

# logger with stream handler
# with fmt that includes function, line no and regular message
logger = logging.getLogger(__name__)
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
level_map = {"DEBUG": logging.DEBUG,
             "INFO": logging.INFO, "WARNING": logging.WARNING}
logger.setLevel(level_map[log_level])
fmt = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
)
handler = logging.StreamHandler()
handler.setFormatter(fmt)
logger.addHandler(handler)


def get_target_polymers_with_features(
    target_dir: str, target_ids: pd.Series
) -> pd.DataFrame:
    #######
    # Load Data
    # Has to support multiple parquet files in a directory
    logger.debug(f"Directory contents: {list(Path(target_dir).iterdir())}")

    no_parquet_files = len(list(Path(target_dir).glob("*.parquet")))
    if no_parquet_files == 0:
        raise ValueError("No parquet files found in target directory")

    dfs = []
    for parquet_file in Path(target_dir).glob("*.parquet"):
        logger.debug(f"Loading {parquet_file}")
        dfs.append(
            pd.read_parquet(
                parquet_file,
                engine="pyarrow",
            )[lambda x: x.ID.isin(target_ids)]
        )

    if len(dfs) == 0:
        raise ValueError("No rows matched target IDs")

    return pd.concat(dfs)


def get_inference_results(inference_results_dir) -> pd.DataFrame:
    """
    Load individual inference results from a directory

    :param inference_results_dir: directory containing inference results
    :return: dataframe with inference results, preds are NN results after sigmoid
    """
    # get all txt files into a dataframe from inference_results dir
    inference_results_dir = Path(inference_results_dir)
    if not inference_results_dir.is_dir():
        raise FileNotFoundError("Inference directory does not exist.")
    
    txt_files = list(inference_results_dir.glob("*.txt"))
    dfs = []
    for txt_file in txt_files:
        dfs.append(pd.read_csv(txt_file))

    df = pd.concat(dfs)
    if "ID" not in df or "pred" not in df:
        raise ValueError(
            "ID and prediction columns not found in inference results")

    return df


def show_exploratory_analysis(df, save_dir, model_name = ""):
    save_dir = Path(save_dir)
    # Show histrogram of predictions
    fig, ax = plt.subplots(1, 1)
    df["pred"].hist(ax=ax, bins=4)
    ax.set_title("Predictions histrogram")
    fig.savefig(save_dir / f"pred_hist_{model_name}.png")

    # Show binary prediction
    logger.info("-" * 100)
    logger.info(f'Binary pred value counts\n{df["binary_prediction"].value_counts()}\n{df["binary_prediction"].value_counts(normalize=True)}')
    logger.info("-" * 100)

    # Compute entropy and sort by entropy
    # Entropy is for uncertiaty sampling
    df.sort_values(by="entropy", ascending=False, inplace=True)
    entropy_df = df[["ID", "entropy", "pred"]]

    logger.info("-" * 100)
    logger.info("Entropy")
    logger.info(f"Entropy df head:\n{entropy_df.head()}")
    logger.info(f"Entropy df describe:\n{entropy_df.describe()}")
    logger.info("-" * 100)

    fig, ax = plt.subplots(1, 1)
    entropy_df["entropy"].hist(ax=ax, bins=4)
    ax.set_title("Entropy distribution")
    fig.savefig(save_dir / f"entropy_hist_{model_name}.png")


def select_most_uncertain(df, save_dir, tolerance=1e-6):
    """
    Select most uncertain samples
    :param df: dataframe with inference results
    :return: dataframe with most uncertain samples
    """
    save_dir = Path(save_dir)
    uncertain_samples = df[df["entropy"] > (1.0 - tolerance)]

    logger.info("-" * 100)
    logger.info("Entropy == 1")
    logger.info(uncertain_samples.head())
    logger.info(uncertain_samples.describe())
    logger.info("-" * 100)

    return uncertain_samples


if __name__ == "__main__":
    root_dir = Path(__file__).parent

    analysis_dir = root_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_results_dir", type=str, default=None)
    parser.add_argument(
        "--sampling_space_with_features_dir",
        type=str,
        default="PolymerDescriptors_Sept2024/",
    )
    args = parser.parse_args()

    inference_results_dir = args.inference_results_dir
    sampling_space_with_features_dir = args.sampling_space_with_features_dir
    inference_results_name = Path(inference_results_dir).name

    if not sampling_space_with_features_dir.exists():
        raise ValueError("Sampling space directory not found")

    ######
    # Process inference results

    df = get_inference_results(inference_results_dir)

    # Add analysis columns
    df["binary_pred"] = df["pred"].map(lambda x: 1 if x > 0.5 else 0)
    df["entropy"] = df["pred"].map(
        lambda x: -1
        * (x * np.log2(x) + (1 - x) * np.log2(1 - x) if x > 0 and x < 1 else 0)
    )

    # Show exploratory analysis
    show_exploratory_analysis(df)

    # select those with entropy == 1
    uncertain_samples: pd.DataFrame = select_most_uncertain(df)
    # Format IDs with fixed id format
    ids_uncertain_samples = uncertain_samples["ID"].map(
        lambda x: f"{x.split('_')[0]}".zfill(
            15) + "_" + f"{x.split('_')[1]}".zfill(5)
    )

    logger.debug(uncertain_samples.head())

    ##########################
    # Load target samples with formulation and features from disk
    target_samples = get_target_polymers_with_features(
        target_dir=sampling_space_with_features_dir,
        target_ids=ids_uncertain_samples,
    )

    logger.debug(
        f"IDs in target samples: {target_samples.ID.sort_values().head()}")
    logger.debug(
        f"IDs in uncertain samples: {ids_uncertain_samples.sort_values().head()}"
    )
    logger.info(f"Samples in uncertain samples: {len(ids_uncertain_samples)}")
    logger.info(f"Samples in target samples: {len(target_samples)}")
    assert len(target_samples) == len(ids_uncertain_samples)

    target_samples.to_csv(analysis_dir / "samples_with_highest_entropy.csv")
