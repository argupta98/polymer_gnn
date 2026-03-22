from pathlib import Path
from sklearn.preprocessing import StandardScaler
from diversity_sampling_utils.picking_utils import get_max_min_picks
import pandas as pd
import numpy as np
import logging
import os

# logger with stream handler
# with fmt that includes function, line no and regular message
logger = logging.getLogger(__name__)
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
level_map = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING}
logger.setLevel(level_map[log_level])
fmt = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
)
handler = logging.StreamHandler()
handler.setFormatter(fmt)
logger.addHandler(handler)


def get_diversity_samples(df: pd.DataFrame, N_picks: int = 20):
    """
    Get diversity sampled dataframe
    1. Scale features of sampling space


    """
    scaled_sampled_df: np.ndarray = StandardScaler().fit_transform(
        df
    )

    diversity_samples_indices = get_max_min_picks(
        scaled_sampled_df, N_picks=N_picks, picker="MaxMin"
    )

    return list(diversity_samples_indices)


if __name__ == "__main__":
    root_dir = Path(__file__).parent

    # Load target samples
    target_samples = pd.read_csv(
        root_dir / "analysis" / "samples_with_highest_entropy.csv"
    )

    logger.info(f"Target samples: {target_samples.shape}")
    logger.info(f"Target samples:\n{target_samples.head()}")
    columns_str = "\n".join(target_samples.columns)
    logger.info(f"{columns_str}")
