#!/usr/bin/env python3
"""
Query By Committee (QBC) sampling for polymers using multiclass GNN ensemble.

Algorithm:
  1. Run multiclass inference across all models in an ensemble
  2. Compute per-sample KL divergence from the ensemble mean probability distribution
  3. Select Q4 (highest-disagreement) samples; remove outliers via 1.5×IQR
  4. Optionally diversity-pick N final candidates using MaxMin from descriptor features

Data required:
  --candidate_file / --candidate_dir  : CSV or directory of parquet files, each with
                                        columns 'ID' and 'sequence'
  --models_dir                        : directory where each subdirectory is a model
                                        (must contain fullmodel.pt + configure.json)
  --scalers                           : path to scalers.pkl (antimicrobial_gnn/scalers.pkl)
  --monomer_features                  : path to rdkit_monomer_features.pkl

Optional (diversity picking):
  --descriptor_dir                    : directory of parquet files with polymer descriptors
  --unique_descriptors                : JSON with {"node": [...descriptor names...]}
  --compositions                      : parquet with polymer composition info
  --n_picks                           : number of final diverse candidates (default 40)

Usage example (full pipeline):
  python sample_qbc.py \\
      --models_dir models/ \\
      --candidate_dir no_dups_samples_one_replicate/ \\
      --scalers antimicrobial_gnn/scalers.pkl \\
      --monomer_features antimicrobial_gnn/monomer_data/rdkit_monomer_features.pkl \\
      --descriptor_dir PolymerDescriptors_Sept2024/ \\
      --unique_descriptors monomer_data/unique_descriptors.json \\
      --compositions antimicrobial_gnn/shoshana_polymers/polymer_combinations.parquet \\
      --n_picks 40 \\
      --output qbc_samples.csv

Usage example (inference + QBC only, no diversity pick):
  python sample_qbc.py \\
      --models_dir models/ \\
      --candidate_file antimicrobial_gnn/shoshana_polymers/round1/uniform.csv \\
      --scalers antimicrobial_gnn/scalers.pkl \\
      --monomer_features antimicrobial_gnn/monomer_data/rdkit_monomer_features.pkl \\
      --output qbc_samples.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import warnings
from pathlib import Path

import dgl
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from diversity_sampling_utils.picking_utils import get_max_min_picks

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(funcName)s:%(lineno)d - %(message)s",
    level=getattr(logging, log_level, logging.INFO),
)

INFERENCE_BATCH_SIZE = 4096
TOLERANCE = 1e-9


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def load_and_scale_features(scalers_path: str, monomer_features_path: str) -> dict:
    """
    Load rdkit monomer features and apply pre-fitted scalers.

    Returns a dict with keys 'node' and 'edge', each mapping
    molecule abbreviation → scaled feature vector (np.ndarray).
    """
    warnings.filterwarnings("ignore", category=UserWarning)

    with open(monomer_features_path, "rb") as f:
        raw_features: dict = pickle.load(f)  # {"node": {monomer: list}, "edge": {bond: list}}

    scalers: dict = joblib.load(scalers_path)  # {"node": MinMaxScaler, "edge": MinMaxScaler}

    scaled = {}
    for component in ["node", "edge"]:
        molecules = list(raw_features[component].keys())
        matrix = np.array([raw_features[component][m] for m in molecules], dtype=float)

        if component == "node":
            scaled_matrix = scalers[component].transform(matrix)
        else:
            # Bond features are zeroed out (ignore_bonds=True, matching original pipeline)
            scaled_matrix = np.zeros_like(matrix)

        scaled[component] = {mol: scaled_matrix[i] for i, mol in enumerate(molecules)}

    return scaled


def _build_dgl_graph(
    sequence: str,
    scaled_feats: dict,
    model_name: str,
    poly_id: str,
) -> dgl.DGLGraph:
    import re
    monomers = re.findall(r"[A-Z][^A-Z]*", sequence)

    g = dgl.graph(([], []), num_nodes=len(monomers))

    node_features = [
        torch.tensor(scaled_feats["node"][m], dtype=torch.float32)
        for m in monomers
    ]
    g.ndata["h"] = torch.stack(node_features)

    src = list(range(len(monomers) - 1))
    dst = list(range(1, len(monomers)))
    g.add_edges(src, dst)

    bond_type = "Amb" if "pep" in poly_id else "Cc"
    edge_features = [
        torch.tensor(scaled_feats["edge"][bond_type], dtype=torch.float32)
    ] * g.number_of_edges()
    g.edata["e"] = torch.stack(edge_features)

    if model_name in ("GCN", "GAT"):
        g = dgl.add_self_loop(g)

    return g


def build_graphs(
    df: pd.DataFrame,
    scaled_feats: dict,
    model_name: str,
) -> list[tuple[str, dgl.DGLGraph]]:
    """Return list of (normalized_ID, dgl_graph) for all rows in df."""
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building graphs"):
        raw_id = str(row["ID"])
        parts = raw_id.split("_")
        norm_id = "_".join(str(int(p)) for p in parts[:2]) if len(parts) >= 2 else raw_id
        g = _build_dgl_graph(row["sequence"], scaled_feats, model_name, norm_id)
        records.append((norm_id, g))
    return records


def _collate(batch):
    ids, graphs = zip(*batch)
    bg = dgl.batch(list(graphs))
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    return list(ids), bg


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    model_path: str | Path,
    candidate_df: pd.DataFrame,
    scaled_feats: dict,
    device: torch.device,
    data: list[tuple[str, dgl.DGLGraph]],
) -> pd.DataFrame:
    """
    Run multiclass inference for a single model.

    Returns a DataFrame with columns:
        ID, prob_class0, prob_class1, ..., prob_classN
    """
    model_path = Path(model_path)
    hyperparams = json.loads((model_path / "configure.json").read_text())
    model_name = hyperparams["model"]
    n_tasks = hyperparams["n_tasks"]

    loader = DataLoader(
        dataset=data,
        batch_size=INFERENCE_BATCH_SIZE,
        shuffle=False,
        collate_fn=_collate,
        num_workers=0,
    )

    logger.info(f"Loading model from {model_path} ...")
    checkpoint = torch.load(
        str(model_path / "fullmodel.pt"),
        map_location=device,
    )
    model = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model = model.to(device)
    model.eval()

    all_ids: list[str] = []
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for ids, bg in loader:
            bg = bg.to(device)
            node_feats = bg.ndata.pop("h").to(device)

            if model_name in ("GCN", "GAT"):
                logits = model(bg, node_feats)
            else:
                edge_feats = bg.edata.pop("e").to(device)
                logits = model(bg, node_feats, edge_feats)

            probs = torch.softmax(logits, dim=-1).cpu().numpy()  # (B, n_classes)
            all_ids.extend(ids)
            all_probs.append(probs)

    all_probs_arr = np.concatenate(all_probs, axis=0)  # (N, n_classes)

    result = pd.DataFrame({"ID": all_ids})
    for c in range(n_tasks):
        result[f"prob_class{c}"] = all_probs_arr[:, c]

    logger.info(f"  → {len(result)} predictions from {model_path.name}")
    return result


# ---------------------------------------------------------------------------
# QBC: multiclass KL divergence
# ---------------------------------------------------------------------------

def build_ensemble_predictions(
    model_results: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Merge per-model prediction DataFrames into a single wide DataFrame.

    Columns: ID (index), prob_class0_<model>, prob_class1_<model>, ...
    """
    dfs = []
    for model_name, df in model_results.items():
        df = df.set_index("ID")
        df.columns = [f"{col}_{model_name}" for col in df.columns]
        dfs.append(df)
    return pd.concat(dfs, axis=1)


def compute_kl_divergence(df_wide: pd.DataFrame, n_classes: int) -> pd.DataFrame:
    """
    Compute mean KL divergence (per sample) across ensemble models.

    KL(model_i || mean) = sum_k  p_k * log(p_k / q_k)
    where q_k = mean of p_k across all models.

    The final KLdiv column is the average KL across all models.

    Returns df_wide with a new 'KLdiv' column.
    """
    model_names = {
        col.split("_", 1)[1].rsplit("_", 1)[0]  # strip prob_classN_ prefix/suffix
        for col in df_wide.columns
    }
    # Better: derive model names from prob_class0_<model> columns
    model_names = [
        col[len("prob_class0_"):]
        for col in df_wide.columns
        if col.startswith("prob_class0_")
    ]

    # Compute ensemble mean probability for each class
    for c in range(n_classes):
        class_cols = [f"prob_class{c}_{m}" for m in model_names]
        mean_col = df_wide[class_cols].mean(axis=1).clip(TOLERANCE, 1 - TOLERANCE)
        df_wide[f"mean_class{c}"] = mean_col

    # Compute KL divergence for each model, then average
    kl_per_model = []
    for model in model_names:
        kl = pd.Series(0.0, index=df_wide.index)
        for c in range(n_classes):
            p = df_wide[f"prob_class{c}_{model}"].clip(TOLERANCE, 1 - TOLERANCE)
            q = df_wide[f"mean_class{c}"]
            kl = kl + p * np.log(p / q)
        kl_per_model.append(kl)

    df_wide["KLdiv"] = pd.concat(kl_per_model, axis=1).mean(axis=1)
    return df_wide


def filter_q4_no_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort by KLdiv descending, keep Q4 (top 25%), remove IQR outliers.
    """
    sorted_df = df.sort_values("KLdiv", ascending=False)

    Q1 = sorted_df["KLdiv"].quantile(0.25)
    Q3 = sorted_df["KLdiv"].quantile(0.75)
    IQR = Q3 - Q1
    upper_whisker = Q3 + 1.5 * IQR

    q4 = sorted_df[(sorted_df["KLdiv"] > Q3) & (sorted_df["KLdiv"] < upper_whisker)]
    logger.info(
        f"Q4 filter: {len(sorted_df)} → {len(q4)} samples "
        f"(KLdiv Q3={Q3:.4f}, upper_whisker={upper_whisker:.4f})"
    )
    return q4


# ---------------------------------------------------------------------------
# Diversity picking
# ---------------------------------------------------------------------------

def get_diversity_samples(features_df: pd.DataFrame, n_picks: int = 40) -> list[int]:
    """
    MaxMin diversity picking on scaled feature vectors.
    Returns list of integer indices into features_df.
    """
    scaled = StandardScaler().fit_transform(features_df.values)
    indices = get_max_min_picks(scaled, N_picks=n_picks, picker="MaxMin")
    return list(indices)


def load_descriptor_features(
    q4_ids: pd.Series,
    descriptor_dir: str,
    unique_descriptors_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load polymer descriptor parquet files and filter to q4_ids.

    Returns (target_samples_df, features_df).
    """
    descriptor_dir = Path(descriptor_dir)
    parquet_files = list(descriptor_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {descriptor_dir}")

    dfs = []
    for pf in parquet_files:
        chunk = pd.read_parquet(pf, engine="pyarrow")
        dfs.append(chunk[chunk["ID"].isin(q4_ids)])

    target_samples = pd.concat(dfs).reset_index(drop=True)

    unique_descriptors: list = json.loads(Path(unique_descriptors_path).read_text())["node"]
    features = target_samples.loc[:, target_samples.columns.isin(unique_descriptors)]

    logger.info(
        f"Loaded {len(target_samples)} descriptor rows, "
        f"{len(features.columns)} features"
    )
    return target_samples, features


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_candidates(args) -> pd.DataFrame:
    """Load candidate polymers from CSV or directory of parquet files."""
    if args.candidate_file:
        path = Path(args.candidate_file)
        if path.suffix == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_parquet(path, engine="pyarrow")
        logger.info(f"Loaded {len(df):,} candidates from {path}")
        return df

    if args.candidate_dir:
        candidate_dir = Path(args.candidate_dir)
        parquet_files = list(candidate_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files in {candidate_dir}")
        dfs = [pd.read_parquet(pf, engine="pyarrow") for pf in parquet_files]
        df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(df):,} candidates from {candidate_dir} ({len(parquet_files)} files)")
        return df

    raise ValueError("Must provide either --candidate_file or --candidate_dir")


def validate_candidate_data(df: pd.DataFrame) -> None:
    """Validate candidate data: requires >1M rows and no MIC columns."""
    n = len(df)
    logger.info(f"Candidate count: {n:,}")

    mic_cols = [c for c in df.columns if "MIC" in c.upper()]
    if mic_cols:
        logger.warning(
            f"WARNING: Candidate data contains MIC columns: {mic_cols}. "
            "These should be unlabeled candidates without MIC values."
        )
    else:
        logger.info("MIC check passed: no MIC columns found.")

    if n < 1_000_000:
        logger.warning(
            f"WARNING: Expected >1,000,000 candidates but found only {n:,}. "
            "The full unlabeled candidate pool may not be loaded."
        )
    else:
        logger.info(f"Candidate count check passed: {n:,} > 1,000,000.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="QBC sampling with multiclass GNN ensemble")

    # Required
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--candidate_file", type=str, help="CSV or parquet with 'ID' and 'sequence' columns")
    group.add_argument("--candidate_dir", type=str, help="Directory of parquet files with 'ID' and 'sequence' columns")

    parser.add_argument("--models_dir", type=str, required=True, help="Directory containing model subdirectories")
    parser.add_argument(
        "--scalers",
        type=str,
        default="antimicrobial_gnn/scalers.pkl",
        help="Path to scalers.pkl (default: antimicrobial_gnn/scalers.pkl)",
    )
    parser.add_argument(
        "--monomer_features",
        type=str,
        default="antimicrobial_gnn/monomer_data/rdkit_monomer_features.pkl",
        help="Path to rdkit_monomer_features.pkl",
    )

    # Optional - diversity picking
    parser.add_argument("--descriptor_dir", type=str, default=None, help="Directory of parquet files with polymer descriptors (for diversity picking)")
    parser.add_argument("--unique_descriptors", type=str, default="monomer_data/unique_descriptors.json", help="JSON file listing descriptor feature names to use")
    parser.add_argument("--compositions", type=str, default=None, help="Parquet file with polymer composition info")
    parser.add_argument("--n_picks", type=int, default=40, help="Number of final diverse candidates to select (default: 40)")

    # Output
    parser.add_argument("--output", type=str, default="qbc_samples.csv", help="Output CSV path")
    parser.add_argument("--save_inference", type=str, default=None, help="Directory to save per-model inference results (optional)")

    # Debug
    parser.add_argument("--debug", action="store_true", help="Debug mode: run the full pipeline on the first 1000 candidates only")

    return parser.parse_args()


def main():
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 1. Load candidates
    # ------------------------------------------------------------------
    candidates = load_candidates(args)

    if args.debug:
        logger.info("DEBUG MODE: truncating to first 1000 candidates")
        candidates = candidates.head(1000).reset_index(drop=True)

    if not args.debug:
        validate_candidate_data(candidates)

    if "ID" not in candidates.columns or "sequence" not in candidates.columns:
        raise ValueError("Candidate data must have 'ID' and 'sequence' columns")

    # ------------------------------------------------------------------
    # 2. Load scaled monomer features (shared across all models)
    # ------------------------------------------------------------------
    logger.info("Loading and scaling monomer features ...")
    scaled_feats = load_and_scale_features(args.scalers, args.monomer_features)
    logger.info(
        f"  node monomers: {len(scaled_feats['node'])}, "
        f"edge bonds: {len(scaled_feats['edge'])}"
    )

    # ------------------------------------------------------------------
    # 3. Run inference for each model
    # ------------------------------------------------------------------
    models_dir = Path(args.models_dir)
    model_dirs = sorted([d for d in models_dir.iterdir() if d.is_dir() and (d / "fullmodel.pt").exists()])

    if not model_dirs:
        raise FileNotFoundError(f"No model directories with fullmodel.pt found in {models_dir}")

    logger.info(f"Found {len(model_dirs)} models: {[d.name for d in model_dirs]}")

    # Detect n_classes from first model
    first_config = json.loads((model_dirs[0] / "configure.json").read_text())
    n_classes = first_config["n_tasks"]
    model_name = first_config["model"]
    logger.info(f"n_classes from models: {n_classes}")

    logger.info(f"Building graphs for {model_dirs[0].name} ({len(candidates)} candidates) ...")
    data = build_graphs(candidates, scaled_feats, model_name)

    model_results: dict[str, pd.DataFrame] = {}
    for model_dir in model_dirs:
        logger.info(f"Running inference: {model_dir.name}")
        result_df = run_inference(model_dir, candidates, scaled_feats, device, data)
        model_results[model_dir.name] = result_df

        if args.save_inference:
            save_dir = Path(args.save_inference)
            save_dir.mkdir(parents=True, exist_ok=True)
            result_df.to_csv(save_dir / f"{model_dir.name}_preds.csv", index=False)
            logger.info(f"  Saved inference results to {save_dir / model_dir.name}_preds.csv")

    # ------------------------------------------------------------------
    # 4. QBC: compute KL divergence and filter Q4
    # ------------------------------------------------------------------
    logger.info("Building ensemble prediction matrix ...")
    df_wide = build_ensemble_predictions(model_results)
    logger.info(f"  Ensemble shape: {df_wide.shape}")

    logger.info("Computing multiclass KL divergence ...")
    df_wide = compute_kl_divergence(df_wide, n_classes)

    logger.info(f"KLdiv stats:\n{df_wide['KLdiv'].describe()}")

    logger.info("Filtering Q4 (highest disagreement, no outliers) ...")
    q4 = filter_q4_no_outliers(df_wide)

    # Normalize IDs for downstream matching (zero-padded)
    q4_ids_normalized = q4.reset_index()["ID"].map(
        lambda x: f"{x.split('_')[0]}".zfill(15) + "_" + f"{x.split('_')[1]}".zfill(5)
        if "_" in str(x) else str(x)
    )

    # ------------------------------------------------------------------
    # 5. Diversity picking (optional — requires descriptor_dir)
    # ------------------------------------------------------------------
    if args.descriptor_dir:
        logger.info(f"Loading descriptor features from {args.descriptor_dir} ...")
        target_samples, features = load_descriptor_features(
            q4_ids_normalized, args.descriptor_dir, args.unique_descriptors
        )

        if len(target_samples) == 0:
            logger.warning("No descriptor features matched Q4 IDs — skipping diversity pick.")
            diverse_selection = target_samples
        else:
            if len(target_samples) <= args.n_picks:
                logger.warning(
                    f"Q4 has only {len(target_samples)} samples ≤ n_picks={args.n_picks}; returning all."
                )
                diverse_selection = target_samples
            else:
                logger.info(f"Running MaxMin diversity picking: {args.n_picks} from {len(target_samples)} ...")
                indices = get_diversity_samples(features, n_picks=args.n_picks)
                diverse_selection = target_samples.reset_index(drop=True).iloc[indices, :]

        selected_ids = diverse_selection["ID"].map(lambda x: int(x.split("_")[0]))
    else:
        logger.info("No --descriptor_dir provided — skipping diversity pick; outputting all Q4 samples.")
        diverse_selection = None
        selected_ids = q4_ids_normalized.map(
            lambda x: int(x.split("_")[0]) if "_" in str(x) else int(x)
        )

    # ------------------------------------------------------------------
    # 6. Merge with composition data and save output
    # ------------------------------------------------------------------
    if args.compositions and Path(args.compositions).exists():
        logger.info(f"Loading compositions from {args.compositions} ...")
        df_compositions = pd.read_parquet(args.compositions, engine="pyarrow")
        df_compositions["ID"] = df_compositions["ID"].str.replace("ID", "").map(int)
        target_compositions = df_compositions[df_compositions["ID"].isin(selected_ids)]

        q4_with_id = q4.reset_index().assign(
            ID=lambda x: x["ID"].map(lambda y: int(y.split("_")[0]) if "_" in str(y) else int(y))
        )
        df_final = target_compositions.merge(q4_with_id, on="ID")

        columns_order = ["ID", "monomers", "class_distribution", "mol_distribution", "KLdiv"]
        columns = columns_order + [c for c in df_final.columns if c not in columns_order]
        df_out = df_final[columns]
    else:
        if diverse_selection is not None:
            q4_with_id = q4.reset_index()
            df_out = diverse_selection.merge(
                q4_with_id.assign(
                    ID=lambda x: x["ID"].map(lambda y: f"{int(y.split('_')[0])}".zfill(15) + "_" + f"{int(y.split('_')[1])}".zfill(5) if "_" in str(y) else str(y))
                ),
                on="ID",
                how="left",
            )
        else:
            df_out = q4.reset_index()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df_out)} candidates to {output_path}")

    # Summary
    logger.info("=" * 60)
    logger.info("QBC Sampling Summary")
    logger.info("=" * 60)
    logger.info(f"  Total candidates:          {len(candidates):,}")
    logger.info(f"  Models used:               {len(model_dirs)} ({[d.name for d in model_dirs]})")
    logger.info(f"  Q4 (high disagreement):    {len(q4):,}")
    if diverse_selection is not None:
        logger.info(f"  After diversity pick:      {len(df_out)}")
    logger.info(f"  Output:                    {output_path}")


if __name__ == "__main__":
    main()
