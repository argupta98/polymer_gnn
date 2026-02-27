import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

def preprocess_dataframe(df: pd.DataFrame, class_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a combined peptide/polymer dataframe into the three inputs
    expected by split_dataset.

    Returns:
        peptides:       one row per peptide, with columns [ID, class_col, ...]
        polymer_groups: one row per polymer group, with columns [group_id, class_col]
        polymers:       one row per polymer, with columns [ID, group_id, class_col, ...]
    """
    
    # Drop rows with missing class labels before splitting
    n_before = len(df)
    df = df.dropna(subset=[class_col]).copy()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"Warning: dropped {n_dropped} rows with missing '{class_col}' values.")

    # ── Detect type from ID ───────────────────────────────────────────────────
    is_polymer = df["ID"].str.match(r"^polyID\d+_S\d+$")
    is_peptide = df["ID"].str.match(r"^pepID\d+$")

    # Sanity check for unrecognized ID formats
    unrecognized = df[~is_polymer & ~is_peptide]["ID"].tolist()
    if unrecognized:
        raise ValueError(f"Unrecognized ID format for: {unrecognized[:5]}")

    # ── Peptides ──────────────────────────────────────────────────────────────
    peptides = df[is_peptide].copy().reset_index(drop=True)

    # ── Polymers ──────────────────────────────────────────────────────────────
    polymers = df[is_polymer].copy().reset_index(drop=True)

    # Extract group_id from e.g. "polyID14_S94" → "polyID14"
    polymers["group_id"] = polymers["ID"].str.extract(r"^(polyID\d+)_S\d+$")

    # ── Polymer groups (one row per group, class must be unique per group) ────
    # Validate: all members of a group share the same class
    group_class_counts = polymers.groupby("group_id")[class_col].nunique()
    inconsistent = group_class_counts[group_class_counts > 1].index.tolist()
    if inconsistent:
        raise ValueError(f"Groups with inconsistent classes: {inconsistent}")

    polymer_groups = (
        polymers.groupby("group_id")[class_col]
        .first()
        .reset_index()
    )

    return peptides, polymer_groups, polymers


def polymers_for_groups(polymers: pd.DataFrame, group_ids) -> pd.DataFrame:
    """Expand a set of group_ids to their individual polymer rows."""
    return polymers[polymers["group_id"].isin(group_ids)].reset_index(drop=True)


def _stratified_split(df, class_col, test_size, random_state):
    """Split a dataframe, preserving class proportions."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    idx_train, idx_test = next(sss.split(df, df[class_col]))
    return df.iloc[idx_train].reset_index(drop=True), df.iloc[idx_test].reset_index(drop=True)


def _split_mixed(peptides, polymer_groups, polymers, class_col, val_size, test_size, random_state):
    """
    Stratified split of both peptides and polymer groups, then recombine.
    Polymer groups are the unit of splitting (not individual polymers).
    """
    # Adjust val_size to be relative to the remaining data after the test split
    # e.g. for 70/15/15: cut off 15% as test, then cut off 0.15/0.85 ≈ 17.6% of remainder as val
    relative_val = val_size / (1 - test_size) if val_size > 0 else 0

    # --- Polymer groups (split at group level) ---
    grp_train, grp_test = _stratified_split(polymer_groups, class_col, test_size, random_state)
    if val_size > 0:
        grp_train, grp_val = _stratified_split(grp_train, class_col, relative_val, random_state)

    # --- Peptides (split at individual level) ---
    pep_train, pep_test = _stratified_split(peptides, class_col, test_size, random_state)
    if val_size > 0:
        pep_train, pep_val = _stratified_split(pep_train, class_col, relative_val, random_state)

    # --- Expand polymer groups → individual polymers and combine with peptides ---
    splits = {
        "train": pd.concat([pep_train, polymers_for_groups(polymers, grp_train["group_id"])], ignore_index=True),
        "test":  pd.concat([pep_test,  polymers_for_groups(polymers, grp_test["group_id"])],  ignore_index=True),
    }
    if val_size > 0:
        splits["val"] = pd.concat([pep_val, polymers_for_groups(polymers, grp_val["group_id"])], ignore_index=True)

    return splits


def _split_peptide_train_polymer_test(
    peptides, polymer_groups, polymers, class_col, val_size, random_state
):
    """
    Test set = all polymers.
    Train (+ optional val) = all peptides, stratified by class.
    """
    # All polymers go to test
    test = polymers.copy()

    # Peptides split into train (+ optional val)
    if val_size > 0:
        train, val = _stratified_split(peptides, class_col, val_size, random_state)
        return {"train": train, "val": val, "test": test}
    else:
        return {"train": peptides.copy(), "test": test}


def split_dataset(
    db_file: str,
    class_col: str,
    val_size: float = 0.15,
    test_size: float = 0.15,
    mixed: bool = True,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Split peptides and polymers into train/val/test sets.

    Args:
        df:             pandas dataframe of polymer+peptide database
        class_col:      column name to use for stratification
        val_size:       fraction of data for validation (0.0 to skip, giving train/test only)
        test_size:      fraction of data for test
        mixed:          if True, peptides and polymers are distributed across all splits;
                        if False, train(/val) = peptides only, test = polymers only
        random_state:   random seed for reproducibility

    Returns:
        dict with keys "train", "test", and optionally "val"
    """
    
    df = pd.read_csv(db_file)
    peptides, polymer_groups, polymers = preprocess_dataframe(df=df, class_col=class_col)
    
    if mixed:
        return _split_mixed(
            peptides, polymer_groups, polymers, class_col, val_size, test_size, random_state
        )
    else:
        return _split_peptide_train_polymer_test(
            peptides, polymer_groups, polymers, class_col, val_size, random_state
        )