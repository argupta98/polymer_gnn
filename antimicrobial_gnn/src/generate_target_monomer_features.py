"""
Pregenerate RDKit for the monomers being used by Appel Lab
"""

import pandas as pd
import numpy as np
import json
import pickle

from rdkit.Chem import Descriptors
from rdkit import Chem


def get_monomer_rdkit_features(MON_SMILES: str, BOND_SMILES: str, DESCRIPTORS: str):
    """
    Get RDKit features for monomers and bonds.

    :param MON_SMILES: Path to monomer SMILES file
    :param BOND_SMILES: Path to bond SMILES file
    :param DESCRIPTORS: Path to descriptors file
    :return: unscaled_feats: Dictionary of monomer and bond features
    """

    df_monomer_smiles = pd.read_csv(MON_SMILES)
    df_bonds_smiles = pd.read_csv(BOND_SMILES)

    df = pd.concat(
        [df_monomer_smiles.assign(type="node"), df_bonds_smiles.assign(type="edge")]
    ).reset_index(drop=True)

    descriptors_to_keep = json.load(open(DESCRIPTORS))

    # Generate features
    df["features"] = df["SMILES"].apply(
        lambda x: Descriptors.CalcMolDescriptors(
            Chem.MolFromSmiles(x), missingVal=-9999, silent=True
        )
    )

    # Convert features to numpy array
    df["np_array_features"] = df.apply(
        lambda x: np.array(
            [x["features"][key] for key in descriptors_to_keep[x["type"]]]
        ),
        axis=1,
    )

    # Node dict
    node_dict = (
        df[df.type == "node"].set_index("Molecule")["np_array_features"].to_dict()
    )

    # Edge dict
    edge_dict = (
        df[df.type == "edge"].set_index("Molecule")["np_array_features"].to_dict()
    )

    # Create features dicts with node or edge as keys
    rdkit_feats = {
        "node": node_dict,
        "edge": edge_dict,
    }

    return rdkit_feats


if __name__ == "__main__":
    MON_SMILES_POLY = "tables_poly/SMILES_polymers_monomer.txt"
    BOND_SMILES_POLY = "tables_poly/SMILES_polymers_bond.txt"
    DESCRIPTORS = "unique_descriptors.json"

    feats = get_monomer_rdkit_features(MON_SMILES_POLY, BOND_SMILES_POLY, DESCRIPTORS)

    pickle.dump(feats, open("monomer_data/rdkit_features_synthetic_monomers.pkl", "wb"))
