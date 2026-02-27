import pandas as pd
import json
from rdkit.Chem import rdMolDescriptors, Descriptors
from rdkit import Chem
import numpy as np
import re
import dgl
import torch

import joblib

from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler

from utils.util_functions import get_unscaled_features


def scale_features(type_, molecules, unscaled_features):
    '''
    Fits MinMax scaler to train set and applies to validation and test sets.
    '''
    
    mol_split = re.findall('[A-Z][^A-Z]*', molecules)
    unique_mols = list(set(mol_split))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = {}

    if len(unique_mols) <= 2:

        # Zero out features if there is no diversity in nodes/edges present (i.e., <= 2)
        # Otherwise, min_max gives one-hot encoding
        for key, val in unscaled_features[type_].items():
            num_features = len(val)
            scaled_features[key] = np.zeros(num_features)

    else:

        # Fit transform features appearing in train set
        train_features_stack_unscaled = np.vstack([unscaled_features[type_][mol] for mol in unique_mols])
        train_features_scaled = scaler.fit_transform(train_features_stack_unscaled)
        scaled_train_features = dict(zip(unique_mols, train_features_scaled))

        # Tranform features appearing in val/test sets
        nontrain_keys = list(set(unscaled_features[type_].keys()) - set(scaled_train_features.keys()))

        if len(nontrain_keys) == 0:
            scaled_features = scaled_train_features
        else:
            nontrain_features_stack_unscaled = np.vstack([unscaled_features[type_][mol] for mol in nontrain_keys])
            nontrain_features_scaled = scaler.transform(nontrain_features_stack_unscaled)
            scaled_nontrain_features = dict(zip(nontrain_keys, nontrain_features_scaled))

            # Combine features
            scaled_features = {**scaled_train_features, **scaled_nontrain_features}
        
    return scaled_features

def scale(train_set, SMILES, DESCRIPTORS):
    '''
    Scales Unscaled RDKit descriptors.
    '''
    
    unscaled_feats = get_unscaled_features(SMILES,DESCRIPTORS)

    # Consider only monomers and bonds in the train dataset
    all_monomers = "".join(train_set['sequence'].tolist())
    all_bonds = "".join(
        (len(row.sequence) - 1) * 'Amb' if 'pep' in row.ID else (len(row.sequence) - 1) * 'Cc'
        for row in train_set[['ID', 'sequence']].itertuples()
    )

    # fit_transform features in train dataset; apply transform to features in val/test sets
    scaled_monomers = scale_features('monomer', all_monomers, unscaled_feats)
    scaled_bonds = scale_features('bond', all_bonds, unscaled_feats)

    return {'monomer': scaled_monomers, 'bond': scaled_bonds}
