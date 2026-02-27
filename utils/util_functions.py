import pandas as pd
import json
from rdkit.Chem import rdMolDescriptors, Descriptors
from rdkit import Chem
import numpy as np
import re
import dgl
import torch


def get_unscaled_features(SMILES, DESCRIPTORS):
    '''
    Get unscaled features of each monomer/bond from RDKit.
    Output: {[Monomer Abbreviation]: [feature vector]}
    '''

    df_smiles = pd.read_csv(SMILES)
    descriptors_to_keep = pd.read_json(DESCRIPTORS).to_dict(orient='records')[0]

    unscaled_feats = {}
    
    for _type in df_smiles['type'].unique():

        df_type = df_smiles[df_smiles['type'] == _type]
        full_features = df_type['SMILES'].apply(
            lambda x: Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(x), missingVal=-9999, silent=True)
        )
        features = full_features.map(lambda x: np.array([x[key] for key in descriptors_to_keep[_type]])) # Remove unwanted features
        feats_dict = dict(zip(df_type['molecule'], features))
        unscaled_feats[_type] = feats_dict
    
    return unscaled_feats

def seq_to_dgl(ID, sequence, features, model):
    '''
    Create and featurize a linear DGL graph given a sequence.
    '''

    monomers = re.findall('[A-Z][^A-Z]*', sequence)

    if 'pep' in ID:
        bond_type = 'Amb'
    else:
        bond_type = 'Cc'

    # Initialize DGL graph
    g = dgl.graph(([], []), num_nodes=len(monomers))

    # Featurize nodes
    node_features = [
        torch.tensor(features["monomer"][monomer], dtype=torch.float32)
        for monomer in monomers
    ]
    g.ndata["h"] = torch.stack(node_features)

    # Linear polymer: Edges are between sequential monomers, i.e., (0->1, 1->2, etc.)
    src_nodes = list(range(len(monomers) - 1))  # Start nodes of edges
    dst_nodes = list(range(1, len(monomers)))  # End nodes of edges
    g.add_edges(src_nodes, dst_nodes)

    # Featurize edges
    edge_features = [
        torch.tensor(features["bond"][bond_type], dtype=torch.float32)
    ] * g.number_of_edges()
    g.edata["e"] = torch.stack(edge_features)

    if model == "GCN" or model == "GAT":
        g = dgl.add_self_loop(g)

    return g