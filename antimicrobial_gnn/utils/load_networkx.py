from __future__ import absolute_import

import os
import numpy as np
import networkx as nx
import re
import collections

import pandas as pd
import joblib

from rdkit.Chem import rdMolDescriptors, Descriptors
from rdkit import Chem

import grakel
import dgl
import torch
from dgl.data.utils import save_graphs, load_graphs
from dgl.data import DGLDataset

from itertools import accumulate
import itertools
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import MinMaxScaler


def _txtread_index0(TXT_DATA_PATH, file):
    '''
    Processes .txt file for macromolecule into node and edge dictionaries
    
    Args:
    TXT_DATA_PATH: str, path to .txt files for dataset to be used for training
    file: str, name of .txt file in TXT_DATA_PATH directory to read
            
    Returns:
    [node_dict, edge_dict]: list, list of two dictionaries, one for nodes and the other for edges
    '''
    node_dict = {} 
    edge_dict = {} 
    with open(os.path.join(TXT_DATA_PATH,file)) as txt_file:
        monbool = False
        bondbool = False
        for line in txt_file: 
            line = line.strip() 
            if line == 'MONOMERS':
                monbool = True 
                continue 
            if line == '': 
                continue
            if line == 'BONDS':
                monbool = False
                bondbool = True 
                continue 
            if monbool == True:
                line_split = line.split(' ') 
                node_dict[int(line_split[0]) - 1] = line_split[1]
            if bondbool == True:
                line_split = line.split(' ') 
                pos_tuple = (int(line_split[0]) - 1, 
                             int(line_split[1]) - 1)
                edge_dict[pos_tuple] = line_split[2]
    return [node_dict, edge_dict]

def _graphgen(TXT_DATA_PATH, file):
    '''
    Processes .txt file for macromolecule into unfeaturized NetworkX graph
    
    Args:
    TXT_DATA_PATH: str, path to .txt files for dataset to be used for training
    file: str, name of .txt file in TXT_DATA_PATH directory to read
            
    Returns:
    graph: NetworkX graph, NetworkX graph corresponding with specified file in TXT_DATA_PATH
    '''
    dict_list = _txtread_index0(TXT_DATA_PATH, file)
    graph = nx.DiGraph()
    graph.add_nodes_from(list(dict_list[0].keys())) 
    graph.add_edges_from(list(dict_list[1].keys())) 
    for node in list(dict_list[0].keys()): 
        graph.nodes[node]['label'] = dict_list[0][node] 
    for edge in list(dict_list[1].keys()): 
        graph.edges[edge]['label'] = dict_list[1][edge] 
    return graph

def _mon_graphsgen(TXT_DATA_PATH):
    '''
    Processes all .txt files of a macromolecule type into a dictionary of unfeaturized NetworkX graphs
    
    Args:
    TXT_DATA_PATH: str, path to .txt files for dataset to be used for training
            
    Returns:
    mon_ordered: dict, sorted dictionary with keys as glycan IDs and values as NetworkX graphs
    '''
    mon_graphs = {}
    for subdir, dirs, files in os.walk(TXT_DATA_PATH):
        for file in files:
            if file in files and file.endswith('.txt'):
                glycan_id = file.split('_')[0]
                mon_graphs[glycan_id] = _graphgen(os.path.join(subdir),file)
    mon_ordered = collections.OrderedDict(sorted(mon_graphs.items()))
    return mon_ordered

def featurize_graphs(split_dict, graphs, scaled_feats):

    featurized_graphs = {}

    for set in split_dict.keys():
        featurized_graphs[set] = {}
        for ID in split_dict[set]:
            graph = graphs[ID]
            add_dict = True
            for node in graph.nodes:
                try:
                    graph.nodes[node]['h'] = torch.FloatTensor(scaled_feats['node'][graph.nodes[node]['label']])
                except:
                    add_dict = False
            for edge in graph.edges:
                try:
                    graph.edges[edge]['e'] = torch.FloatTensor(scaled_feats['edge'][graph.edges[edge]['label']])
                except:
                    add_dict = False
            if add_dict == True:
                featurized_graphs[set][ID] = graph
            else:
                print("Omitted Peptide: " + str(ID))

    return featurized_graphs

def scale_data(data, SCALER_TYPE = None, SCALER = None, N_QUANTILES = None):
    
    if SCALER_TYPE is not None:
        if SCALER_TYPE == 'MinMax':
            scaler = MinMaxScaler(feature_range=(0, 1))     
        elif SCALER_TYPE == 'quantile':
            scaler = QuantileTransformer(output_distribution='normal', n_quantiles=N_QUANTILES)

        scaled_data = scaler.fit_transform(data)
    else:
        scaler = SCALER
        scaled_data = scaler.transform(data)  

    return scaled_data, scaler

def scale_features(unscaled_feats, IDs, GRAPHS, SCALER_TYPE = None, SCALER = None):

    scalers = {}
    scaled_feats = {}
                
    for type in unscaled_feats.keys():
        scaler = SCALER[type] if SCALER else None        
        if len(unscaled_feats[type]) == 1:
            data = unscaled_feats[type][list(unscaled_feats[type].keys())[0]].reshape(-1, 1)
            scaled_data, scaler = scale_data(data, SCALER_TYPE, scaler, N_QUANTILES = len(data))

            scalers[type] = scaler
            scaled_feats[type] = {str(list(unscaled_feats[type].keys())[0]): scaled_data.flatten()}
        else:
            if SCALER_TYPE is not None:

                in_IDs = []
        
                for ID in IDs:
                    if type == 'node':
                        for idx in GRAPHS[ID].nodes():
                            in_IDs.append(GRAPHS[ID].nodes[idx]['label'])
                    elif type == 'edge':
                         for idx in GRAPHS[ID].edges():
                            in_IDs.append(GRAPHS[ID].edges[idx]['label'])
    
                df = pd.DataFrame({
                    str(type): in_IDs,
                    'feats': [unscaled_feats[type][x] for x in in_IDs]
                })
    
                feats_expanded = pd.DataFrame(df['feats'].tolist())
                df = pd.concat([df[type], feats_expanded], axis=1)
    
                data = df.iloc[:, 1:]
            else:
                df = pd.DataFrame(unscaled_feats[type]).T.reset_index().rename(columns={'index': type})
                data = df.iloc[:, 1:]
            
            scaled_data, scaler = scale_data(data, SCALER_TYPE, scaler, N_QUANTILES = df[type].nunique())
            
            scalers[type] = scaler
            
            df.iloc[:, 1:] = scaled_data
            df = df.drop_duplicates(type, keep='first')
            scaled_feats[type] = df.set_index(type).apply(lambda row: np.array(row.tolist()), axis=1).to_dict()

    if SCALER_TYPE is not None:
        joblib.dump(scalers, 'scalers.pkl')
        return scaled_feats, scalers
    else:
        return scaled_feats

def get_unscaled_features(MON_SMILES, BOND_SMILES, DESCRIPTORS):

    df_monomer_smiles = pd.read_csv(MON_SMILES)
    df_bonds_smiles = pd.read_csv(BOND_SMILES)
    df = pd.concat(
        [df_monomer_smiles.assign(type='node'), df_bonds_smiles.assign(type='edge')]
    ).reset_index(drop=True)

    descriptors_to_keep = pd.read_json(DESCRIPTORS).to_dict(orient='records')[0]

    unscaled_feats = {}
    
    for type in df['type'].unique():

        df_type = df[df['type'] == type]
        full_features = df_type['SMILES'].apply(
            lambda x: Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(x), missingVal=-9999, silent=True)
        )
        features = full_features.map(lambda x: np.array([x[key] for key in descriptors_to_keep[type]]))
        feats_dict = dict(zip(df_type['Molecule'], features))
        unscaled_feats[type] = feats_dict
    
    return unscaled_feats

def split_dataset(dataset, SPLIT, SEED):

    frac_list = np.array(list(map(float, SPLIT.split(','))))
    num_data = len(dataset)
    lengths = (num_data * frac_list).astype(int)
    lengths[-1] = num_data - np.sum(lengths[:-1])
    indices = np.random.RandomState(seed=SEED).permutation(num_data)

    train_IDs, val_IDs = [
        dataset[indices[offset - length : offset]] for offset, length in zip(accumulate(lengths), lengths)
    ]

    return {'train': list(train_IDs),
            'val': list(val_IDs)}

def networkx_feat(TXT_DATA_PATH, MON_SMILES, BOND_SMILES, DESCRIPTORS, SPLIT = None, SEED = None, SCALER_TYPE = None, SCALER = None):
    '''
    Processes all .txt files of a macromolecule type into a dictionary of featurized NetworkX graphs
    
    Args:
    TXT_DATA_PATH: str, path to .txt files for dataset to be used for training
    MON_SMILES: str, path to .txt file of all monomers that comprise macromolecule and corresponding SMILES
    BOND_SMILES: str, path to .txt file of all bonds that comprise macromolecule and corresponding SMILES
    FEAT: str, type of attribute with which to featurizer nodes and edges of macromolecule NetworkX graphs (default = 'fp')
        
    Returns:
    graphs_feat: dict, dictionary of molecular fingerprint-featurized graphs for all molecule IDs
    '''
    mon_graphs = {}
    mon_graphs = _mon_graphsgen(TXT_DATA_PATH)
    
    IDs = np.array(list(mon_graphs.keys()))

    # Split Dataset
    if SCALER_TYPE is not None:
        split_dict = split_dataset(IDs, SPLIT, SEED)
    else:
        split_dict = {'inference': list(IDs)}

    # Get unscaled features
    unscaled_feats = get_unscaled_features(MON_SMILES, BOND_SMILES, DESCRIPTORS)

    # Scale features
    if SCALER_TYPE is not None:
        scaled_feats, scalers = scale_features(unscaled_feats, 
                                               IDs = split_dict['train'],
                                               GRAPHS = mon_graphs,
                                               SCALER_TYPE = SCALER_TYPE)
    elif SCALER is not None:
        scaled_feats = scale_features(unscaled_feats,
                                      IDs = split_dict['inference'],
                                      GRAPHS = mon_graphs,
                                      SCALER = SCALER)        

    # Featurize graphs
    featurized_graphs = featurize_graphs(split_dict, mon_graphs, scaled_feats)

    if SCALER_TYPE is not None:
        return featurized_graphs, scalers
    else:
        return featurized_graphs

