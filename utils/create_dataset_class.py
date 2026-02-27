import pandas as pd
import numpy as np
import re

import dgl
import torch

from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

from utils.util_functions import seq_to_dgl

class DataSet():
    def __init__(self, data_split, features, LABELNAME, MIXED, TASK, MODEL):
        
        self._data_split = data_split
        self._labelname = LABELNAME
        self._model = MODEL
        self._task = TASK
        self._mixed = MIXED
        self._features = features
        
        self._data = self.structure_data(self._data_split)

    def structure_data(self, split_dataset):
        '''
        Structure data to have the form: (ID, featurized graph, label, mask)
        '''
        
        data = {}
        
        unique_labels = set()
        
        for _set, df in split_dataset.items():
            
            df = df.copy()
            df['dgl'] = df.apply(lambda row: seq_to_dgl(row['ID'], row['sequence'], self._features, self._model), axis=1)
            IDs = df['ID'].tolist()
            
            dgl_graphs = df['dgl'].tolist()
            raw_labels = df[self._labelname].tolist()
            unique_labels.update({x for x in raw_labels if pd.notna(x)})

            masks = [torch.tensor(1.0 if pd.notna(x) else 0.0, dtype=torch.float32) for x in raw_labels]
            labels = [torch.tensor(int(x) if pd.notna(x) else 0, dtype=torch.long) for x in raw_labels] # make labels list of tensors

            data[_set] = list(zip(IDs, dgl_graphs, labels, masks))
            
        if self._task == 'classification':
            self._ntask = len(unique_labels)
            
        return data
    