import pandas as pd
import numpy as np

import dgl
import torch

from sklearn.preprocessing import StandardScaler, QuantileTransformer


class MacroDataset():
    def __init__(self, DF_PATH, TASK, LABELNAME, MODEL, NX_GRAPHS):
        '''
        Initializes a MacroDataset object
        
        Args:
        DF_PATH: str, path to DataFrame containing all macromolecules and corresponding labels
        SEED: int, random seed for shuffling dataset
        LABELNAME: str, name of label to classify
        NX_GRAPHS: dict, dictionary of featurized NetworkX graph for each macromolecule ID
        NORM: str, normalization method for regression dataset (default=None)
        
        Attributes:
        IDs: list, list of macromolecule IDs in dataset
        graphs: list, list of graphs corresponding to each ID
        labels: list, list of labels corresponding to each ID
        masks: list, list of masks corresponding to each ID
        task: str, classification or regression
        n_tasks: int, number of tasks
        classtype: str, binary, multilabel, or multiclass for classification tasks
        normalizer: StandardScaler or QuantileTransformer for normalization
        
        '''
        self._df = pd.read_csv(DF_PATH)
        self._labelname = LABELNAME
        self._model = MODEL
        self._nx_graphs = NX_GRAPHS
        self.task = TASK
        self.dgl_data = {}
        self._convert_dgl()

    def _convert_dgl(self):

        for set in self._nx_graphs.keys():
            IDs = list(self._nx_graphs[set].keys())
            graphs = list(self._nx_graphs[set].values())
            
            if self._model == 'GCN' or self._model == 'GAT':
                dgl_graphs = [dgl.from_networkx(graph, node_attrs=['h'], edge_attrs=['e'], idtype=torch.int32) for graph in graphs]
                dgl_graphs = [dgl.add_self_loop(graph) for graph in dgl_graphs]
            else:
                dgl_graphs = [dgl.from_networkx(graph, node_attrs=['h'], edge_attrs=['e'], idtype=torch.int32) for graph in graphs]

            labels = self._df.set_index('ID').loc[IDs, self._labelname].reset_index(drop=True)
            
            masks = pd.isnull(labels)
            masks = masks.apply(lambda x: torch.tensor([1], dtype=torch.float32) if not x else torch.tensor([0], dtype=torch.float32))

            unique_labels = sorted(labels.unique())
            labels = labels.map(lambda x: torch.tensor([unique_labels.index(x)], dtype=torch.float32))

            if len(unique_labels) == 2:
                self.classtype = 'binary'
                self.n_tasks = 1

            dgl_set = list(zip(IDs, dgl_graphs, labels, masks))
            self.dgl_data[set] = dgl.data.utils.Subset(dgl_set, range(len(dgl_set)))

        return
    
