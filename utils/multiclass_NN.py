import pandas as pd
import os
import errno
import json
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import shutil
import tempfile
import dgl
import torch
import random
from utils.stopper import Stopper
from utils.meter import Meter
from torch.utils.data import DataLoader

from datetime import datetime

import seaborn as sns
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score, confusion_matrix


from dgllife.utils import RandomSplitter

from torch.optim import Adam

class multiclass_NN():
    def __init__(self, dataset, MODEL, NUM_EPOCHS, NUM_WORKERS, DESCRIPTORS, CUSTOM_PARAMS, MODEL_PATH=None, SAVE_MODEL=False, SAVE_OPT=False, SAVE_CONFIG=False):
        '''
        Initializes a MacroSupervised object
        
        Args:
        MacroDataset: MacroDataset, MacroDataset object for DGL Dataset
        MON_SMILES: str, path to .txt file of all monomers that comprise macromolecule and corresponding SMILES
        BOND_SMILES: str, path to .txt file of all bonds that comprise macromolecule and corresponding SMILES
        FEAT: str, type of attribute with which to featurizer nodes and edges of macromolecule
        FP_BITS_MON: int, size of fingerprint bit-vector for monomer 
        FP_BITS_BOND: int, size of fingerprint bit-vector for bond
        SEED: int, random seed for shuffling dataset
        MODEL: str, model architecture for supervised learning 
        SPLIT: str, proportion of the dataset to use for training, validation and test
        NUM_EPOCHS: int, maximum number of epochs allowed for training
        NUM_WORKERS: int, number of processes for data loading
        CUSTOM_PARAMS: dict, dictionary of custom hyperparameters
        MODEL_PATH: str, path to save models and configuration files (default=None)
        SAVE_MODEL: boolean, whether to save full model file (default=False)
        SAVE_OPT: boolean, whether to save optimizer files (default=False)
        SAVE_CONFIG: boolean, whether to save configuration file (default=False)
        
        Attributes:
        train_set: Subset, Subset of graphs for model training
        val_set: Subset, Subset of graphs for model validation
        test_set: Subset, Subset of graphs used for model testing
        model_load: dgllife model, Predictor with set hyperparameters
        
        '''
        self._dataset = dataset
        
        descriptors_dict = pd.read_json(DESCRIPTORS).to_dict(orient='records')[0]
        self._num_node_descriptors = len(descriptors_dict['monomer'])
        self._num_edge_descriptors = len(descriptors_dict['bond'])
        
        self._model_name = MODEL
        self._num_epochs = NUM_EPOCHS
        self._num_workers = NUM_WORKERS
        self._custom_params = CUSTOM_PARAMS
        self._model_path = MODEL_PATH
        self._save_model = SAVE_MODEL
        self._save_opt = SAVE_OPT
        self._save_config = SAVE_CONFIG

        self._log = [] # cols initialized during the first training run

        self._ntask = self._dataset._ntask
        self._task = self._dataset._task

        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')    
        
        # define loss function
        if self._task == "classification":
            y = self._dataset._data_split["train"][self._dataset._labelname].to_numpy()
            y = y[pd.notna(y)]
            C = self._ntask  # number of classes (or set manually)
            
            counts = np.bincount(y.astype(int), minlength=C)
            
            if (counts == 0).any():
                missing = np.where(counts == 0)[0].tolist()
                print(f"Warning: classes missing from train: {missing}")

            counts = np.maximum(counts, 1) # avoid divide-by-zero if a class is missing in train

            weights = counts.sum() / (C * counts)   # "balanced" weights
            weights = torch.tensor(weights, dtype=torch.float32).to(self._device)

            self._loss_criterion = nn.CrossEntropyLoss(weight=weights, reduction="none")  # reduction as you need
            # self._loss_criterion = nn.CrossEntropyLoss(reduction='none')
            
        self._exp_config = {}
        if self._model_path != None:
            self._mkdir_p()

        self._load_hparams()  # load in hyper parameters from model_hparams
        self._config_update() # update num node/edge features + num tasks
    
    def _config_update(self):
        ''' Utility function for update of configuration dictionary '''
        self._exp_config['model'] = self._model_name
        self._exp_config['n_tasks'] = self._ntask
            
        self._exp_config['in_node_feats'] = self._num_node_descriptors
        self._exp_config['in_edge_feats'] = self._num_edge_descriptors
    
    def _mkdir_p(self):
        ''' Utility function for creation of folder for given path'''
        try:
            os.makedirs(self._model_path)
            print('Created directory {}'.format(self._model_path))
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(self._model_path):
                print('Directory {} already exists.'.format(self._model_path))
            else:
                raise
        
    def _load_hparams(self):
        ''' Utility function for loading default hyperparameters and updating them to reflect custom hyperparameters '''
        with open('./model_hparams/{}.json'.format(self._model_name), 'r') as f:
            config = json.load(f)
        config.update(self._custom_params)
        self._exp_config.update(config)
        
    def _load_model(self):
        ''' Utility function for loading model 
        
        Returns:
        model: dgllife model, Predictor with set hyperparameters
        '''
        if self._model_name == 'GCN':
            from dgllife.model import GCNPredictor
            model = GCNPredictor(
                in_feats=self._exp_config['in_node_feats'],
                hidden_feats=[self._exp_config['gnn_hidden_feats']] * self._exp_config['num_gnn_layers'],
                activation=[F.relu] * self._exp_config['num_gnn_layers'],
                residual=[self._exp_config['residual']] * self._exp_config['num_gnn_layers'],
                batchnorm=[self._exp_config['batchnorm']] * self._exp_config['num_gnn_layers'],
                dropout=[self._exp_config['dropout']] * self._exp_config['num_gnn_layers'],
                predictor_hidden_feats=self._exp_config['predictor_hidden_feats'],
                predictor_dropout=self._exp_config['dropout'],
                n_tasks=self._exp_config['n_tasks'])
        elif self._model_name == 'GAT':
            from dgllife.model import GATPredictor
            model = GATPredictor(
                in_feats=self._exp_config['in_node_feats'],
                hidden_feats=[self._exp_config['gnn_hidden_feats']] * self._exp_config['num_gnn_layers'],
                num_heads=[self._exp_config['num_heads']] * self._exp_config['num_gnn_layers'],
                feat_drops=[self._exp_config['dropout']] * self._exp_config['num_gnn_layers'],
                attn_drops=[self._exp_config['dropout']] * self._exp_config['num_gnn_layers'],
                alphas=[self._exp_config['alpha']] * self._exp_config['num_gnn_layers'],
                residuals=[self._exp_config['residual']] * self._exp_config['num_gnn_layers'],
                predictor_hidden_feats=self._exp_config['predictor_hidden_feats'],
                predictor_dropout=self._exp_config['dropout'],
                n_tasks=self._exp_config['n_tasks']
            )
        elif self._model_name == 'Weave':
            from dgllife.model import WeavePredictor
            model = WeavePredictor(
                node_in_feats=self._exp_config['in_node_feats'],
                edge_in_feats=self._exp_config['in_edge_feats'],
                num_gnn_layers=self._exp_config['num_gnn_layers'],
                gnn_hidden_feats=self._exp_config['gnn_hidden_feats'],
                graph_feats=self._exp_config['graph_feats'],
                gaussian_expand=self._exp_config['gaussian_expand'],
                n_tasks=self._exp_config['n_tasks']
            )
        elif self._model_name == 'MPNN':
            from dgllife.model import MPNNPredictor
            model = MPNNPredictor(
                node_in_feats=self._exp_config['in_node_feats'],
                edge_in_feats=self._exp_config['in_edge_feats'],
                node_out_feats=self._exp_config['node_out_feats'],
                edge_hidden_feats=self._exp_config['edge_hidden_feats'],
                num_step_message_passing=self._exp_config['num_step_message_passing'],
                num_step_set2set=self._exp_config['num_step_set2set'],
                num_layer_set2set=self._exp_config['num_layer_set2set'],
                n_tasks=self._exp_config['n_tasks']
            )
        elif self._model_name == 'AttentiveFP':
            from dgllife.model import AttentiveFPPredictor
            model = AttentiveFPPredictor(
                node_feat_size=self._exp_config['in_node_feats'],
                edge_feat_size=self._exp_config['in_edge_feats'],
                num_layers=self._exp_config['num_layers'],
                num_timesteps=self._exp_config['num_timesteps'],
                graph_feat_size=self._exp_config['graph_feat_size'],
                dropout=self._exp_config['dropout'],
                n_tasks=self._exp_config['n_tasks']
            )
            
        return model
    
    def _collate_molgraphs(self, data):
        ''' Utility function for batching list of datapoints for Dataloader 
        
        Args:
        data : list, list of 4-tuples, each for a single datapoint consisting of an ID, a DGLGraph, all-task labels and a binary
        mask indicating the existence of labels
        
        Returns:
        IDs : list, list of GBIDs
        bg : DGLGraph, batched DGLGraph.
        labels : Tensor of dtype float32 and shape (len(data), data.n_tasks), batched datapoint labels
        masks : Tensor of dtype float32 and shape (len(data), data.n_tasks), batched datapoint binary mask indicating the
        existence of labels.
        '''
        IDs, graphs, labels, masks = map(list, zip(*data))

        bg = dgl.batch(graphs)
        bg.set_n_initializer(dgl.init.zero_initializer)
        bg.set_e_initializer(dgl.init.zero_initializer)
        labels = torch.stack(labels, dim=0)

        if masks is None:
            masks = torch.ones(labels.shape)
        else:
            masks = torch.stack(masks, dim=0)

        return IDs, bg, labels, masks

    def _run_a_train_epoch(self, epoch, model, data_loader, optimizer):
        ''' Utility function for running a train epoch 
        
        Args:
        epoch : int, training epoch count
        model : dgllife model, Predictor with set hyperparameters
        data_loader : DataLoader, DataLoader for train, validation, or test
        optimizer : torch.optim.Adam, Adam object
        '''
        epoch_start_time = datetime.now()
        
        model.train()
        train_meter = Meter()
        loss_list = []
        
        for batch_id, batch_data in enumerate(data_loader):
            
            IDs, bg, labels, masks = batch_data
            labels, masks = labels.to(self._device), masks.to(self._device) # (B,), (B,)
            logits = self._predict(model, bg) # (B, n_tasks)

            losslabels = labels.long() # (B,)

            loss = (self._loss_criterion(logits, losslabels) * (masks != 0).float()).mean() # (B,) * (B,)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_meter.update(IDs, logits, labels, masks)
            loss_list.append(loss.item())

        epoch_end_time = datetime.now()

        metrics = train_meter.compute_metrics(loss_list)
        
        # if log is empty, append title
        if len(self._log) == 0:
            self._log.append(sum([['dataset', 'epoch'], list(metrics.keys()), ['time']], []))
        
        data = sum([['train', epoch+1], list(metrics.values()), [epoch_end_time - epoch_start_time]], [])
        self._log.append(data)
        
    def _run_an_eval_epoch(self, model, data_loader, dataset, epoch):
        ''' Utility function for running an evaluation (validation/test) epoch
        
        Args:
        model : dgllife model, Predictor with set hyperparameters
        data_loader : DataLoader, DataLoader for train, validation, or test
        
        Returns:
        metric_dict : dict, dictionary of metric names and corresponding evaluation values
        '''
        eval_start_time = datetime.now()
        model.eval()
        eval_meter = Meter()
        loss_list = []
        
        with torch.no_grad():
            for batch_id, batch_data in enumerate(data_loader):                
                IDs, bg, labels, masks = batch_data
                labels, masks = labels.to(self._device), masks.to(self._device)                
                logits = self._predict(model, bg)
                eval_meter.update(IDs, logits, labels, masks)

                losslabels = labels.long()
                
                loss = (self._loss_criterion(logits, losslabels) * (masks != 0).float()).mean()
                loss_list.append(loss.item())

        eval_end_time = datetime.now()
        
        metrics = eval_meter.compute_metrics(loss_list)
        data = sum([[dataset, epoch+1], list(metrics.values()), [eval_end_time - eval_start_time]], [])
        self._log.append(data)

        IDs, mask, logits, y_true = eval_meter._finalize()

        mask = mask.ravel().long()           # (N,)
        y_probs = logits.softmax(dim=1)      # (N,3)
        y_pred = logits.argmax(dim=1).long() # (N,)
        y_true = y_true.ravel().long()       # (N,)
                        
        return metrics, [IDs, mask, y_probs, y_pred, y_true]

    def _predict(self, model, bg):
        ''' Utility function for moving batched graph and node/edge feats to device
        
        Args:
        model : dgllife model, Predictor with set hyperparameters
        bg : DGLGraph, batched DGLGraph
        
        Returns:
        model(bg, node_feats, edge_feats) : model moved to device
        '''
        bg = bg.to(self._device)
        if self._model_name in ['GCN', 'GAT']:
            node_feats = bg.ndata.pop('h').to(self._device)
            return model(bg, node_feats)
        else:
            node_feats = bg.ndata.pop('h').to(self._device)
            edge_feats = bg.edata.pop('e').to(self._device)
            return model(bg, node_feats, edge_feats)
    
    def main(self):
        ''' Performs training, validation, and testing of dataset with output of metrics to centralized files'''
        
        train_loader = DataLoader(dataset=self._dataset._data['train'], batch_size=self._exp_config['batch_size'], shuffle=True,
                              collate_fn=self._collate_molgraphs, num_workers=self._num_workers)
        val_loader = DataLoader(dataset=self._dataset._data['val'], batch_size=self._exp_config['batch_size'], shuffle=True,
                            collate_fn=self._collate_molgraphs, num_workers=self._num_workers)
        test_loader = DataLoader(dataset=self._dataset._data['test'], batch_size=self._exp_config['batch_size'], shuffle=True,
                            collate_fn=self._collate_molgraphs, num_workers=self._num_workers)

        self.model_load = self._load_model()
        model = self.model_load.to(self._device)
                
        if self._model_path == None:
            tmp_dir = tempfile.mkdtemp()
            tmppath = tempfile.NamedTemporaryFile(prefix='model',suffix='.pth',dir=tmp_dir)
        else:
            tmppath = tempfile.NamedTemporaryFile(prefix='model',suffix='.pth',dir=self._model_path)
        
        optimizer = Adam(model.parameters(), lr=self._exp_config['lr'], weight_decay=self._exp_config['weight_decay'])
        stopper = Stopper(savepath=self._model_path, mode='lower', patience=self._exp_config['patience'], filename=tmppath.name)

        for epoch in range(self._num_epochs):
            # print(epoch + 1)
            self._run_a_train_epoch(epoch, model, train_loader, optimizer)
            val_score = self._run_an_eval_epoch(model, val_loader, dataset = 'val', epoch = epoch)[0] 

            early_stop = stopper.step(val_score['loss'], model, optimizer, self._model_name, self._save_model, self._save_opt)  
            
            if early_stop:
                break

        stopper.load_checkpoint(model)

        test_score = self._run_an_eval_epoch(model, test_loader, dataset = 'test', epoch = epoch)
                
        tmppath.close()
        if self._model_path == None:
            shutil.rmtree(tmp_dir)
        
        if self._save_config == True:
            with open(self._model_path + '/configure.json', 'w') as f:
                self._exp_config = {key: int(value) if isinstance(value, np.int64) else value for key, value in self._exp_config.items()}
                json.dump(self._exp_config, f, indent=2)

        log_df = pd.DataFrame(self._log[1:], columns=self._log[0])
        log_df.to_csv(self._model_path + '/train_log.txt', index=False)

        self.plot_loss(log_df)
        
        self.model = model

        results_df = self.export_results(test_score[1])
        
        self.cm_plot(results_df)
        self.rocauc_plot(results_df)
        self.prauc_plot(results_df)

        if self._ntask == 2:
            return -test_score[0]['roc_auc']
        else:
            most_potent_bin = self._ntask - 1
            # return -test_score[0][f"{most_potent_bin}vr_roc_auc"]
            return -test_score[0][f"{most_potent_bin}vr_pr_auc"]
    
    def export_results(self, data):
        
        IDs, mask, y_probs, y_pred, y_true = data

        df = pd.DataFrame({'ID': IDs,
                           'y_probs': y_probs.tolist(),
                           'y_pred': y_pred.tolist(),
                           'y_true': y_true.tolist(),
                            'mask': mask.tolist()
                          })

        df.to_csv(self._model_path + '/results.txt', index=False)

        return df
    
    ### MAKE SEPARATE FILE FOR PLOTS
        
    def plot_loss(self, df):

        epochs = df['epoch'].unique()

        train_loss = df[df['dataset'] == 'train']['loss']
        val_loss = df[df['dataset'] == 'val']['loss']

        plt.plot(epochs, train_loss, label='Train Loss')
        plt.plot(epochs, val_loss, label='Validation Loss')
        
        # Add legend
        plt.legend()
        
        # Add labels and title
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        
        plt.savefig(self._model_path + '/loss_fig.png')
        plt.close()

    def cm_plot(self, df):

        lbs = [i for i in range(int(self._dataset._ntask))]

        contains_poly = df['ID'].str.contains("poly", na=False).any()
        contains_pep = df['ID'].str.contains("pep", na=False).any()
        
        if contains_poly and contains_pep:
            data1 = confusion_matrix(df['y_true'].to_numpy(), df['y_pred'].to_numpy(), labels=lbs)
            disp = ConfusionMatrixDisplay(confusion_matrix=data1)
            disp.plot()
            plt.savefig(self._model_path + '/CM_all.png')
            plt.close()

        if contains_poly:
            polymers = df[df["ID"].str.contains("poly", na=False)].copy()
            split_ID = polymers["ID"].str.split("_S", n=1, expand=True)
            
            polymers["ID"] = split_ID[0]
            polymers["sample"] = split_ID[1].astype(int)
            
            data2 = confusion_matrix(polymers['y_true'].to_numpy(), polymers['y_pred'].to_numpy(), labels=lbs)

            disp = ConfusionMatrixDisplay(confusion_matrix=data2)
            disp.plot()
            plt.savefig(self._model_path + '/CM_all_poly.png')
            plt.close()

            # majority vote
            grouped = (polymers.groupby("ID", sort=False)
                       .agg(y_true=("y_true", "first"),
                            y_pred=("y_pred", lambda s: s.mode().iat[0]))
                       .reset_index())
            grouped = grouped.drop(columns="sample", errors="ignore")

            # convert them all to int
            num_cols = grouped.select_dtypes(include="number").columns
            grouped[num_cols] = grouped[num_cols].round().astype(int)

            data3 = confusion_matrix(grouped['y_true'].to_numpy(), grouped['y_pred'].to_numpy(), labels=lbs)

            disp = ConfusionMatrixDisplay(confusion_matrix=data3)
            disp.plot()

            plt.savefig(self._model_path + '/CM_poly_avg.png')
            plt.close()
            
        if contains_pep:
            pep = df[df["ID"].str.contains("pep", na=False)].copy()
            data4 = confusion_matrix(pep['y_true'].to_numpy(), pep['y_pred'].to_numpy(), labels=lbs)

            disp = ConfusionMatrixDisplay(confusion_matrix=data4)
            disp.plot()
            plt.savefig(self._model_path + '/CM_peptides.png')
            plt.close()

        return

    def rocauc_plot(self, df):
        ''' Plots ROC-AUC curve for classification task
        
        Args:
        plottype : str, dataset to plot, 'val' for validation or 'test' for test
        fig_path : str, path to save figure
        '''

        y_probs = np.vstack(df['y_probs'].values)    # shape (n_samples, n_classes)
        y_true = np.array(df['y_true'].tolist())    # shape (n_samples,)
        
        plt.figure()
        n_classes = y_probs.shape[1]
        
        if n_classes == 2:
            title = "ROC Curves"
            fpr, tpr, _ = roc_curve(y_true, y_probs[:,1])
            roc_auc    = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
        else:
            title = "ROC Curves (One-vs-Rest)"
            for i in range(n_classes):
                # one-vs-rest truth vector
                bin_true = (y_true == i).astype(int)
                class_probs = y_probs[:, i]

                # skip if y_true never equals this class in your data
                if len(np.unique(bin_true)) < 2:
                    continue

                fpr, tpr, _ = roc_curve(bin_true, class_probs)
                roc_auc    = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f"OVR Class {i} (AUC = {roc_auc:.3f})")
        
        # plot the chance diagonal
        plt.plot([0, 1], [0, 1],linestyle="--", lw=2, color="#B2B2B2")
        
        plt.xlim(0, 1)
        plt.ylim(0, 1.05)
        plt.xlabel("False Positive Rate", fontsize=14)
        plt.ylabel("True Positive Rate",  fontsize=14)
        plt.title(title, fontsize=16)
        plt.tick_params(labelsize=12)
        plt.legend(loc="lower right", fontsize=12)
        plt.tight_layout()
        plt.savefig(self._model_path + "/ROC_AUC.png")
        plt.close()
 
        return

    def prauc_plot(self, df):
    
        # stack the per-sample probability lists into an (N × C) array
        y_probs = np.vstack(df['y_probs'].values) # (n_samples, n_classes)
        y_true  = np.array(df['y_true'].tolist()) # (n_samples,)
    
        n_classes = y_probs.shape[1]
        plt.figure()
        
        if n_classes == 2:
            title = "Precision-Recall Curves"
            precision, recall, _ = precision_recall_curve(y_true, y_probs[:,1])
            ap = average_precision_score(y_true, y_probs[:,1])
            plt.plot(recall, precision, lw=2, label=f"AUC = {ap:.3f}")
        else:
            title = "Precision-Recall Curves (One-vs-Rest)"
            for i in range(n_classes):
                # one-vs-rest truth vector
                bin_true = (y_true == i).astype(int)
                class_probs = y_probs[:, i]

                # skip if y_true never equals this class in your data
                if len(np.unique(bin_true)) < 2:
                    continue

                precision, recall, _ = precision_recall_curve(bin_true, class_probs)
                ap = average_precision_score(bin_true, class_probs)

                plt.plot(recall, precision, lw=2, label=f'OVR Class {i} (AP={ap:.3f})')
    
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title(title, fontsize=16)
        plt.legend(loc="lower right", fontsize=12)
        plt.tight_layout()
        plt.savefig(self._model_path + "/PR_Curve.png")
        plt.close()
        
        return
    