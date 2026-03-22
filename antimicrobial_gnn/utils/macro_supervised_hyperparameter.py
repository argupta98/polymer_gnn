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
from utils.macro_dataset import MacroDataset
from utils.stopper import Stopper_v2
from utils.meter import Meter_v2
from torch.utils.data import DataLoader

from datetime import datetime

import seaborn as sns
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score

from dgllife.utils import RandomSplitter

from torch.optim import Adam

class MacroSupervised():
    def __init__(self, MacroDataset, MODEL, NUM_EPOCHS, NUM_WORKERS, DESCRIPTORS, CUSTOM_PARAMS, INFERENCE = None, MODEL_PATH=None, SAVE_MODEL=False, SAVE_OPT=False, SAVE_CONFIG=False):
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
        self._dataset = MacroDataset
        self._inference = INFERENCE
        
        # self._fp_bits_mon = FP_BITS_MON
        # self._fp_bits_bond = FP_BITS_BOND
        descriptors_dict = pd.read_json(DESCRIPTORS).to_dict(orient='records')[0]
        self._num_node_descriptors = len(descriptors_dict['node'])
        self._num_edge_descriptors = len(descriptors_dict['edge'])
        
        self._model_name = MODEL
        self._num_epochs = NUM_EPOCHS
        self._num_workers = NUM_WORKERS
        self._custom_params = CUSTOM_PARAMS
        self._model_path = MODEL_PATH
        self._save_model = SAVE_MODEL
        self._save_opt = SAVE_OPT
        self._save_config = SAVE_CONFIG

        self._log = [['set_type', 'epoch', 'loss', 'roc_auc', 'f1', 'recall', 'precision', 'accuracy', 'confusion_matrix', 'time']]

        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')
        
        if self._dataset.task == 'classification':
            if self._dataset.classtype == 'binary':
                self._loss_criterion = nn.BCEWithLogitsLoss(reduction='none')
            elif self._dataset.classtype == 'multiclass':
                self._loss_criterion = nn.CrossEntropyLoss()
            elif self._dataset.classtype == 'multilabel':
                self._loss_criterion = nn.BCEWithLogitsLoss(reduction='none')
        elif self._dataset.task == 'regression':
            self._loss_criterion = nn.SmoothL1Loss(reduction='none')
            
        self._config_update()
        if self._model_path != None:
            self._mkdir_p()
        self._split_dataset()
        self._load_hparams()
    
    def _config_update(self):
        ''' Utility function for update of configuration dictionary '''
        self._exp_config = {}
        self._exp_config['model'] = self._model_name
        self._exp_config['n_tasks'] = self._dataset.n_tasks
            
        # self._exp_config['in_node_feats'] = self._fp_bits_mon
        # self._exp_config['in_edge_feats'] = self._fp_bits_bond
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
        
    def _split_dataset(self):
        ''' Utility function for splitting Dataset into Subsets for train, validation, and testing '''
        self.train_set = self._dataset.dgl_data['train']
        self.val_set = self._dataset.dgl_data['val']

        if self._inference is not None:
            self.inference_set = self._inference.dgl_data['inference']
        
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
        train_meter = Meter_v2()
        loss_list = []
        
        for batch_id, batch_data in enumerate(data_loader):            
            IDs, bg, labels, masks = batch_data
            labels, masks = labels.to(self._device), masks.to(self._device)
            logits = self._predict(model, bg)
            if self._dataset.task == 'classification':
                if self._dataset.classtype == 'multiclass':
                    losslabels = torch.max(labels, 1)[1]
                else:
                    losslabels = labels
            else:
                losslabels = labels
                
            loss = (self._loss_criterion(logits, losslabels) * (masks != 0).float()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_meter.update(IDs, logits, labels, masks)
            loss_list.append(loss.item())

        epoch_end_time = datetime.now()

        self._log.append(['train', epoch+1, np.mean(loss_list), train_meter.compute_metric('roc_auc_score'), train_meter.compute_metric('f1_score'), 
                         train_meter.compute_metric('recall_score'), train_meter.compute_metric('precision_score'), 
                         train_meter.compute_metric('accuracy_score'), train_meter.compute_metric('confusion_matrix'),
                         epoch_end_time - epoch_start_time])        
        
    def _run_an_eval_epoch(self, model, data_loader, set_type = None, epoch = None):
        ''' Utility function for running an evaluation (validation/test) epoch
        
        Args:
        model : dgllife model, Predictor with set hyperparameters
        data_loader : DataLoader, DataLoader for train, validation, or test
        
        Returns:
        metric_dict : dict, dictionary of metric names and corresponding evaluation values
        '''
        eval_start_time = datetime.now()
        model.eval()
        eval_meter = Meter_v2()
        loss_list = []
        with torch.no_grad():
            for batch_id, batch_data in enumerate(data_loader):                
                IDs, bg, labels, masks = batch_data
                labels, masks = labels.to(self._device), masks.to(self._device)                
                logits = self._predict(model, bg)
                eval_meter.update(IDs, logits, labels, masks)
                if self._dataset.task == 'classification':
                    if self._dataset.classtype == 'multiclass':
                        losslabels = torch.max(labels, 1)[1]
                    else:
                        losslabels = labels
                else:
                    losslabels = labels
                loss = (self._loss_criterion(logits, losslabels) * (masks != 0).float()).mean()
                loss_list.append(loss.item())

        eval_end_time = datetime.now()
        
        if self._dataset.task == 'classification':
            if self._dataset.classtype == 'binary':
                metric_dict = {'loss': np.mean(loss_list), 'ROC-AUC': eval_meter.compute_metric('roc_auc_score'),
                               'F1': eval_meter.compute_metric('f1_score'), 'recall': eval_meter.compute_metric('recall_score'),
                               'precision': eval_meter.compute_metric('precision_score'), 'accuracy': eval_meter.compute_metric('accuracy_score'),
                               'confusion_matrix': eval_meter.compute_metric('confusion_matrix')}

                if epoch is not None:
                    data = sum([[set_type, epoch+1], list(metric_dict.values()), [eval_end_time - eval_start_time]], [])
                    self._log.append(data)

                IDs, mask, y_pred, y_true = eval_meter._finalize(include_IDs = True)
        
                y_pred = torch.FloatTensor(torch.sigmoid(y_pred).numpy()) * (mask != 0).float()
                y_pred = y_pred.numpy().ravel()
                mask = mask.numpy().ravel()
                y_true = y_true.long().numpy().ravel()
                
        return metric_dict, [IDs, mask, y_pred, y_true]
        
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
        train_loader = DataLoader(dataset=self.train_set, batch_size=self._exp_config['batch_size'], shuffle=True,
                              collate_fn=self._collate_molgraphs, num_workers=self._num_workers)
        val_loader = DataLoader(dataset=self.val_set, batch_size=self._exp_config['batch_size'], shuffle=True,
                            collate_fn=self._collate_molgraphs, num_workers=self._num_workers)

        if self._inference is not None:
            inference_loader = DataLoader(dataset=self.inference_set, batch_size=self._exp_config['batch_size'], shuffle=True,
                                collate_fn=self._collate_molgraphs, num_workers=self._num_workers)
        
        self.model_load = self._load_model()
        model = self.model_load.to(self._device)
        
        if self._model_path == None:
            tmp_dir = tempfile.mkdtemp()
            tmppath = tempfile.NamedTemporaryFile(prefix='model',suffix='.pth',dir=tmp_dir)
        else:
            tmppath = tempfile.NamedTemporaryFile(prefix='model',suffix='.pth',dir=self._model_path)
        
        optimizer = Adam(model.parameters(), lr=self._exp_config['lr'],
                     weight_decay=self._exp_config['weight_decay'])
        stopper = Stopper_v2(savepath=self._model_path, patience=self._exp_config['patience'], 
                            filename=tmppath.name)

        for epoch in range(self._num_epochs):

            self._run_a_train_epoch(epoch, model, train_loader, optimizer)
            val_score = self._run_an_eval_epoch(model, val_loader, set_type = 'val', epoch = epoch)[0] 

            if self._inference is not None:
                inference_score = self._run_an_eval_epoch(model, inference_loader, set_type = 'inference', epoch = epoch)[0]
                early_stop = stopper.step(
                    val_score['loss'], 
                    model, optimizer, self._model_name, self._save_model, self._save_opt,
                    INFERENCE_SCORE = inference_score['precision']) 
            else:
                early_stop = stopper.step(
                    val_score['loss'], 
                    model, optimizer, self._model_name, self._save_model, self._save_opt)  
            
            if early_stop:
                break

        stopper.load_checkpoint(model, set_type = 'val')
            
        self._best_val_on_val = self._run_an_eval_epoch(model, val_loader)
        self.get_results(self._best_val_on_val, folder_name = 'val_model_on_val_set')

        if self._inference is not None:

            self._best_val_on_infer = self._run_an_eval_epoch(model, inference_loader)
            self.get_results(self._best_val_on_infer, folder_name = 'val_model_on_infer_set')
            
            stopper.load_checkpoint(model, set_type = 'inference')
            
            self._best_infer_on_infer = self._run_an_eval_epoch(model, inference_loader)
            self.get_results(self._best_infer_on_infer, folder_name = 'infer_model_on_infer_set')

            self._best_infer_on_val = self._run_an_eval_epoch(model, val_loader)
            self.get_results(self._best_infer_on_val, folder_name = 'infer_model_on_val_set')
        
        tmppath.close()
        if self._model_path == None:
            shutil.rmtree(tmp_dir)
        
        if self._save_config == True:
            with open(self._model_path + '/configure.json', 'w') as f:
                self._exp_config = {key: int(value) if isinstance(value, np.int64) else value for key, value in self._exp_config.items()}
                json.dump(self._exp_config, f, indent=2)

        log_df = pd.DataFrame(self._log[1:], columns=self._log[0])
        log_df['confusion_matrix'] = log_df['confusion_matrix'].apply(lambda x: np.array_str(x).replace('\n', ', ').replace('  ', ' '))
        log_df.to_csv(self._model_path + '/train_log.txt', index=False)

        self.plot_loss(log_df)
        
        self.model = model

        if self._inference is not None:
            return -self._best_val_on_infer[0]['precision']
        else:
            return self._best_val_on_val[0]['loss']

    def get_results(self, eval_data, folder_name):
        
        directory_path = os.path.join(self._model_path, folder_name)
        os.makedirs(directory_path, exist_ok=True)
        
        self.cm_plot(eval_data[0]['confusion_matrix'], directory_path)
        self.rocauc_plot(eval_data[1], directory_path)
        self.prauc_plot(eval_data[1], directory_path)
        self.export_results(eval_data[1], directory_path)
    
    def export_results(self, data, path):
        
        IDs, mask, y_pred, y_true = data

        df = pd.DataFrame({'ID': IDs,
                           'y_pred': y_pred.tolist(),
                           'y_true': y_true.tolist(),
                            'mask': mask.tolist()
                          })

        df.to_csv(path + '/results.txt', index=False)
        
    def plot_loss(self, df):

       # Extract unique epochs and train loss values
        epochs = df['epoch'].unique()
        train_loss = df[df['set_type'] == 'train']['loss'].values
    
        # Validation loss
        val_loss = df[df['set_type'] == 'val']['loss'].values
        val_df = df[df['set_type'] == 'val']
        min_val_epoch = val_df.loc[val_df['loss'].idxmin(), 'epoch']
        min_val_loss = val_df.loc[val_df['loss'].idxmin(), 'loss']
    
        # Limit epochs for validation loss plot
        max_epoch_to_plot_val = min_val_epoch + 5
        epochs_limited_val = epochs[epochs <= max_epoch_to_plot_val]
        train_loss_limited_val = train_loss[:len(epochs_limited_val)]
        val_loss_limited_val = val_loss[:len(epochs_limited_val)]
    
        if self._inference is not None:
            # Inference data
            inference_loss = df[df['set_type'] == 'inference']['loss'].values
            inference_df = df[df['set_type'] == 'inference']
            max_infer_epoch = inference_df.loc[inference_df['precision'].idxmax(), 'epoch']
            max_infer_loss = inference_df.loc[inference_df['precision'].idxmax(), 'loss']
    
            # Limit epochs for inference loss plot
            max_epoch_to_plot_inf = max_infer_epoch + 5
            epochs_limited_inf = epochs[epochs <= max_epoch_to_plot_inf]
            train_loss_limited_inf = train_loss[:len(epochs_limited_inf)]
            inference_loss_limited_inf = inference_loss[:len(epochs_limited_inf)]
    
            # Create figure with three subplots
            fig, ax = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns
    
            # First subplot: train, val, and inference loss across all epochs
            ax[0].plot(epochs, train_loss, marker='o', linestyle='-', color='r', label='train', markersize=3)
            ax[0].plot(epochs, val_loss, marker='o', linestyle='-', color='g', label='val', markersize=3)
            ax[0].plot(epochs, inference_loss, marker='o', linestyle='-', color='b', label='inference', markersize=3)
            ax[0].plot(min_val_epoch, min_val_loss, marker='*', color='purple', markersize=10, label='min val loss')
            ax[0].plot(max_infer_epoch, max_infer_loss, marker='*', color='gold', markersize=10, label='max precision')
            ax[0].set_title('All Losses (Train, Val, Inference)')
            ax[0].set_xlabel('Epoch')
            ax[0].set_ylabel('Loss')
            ax[0].legend()
    
            # Second subplot: train and validation loss (limited by min_val_epoch + 5)
            ax[1].plot(epochs_limited_val, train_loss_limited_val, marker='o', linestyle='-', color='r', label='train', markersize=3)
            ax[1].plot(epochs_limited_val, val_loss_limited_val, marker='o', linestyle='-', color='g', label='val', markersize=3)
            ax[1].plot(min_val_epoch, min_val_loss, marker='*', color='purple', markersize=10, label='min val loss')
            ax[1].set_title('Train & Val Loss')
            ax[1].set_xlabel('Epoch')
            ax[1].set_ylabel('Loss')
            ax[1].legend()
    
            # Third subplot: train and inference loss (limited by max_infer_epoch + 5)
            ax[2].plot(epochs_limited_inf, train_loss_limited_inf, marker='o', linestyle='-', color='r', label='train', markersize=3)
            ax[2].plot(epochs_limited_inf, inference_loss_limited_inf, marker='o', linestyle='-', color='b', label='inference', markersize=3)
            ax[2].plot(max_infer_epoch, max_infer_loss, marker='*', color='gold', markersize=10, label='max precision')
            ax[2].set_title('Train & Inference Loss')
            ax[2].set_xlabel('Epoch')
            ax[2].set_ylabel('Loss')
            ax[2].legend()
    
        else:
            # No inference data, just plot train and val
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns
    
            # First subplot: train and val loss across all epochs
            ax[0].plot(epochs, train_loss, marker='o', linestyle='-', color='r', label='train', markersize=3)
            ax[0].plot(epochs, val_loss, marker='o', linestyle='-', color='g', label='val', markersize=3)
            ax[0].plot(min_val_epoch, min_val_loss, marker='*', color='purple', markersize=10, label='min val loss')
            ax[0].set_title('All Losses (Train & Val)')
            ax[0].set_xlabel('Epoch')
            ax[0].set_ylabel('Loss')
            ax[0].legend()
    
            # Second subplot: train and val loss (limited by min_val_epoch + 5)
            ax[1].plot(epochs_limited_val, train_loss_limited_val, marker='o', linestyle='-', color='r', label='train', markersize=3)
            ax[1].plot(epochs_limited_val, val_loss_limited_val, marker='o', linestyle='-', color='g', label='val', markersize=3)
            ax[1].plot(min_val_epoch, min_val_loss, marker='*', color='purple', markersize=10, label='min val loss')
            ax[1].set_title('Train & Val Loss')
            ax[1].set_xlabel('Epoch')
            ax[1].set_ylabel('Loss')
            ax[1].legend()
    
        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig(self._model_path + '/loss_fig.png')
        plt.close()
    
    def cm_plot(self, data, path):
        disp = ConfusionMatrixDisplay(confusion_matrix=data)
        disp.plot()
        plt.savefig(path + '/CM.png')
        plt.close()

    def prauc_plot(self, plotdata, path):
        
        y_pred, y_true = plotdata[2], plotdata[3]

        # Compute precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        
        # Compute the average precision score (area under the precision-recall curve)
        average_precision = average_precision_score(y_true, y_pred)
        
        # Plot Precision-Recall curve
        plt.figure()
        lw = 2
        plt.plot(recall, precision, color='#2C7FFF', lw=lw, label='Precision-Recall curve')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=18)
        plt.ylabel('Precision', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.text(0.95, 0.03, 'Avg Precision = %0.2f' % (average_precision),
                 verticalalignment='bottom', horizontalalignment='right',
                 fontsize=18)
        

        plt.axhline(y=0.5, color='#B2B2B2', lw=lw, linestyle='--')

        plt.tight_layout()
        plt.savefig(path + '/PR_Curve.png')
        plt.close()
    
    def rocauc_plot(self, plotdata, path):
        ''' Plots ROC-AUC curve for classification task
        
        Args:
        plottype : str, dataset to plot, 'val' for validation or 'test' for test
        fig_path : str, path to save figure
        '''

        y_pred, y_true = plotdata[2], plotdata[3]
        mean_fpr, mean_tpr, _ = roc_curve(y_true, y_pred)
        
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)

        plt.figure()
        lw = 2
        plt.plot(mean_fpr, mean_tpr, color='#2C7FFF',lw=lw)
        plt.plot([0, 1], [0, 1], color='#B2B2B2', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=18)
        plt.ylabel('True Positive Rate', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.text(0.95, 0.03, 'ROC-AUC = %0.2f' % (mean_auc),
        verticalalignment='bottom', horizontalalignment='right',
        fontsize=18)
        
        plt.tight_layout()
        plt.savefig(path + '/ROC_AUC.png')
        plt.close()
