import torch

import numpy as np
import torch.nn.functional as F

import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve, auc, hamming_loss, roc_curve, confusion_matrix, ConfusionMatrixDisplay, average_precision_score


class Meter():
    def __init__(self, mean=None, std=None):
        '''
        Initializes a Meter_v2 object
        
        Args:
        mean : torch.float32 tensor of shape (T) or None, mean of existing training labels across tasks
        std : torch.float32 tensor of shape (T) or None, std of existing training labels across tasks
        
        '''
        self.IDs = []
        self._mask = []
        self.logits = []
        self.labels = []


    def update(self, IDs, logits, label, mask=None):
        '''Updates for the result of an iteration

        Args:
        logits : float32 tensor, predicted labels with shape (B, T), B for number of graphs in the batch and T for number of tasks
        labels : float32 tensor, ground truth labels with shape (B, T), B for number of graphs in the batch and T for number of tasks
        mask : None or float32 tensor, binary mask indicating the existence of ground truth labels
        '''
        self.IDs.append(IDs)
        self.logits.append(logits.detach().cpu())
        self.labels.append(label.detach().cpu())
        if mask is None:
            self._mask.append(torch.ones(self.logits[-1].shape))
        else:
            self._mask.append(mask.detach().cpu())
    
    def _finalize(self, include_IDs = False):
        '''Utility function for preparing for evaluation.

        Returns:
        mask : float32 tensor, binary mask indicating the existence of ground truth labels
        logits : float32 tensor, predicted labels with shape (B, T), B for number of graphs in the batch and T for number of tasks
        labels : float32 tensor, ground truth labels with shape (B, T), B for number of graphs in the batch and T for number of tasks
        '''
        IDs = sum(self.IDs, [])
        mask = torch.cat(self._mask, dim=0)
        logits = torch.cat(self.logits, dim=0)
        labels = torch.cat(self.labels, dim=0)

        return IDs, mask, logits, labels

    def compute_metrics(self, loss):

        _, mask, logits, labels = self._finalize()
        
        labels = labels.ravel().long().numpy() # (N,)
        preds = logits.argmax(dim=1).long().numpy() # (N,)
        probs = logits.softmax(dim=1).numpy() # (N, num_classes)
        
        num_classes = probs.shape[-1]

        metrics = { 'loss': float(np.mean(loss)),
                    'F1': f1_score(labels, preds, average='weighted', zero_division=0),
                    'recall': recall_score(labels, preds, average='weighted', zero_division=0),
                    'precision': precision_score(labels, preds, average='weighted', zero_division=0),
                    'accuracy': accuracy_score(labels, preds),
                    'confusion_matrix': confusion_matrix(labels, preds, labels=list(range(num_classes)))
                }

        # ROC-AUC (unweighted average across classes)
        if num_classes == 2: # if binary
            if np.unique(labels).size < 2: # makes sure both classes are present
                metrics["roc_auc"] = np.nan
                metrics["pr_auc"] = np.nan
            else:
                metrics["roc_auc"] = roc_auc_score(labels, probs[:,1])
                metrics["pr_auc"] = average_precision_score(labels, probs[:,1])
        else:
            if np.unique(labels).size < num_classes: # makes sure all classes are present
                metrics["roc_auc"] = np.nan
                metrics["pr_auc"] = np.nan
            else:
                metrics["roc_auc"] = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
                
                Y = label_binarize(labels, classes=list(range(num_classes)))
                metrics["pr_auc"] = average_precision_score(Y, probs, average="macro")
                
            # One-vs-rest AUC per class -- ONLY FOR classes > 2
            for i in range(num_classes):
                bin_label = (labels == i).astype(int)
                if np.unique(bin_label).size < 2:
                    metrics[f"{i}vr_roc_auc"] = np.nan
                    metrics[f"{i}vr_pr_auc"] = np.nan
                else:
                    metrics[f"{i}vr_roc_auc"] = roc_auc_score(bin_label, probs[:, i])
                    metrics[f"{i}vr_pr_auc"] = average_precision_score(bin_label, probs[:, i])
                
        return metrics
