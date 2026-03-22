import torch
import datetime
import os
from dgllife.utils import EarlyStopping


class Stopper_v2(object):
    def __init__(self, savepath, patience=10, filename=None, metric=None):
        '''
        Initializes a Stopper_v2 object
        
        Args:
        mode : str, 'higher' if higher metric suggests a better model or 'lower' if lower metric suggests a better model
        patience : int, number of consecutive epochs with no observed performance required for early stopping
        filename : str or None, filename for storing the model checkpoint. 
        If not specified, we will automatically generate a file starting with 'early_stop'
        based on the current time.
        metric : str or None, metric name
        '''
        
        self._patience = patience
        self._counter = 0
        self._savepath = savepath
        self._filename = filename
        self.best_val_score = None
        self.best_inference_score = None
        self._early_stop = False

    def _check_higher(self, score, prev_best_score):
        return score > prev_best_score

    def _check_lower(self, score, prev_best_score):
        return score < prev_best_score

    def step(self, val_score, model, optimizer, modelname, save_model, save_opt, INFERENCE_SCORE = None):
        '''Updates based on a new score, which is typically model performance on the validation set
        for a new epoch.

        Args:
        score : float, new score
        model : nn.Module, model instance
        optimizer : torch.optim.Adam, Adam instance
        modelname : str, name of model architecture
        save_model : boolean, whether to save full model
        save_opt : boolean, whether to save optimizer

        Returns:
        self._early_stop: boolean, whether an early stop should be performed.
        '''

        self.save_model = save_model
        self.save_opt = save_opt
        
        if self.best_val_score is None:
            self.best_val_score = val_score
            self.save_checkpoint(model, optimizer, set_type = 'val')
        elif self._check_lower(val_score, self.best_val_score):
            self.best_val_score = val_score
            self.save_checkpoint(model, optimizer, set_type = 'val')
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self._patience:
                self._early_stop = True

        if INFERENCE_SCORE is not None:
            if self.best_inference_score is None or self._check_higher(INFERENCE_SCORE, self.best_inference_score):
                self.best_inference_score = INFERENCE_SCORE
                self.save_checkpoint(model, optimizer, set_type = 'inference')
        
        return self._early_stop


    def save_checkpoint(self, model, optimizer, set_type = None):
        '''Saves model when the metric on the validation set gets improved.

        Args:
        model : nn.Module, model instance
        optimizer : torch.optim.Adam, Adam instance
        modelname : str, name of model architecture
        save_model : boolean, whether to save full model
        save_opt : boolean, whether to save optimizer
        '''

        torch.save({'model_state_dict': model.state_dict()}, self._filename)
        
        if self.save_model and self.save_opt:
            torch.save({
                'model': model,
                'optimizer': optimizer
            }, self._savepath + '/' + str(set_type) + '_fullmodel.pt')
        elif self.save_model:
            torch.save(model, self._savepath + '/' + str(set_type) + '_model.pt')
        elif self.save_opt:
            torch.save(optimizer, self._savepath + '/' + str(set_type) + '_optimizer.pt')

    def load_checkpoint(self, model, set_type = None):
        '''Load the latest checkpoint

        Args:
        model : nn.Module, model instance
        '''
        if set_type == None:
            model.load_state_dict(torch.load(self._filename)['model_state_dict'])
        else:
            if self.save_model and self.save_opt:
                checkpoint = torch.load(self._savepath + '/' + str(set_type) + '_fullmodel.pt')
                model.load_state_dict(checkpoint['model'].state_dict())    
            elif self.save_model:
                checkpoint = torch.load(self._savepath + '/' + str(set_type) + '_model.pt')
                model.load_state_dict(checkpoint.state_dict()) 


