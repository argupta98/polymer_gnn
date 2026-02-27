import os
import argparse
import torch
import random
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole

from utils.create_dataset_class import DataSet
from utils.multiclass_NN import multiclass_NN
from utils.split_dataset import split_dataset
from utils.scale_graph_features import scale

import optuna
import optunahub
from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution

from utils.HEBO.sampler import HEBOSampler

import numpy as np
import joblib

class HyperparameterOptimization:
    def __init__(self, seed, GPU, model, num_epochs, num_trials):
        # Generic parameters
        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)
        self.DESCRIPTORS = 'monomer_data/unique_descriptors.json'
        self.SEED = seed
        self.TASK = 'classification'
        self.MODEL = model
        self.LABELNAME = '3_classes'
        self.NUM_EPOCHS = num_epochs
        self.NUM_WORKERS = len(os.sched_getaffinity(0))
        self.MODEL_PATH = './past_trials/' + model + '/hyperparameter_optimization'
        self.SAVE_MODEL = True
        self.SAVE_OPT = True
        self.SAVE_CONFIG = True
        self.CUSTOM_PARAMS = {}

        # Train/validation dataset setup
        self.db_file = 'data_preprocessing/db.csv'
        self.MIXED = True
        self.SMILES = 'data_preprocessing/SMILES.txt'

        data_split = split_dataset(db_file = self.db_file,
                                   class_col = self.LABELNAME,
                                   val_size = 0.3,
                                   test_size = 0.3,
                                   mixed = self.MIXED,
                                   random_state = self.SEED,)
        
        # Scale dataset using only node and edge features in train set
        self.features = scale(data_split['train'], self.SMILES, self.DESCRIPTORS)
        
        # create dataloader
        self.dataset = DataSet(data_split, self.features, self.LABELNAME, self.MIXED, self.TASK, self.MODEL)

    def objective(self, trial):
        # The parameters are automatically suggested based on the search space
        lr = trial.suggest_float('lr', 1e-5, 0.5, log=True) # original: 1e-2
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 0.5, log=True) # original: 1e-3

        params = {
            'lr': lr,
            'weight_decay': weight_decay,
        }

        if self.MODEL not in ["Weave", "MPNN"]:
            dropout = trial.suggest_float('dropout', 0.0, 0.5)
            params['dropout'] = dropout

        new_path = self.MODEL_PATH + '/model_' + str(trial.number)
        multiclassNN = multiclass_NN(dataset=self.dataset, 
                               MODEL=self.MODEL, 
                               NUM_EPOCHS=self.NUM_EPOCHS, 
                               NUM_WORKERS=self.NUM_WORKERS,
                               DESCRIPTORS=self.DESCRIPTORS,
                               CUSTOM_PARAMS=params,
                               MODEL_PATH=new_path,
                               SAVE_MODEL=self.SAVE_MODEL,
                               SAVE_OPT=self.SAVE_OPT,
                               SAVE_CONFIG=self.SAVE_CONFIG)

        return multiclassNN.main()

    def run_hpo(self, num_trials, rand_samples):
        # Initialize Optuna Study
        # module = optunahub.load_module("samplers/hebo")
        # HEBOSampler = module.HEBOSampler

        search_space = {
            'lr': FloatDistribution(1e-5, 0.5, log=True),
            'weight_decay': FloatDistribution(1e-6, 0.5, log=True),
        }

        # Only add 'dropout' to the search space if the model is not "Weave" or "MPNN"
        if self.MODEL not in ["Weave", "MPNN"]:
            search_space['dropout'] = FloatDistribution(0.0, 0.5)
        
        sampler = HEBOSampler(search_space, rand_sample=rand_samples)

        # Define the path for your database
        db_directory = 'past_trials/' + self.MODEL
        db_path = os.path.join(db_directory, self.MODEL + '_hpo.db')
        db_url = f'sqlite:///{db_path}'

        # Create the directory if it doesn't exist
        os.makedirs(db_directory, exist_ok=True)

        ### DUMP CORRECT FEATURES
        joblib.dump(self.features, db_directory + '/features.pkl')

        # Create a study object with the SQLite storage
        study = optuna.create_study(
            study_name="hebo",
            sampler=sampler,
            direction="minimize",
            storage=db_url,  # Use SQLite as the storage backend
            load_if_exists=True  # Load the study if it already exists
        )

        study.optimize(self.objective, n_trials=num_trials)


# Main block to parse command-line arguments and run optimization
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Hyperparameter optimization")
    parser.add_argument("--seed", type=int, required=True, help="Seed for splitting data")
    parser.add_argument("--GPU", type=str, required=True, help="GPU to use for the optimization")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--num_epochs", type=int, required=True, help="Number of epochs per trial")
    parser.add_argument("--num_trials", type=int, required=True, help="Number of trials for optimization")
    parser.add_argument("--rand_samples", type=int, required=True, help="Number of initial random samples to explore hyperparameter landscape")

    args = parser.parse_args()

    optimizer = HyperparameterOptimization(seed=args.seed,
                                           GPU=args.GPU,
                                           model=args.model,
                                           num_epochs=args.num_epochs,
                                           num_trials=args.num_trials)
    
    optimizer.run_hpo(num_trials=args.num_trials, rand_samples=args.rand_samples)
