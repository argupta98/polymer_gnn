import os
import argparse
import torch
import random
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from utils.load_networkx import networkx_feat
from utils.macro_dataset import MacroDataset
from utils.macro_supervised_hyperparameter import MacroSupervised
import optuna
import optunahub
from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution

from utils.HEBO.sampler import HEBOSampler


class HyperparameterOptimization:
    def __init__(self, seed, GPU, model, num_epochs, num_trials):
        # Generic parameters
        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)
        self.DESCRIPTORS = './unique_descriptors.json'
        self.SEED = seed
        self.TASK = 'classification'
        self.MODEL = model
        self.LABELNAME = 'immunogenic'
        self.NUM_EPOCHS = num_epochs
        self.NUM_WORKERS = os.cpu_count()
        self.MODEL_PATH = './past_trials/' + model + '/hyperparameter_optimization'
        self.SAVE_MODEL = True
        self.SAVE_OPT = True
        self.SAVE_CONFIG = True
        self.CUSTOM_PARAMS = {}

        # Train/validation dataset setup
        self.MON_SMILES = './tables/SMILES_peptides_monomer.txt'
        self.BOND_SMILES = './tables/SMILES_peptides_bond.txt'
        self.TXT_DATA_PATH = './dataset/classification/'
        self.DF_PATH = './tables/immuno_peptides.txt'
        self.SPLIT = '0.7,0.3'
        self.SCALER_TYPE = 'MinMax'
        self.NX_GRAPHS, self.SCALERS = networkx_feat(
            TXT_DATA_PATH=self.TXT_DATA_PATH,
            MON_SMILES=self.MON_SMILES,
            BOND_SMILES=self.BOND_SMILES,
            DESCRIPTORS=self.DESCRIPTORS,
            SPLIT=self.SPLIT,
            SEED=self.SEED,
            SCALER_TYPE=self.SCALER_TYPE
        )
        self.dgl_dict = MacroDataset(
            DF_PATH=self.DF_PATH,
            TASK=self.TASK,
            LABELNAME=self.LABELNAME,
            MODEL=self.MODEL,
            NX_GRAPHS=self.NX_GRAPHS
        )

        # Inference dataset setup
        self.MON_SMILES_POLY = './tables_poly/SMILES_polymers_monomer.txt'
        self.BOND_SMILES_POLY = './tables_poly/SMILES_polymers_bond.txt'
        self.TXT_DATA_PATH_POLY = './shoshana_polymers/dataset_uniform/classification'
        self.DF_PATH_POLY = './tables_poly/immuno_polymers.txt'
        self.NX_GRAPHS_INFER = networkx_feat(
            TXT_DATA_PATH=self.TXT_DATA_PATH_POLY,
            MON_SMILES=self.MON_SMILES_POLY,
            BOND_SMILES=self.BOND_SMILES_POLY,
            DESCRIPTORS=self.DESCRIPTORS,
            SCALER=self.SCALERS
        )
        self.dgl_dict_infer = MacroDataset(
            DF_PATH=self.DF_PATH_POLY,
            TASK=self.TASK,
            LABELNAME=self.LABELNAME,
            MODEL=self.MODEL,
            NX_GRAPHS=self.NX_GRAPHS_INFER
        )

    def objective(self, trial):
        # The parameters are automatically suggested based on the search space
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

        params = {
            'lr': lr,
            'weight_decay': weight_decay,
        }

        if self.MODEL not in ["Weave", "MPNN"]:
            dropout = trial.suggest_float('dropout', 0.0, 0.5)
            params['dropout'] = dropout

        new_path = self.MODEL_PATH + '/model_' + str(trial.number)
        macro_supervised = MacroSupervised(
            MacroDataset=self.dgl_dict,
            MODEL=self.MODEL,
            NUM_EPOCHS=self.NUM_EPOCHS,
            NUM_WORKERS=self.NUM_WORKERS,
            DESCRIPTORS=self.DESCRIPTORS,
            CUSTOM_PARAMS=params,
            INFERENCE=self.dgl_dict_infer,
            MODEL_PATH=new_path,
            SAVE_MODEL=self.SAVE_MODEL,
            SAVE_OPT=self.SAVE_OPT,
            SAVE_CONFIG=self.SAVE_CONFIG
        )

        return macro_supervised.main()

    def run_hpo(self, num_trials, rand_samples):
        # Initialize Optuna Study
        # module = optunahub.load_module("samplers/hebo")
        # HEBOSampler = module.HEBOSampler

        search_space = {
            'lr': FloatDistribution(1e-5, 1e-2, log=True),
            'weight_decay': FloatDistribution(1e-6, 1e-3, log=True),
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
    parser.add_argument("--rand_samples", type=int, required=True, help="Number of initial random samples to explore landscape")

    args = parser.parse_args()

    optimizer = HyperparameterOptimization(seed=args.seed, GPU=args.GPU, model=args.model, num_epochs=args.num_epochs,
                                           num_trials=args.num_trials)
    optimizer.run_hpo(num_trials=args.num_trials, rand_samples=args.rand_samples)
