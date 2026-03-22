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


class HyperparameterOptimization:
    def __init__(self, seed, GPU, model, num_epochs, num_trials):
        # Generic parameters
        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)
        self.DESCRIPTORS = 'unique_descriptors.json'
        self.SEED = seed
        self.TASK = 'classification'
        self.MODEL = model
        self.LABELNAME = 'immunogenic'
        self.NUM_EPOCHS = num_epochs
        self.NUM_WORKERS = os.cpu_count()
        self.MODEL_PATH = 'past_trials/' + model + '/hyperparameter_optimization'
        self.SAVE_MODEL = True
        self.SAVE_OPT = True
        self.SAVE_CONFIG = True
        self.CUSTOM_PARAMS = {}

        # Train/validation dataset setup
        self.MON_SMILES = 'tables/SMILES_peptides_monomer.txt'
        self.BOND_SMILES = 'tables/SMILES_peptides_bond.txt'
        self.TXT_DATA_PATH = 'dataset_abridged/classification/'
        self.DF_PATH = 'tables/immuno_peptides.txt'
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
        self.MON_SMILES_POLY = 'tables_poly/SMILES_polymers_monomer.txt'
        self.BOND_SMILES_POLY = 'tables_poly/SMILES_polymers_bond.txt'
        self.TXT_DATA_PATH_POLY = 'shoshana_polymers/dataset_uniform_abridged/classification'
        self.DF_PATH_POLY = 'tables_poly/immuno_polymers.txt'
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

    def get_search_space(self):
        # Define model-specific search spaces
        if self.MODEL == "MPNN":
            return {
                'node_out_feats': IntDistribution(32, 128),
                'edge_hidden_feats': IntDistribution(64, 256),
                'num_step_message_passing': IntDistribution(3, 8),
                'num_step_set2set': CategoricalDistribution([2, 3, 4]),
                'num_layer_set2set': CategoricalDistribution([2, 3, 4]),
            }
        elif self.MODEL == "Weave":
            return {
                'num_gnn_layers': IntDistribution(2, 6),
                'gnn_hidden_feats': IntDistribution(64, 256),
                'graph_feats': IntDistribution(16, 128),
            }
        elif self.MODEL == "GAT":
            return {
                'gnn_hidden_feats': IntDistribution(16, 64),
                'num_heads': IntDistribution(2, 8),
                'predictor_hidden_feats': IntDistribution(64, 256),
                'num_gnn_layers': IntDistribution(2, 6),
            }
        elif self.MODEL == "GCN":
            return {
                'gnn_hidden_feats': IntDistribution(128, 512),
                'predictor_hidden_feats': IntDistribution(32, 128),
                'num_gnn_layers': IntDistribution(3, 6),
            }
        elif self.MODEL == "AttentiveFP":
            return {
                'num_layers': IntDistribution(2, 5),
                'num_timesteps': IntDistribution(1, 3),
                'graph_feat_size': IntDistribution(16, 128),
            }
        else:
            raise ValueError("Model not supported")

    def objective(self, trial):
        # Fetch the search space dynamically based on the model
        search_space = self.get_search_space()

        # Create the parameters dictionary by calling trial.suggest_* based on the distribution type
        params = {}
        for param, dist in search_space.items():
            if isinstance(dist, optuna.distributions.IntDistribution):
                params[param] = trial.suggest_int(param, dist.low, dist.high)
            elif isinstance(dist, optuna.distributions.CategoricalDistribution):
                params[param] = trial.suggest_categorical(param, dist.choices)

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

    def run_hpo(self, num_trials):
        # Initialize Optuna Study
        module = optunahub.load_module("samplers/hebo")
        HEBOSampler = module.HEBOSampler

        search_space = self.get_search_space()  # Use the model-specific search space

        sampler = HEBOSampler(search_space)

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

    args = parser.parse_args()

    optimizer = HyperparameterOptimization(seed=args.seed, GPU=args.GPU, model=args.model, num_epochs=args.num_epochs,
                                           num_trials=args.num_trials)
    optimizer.run_hpo(num_trials=args.num_trials)
