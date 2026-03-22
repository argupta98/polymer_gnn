"""
Utility classes to create polymer sequences from compositions of monomers
and to generate SMILES sequences from those sequences.

We first generate sequences in Text format for compatibility with graph neural networks.
We create SMILES sequence for each sequence to eventurally create whole polymer features using
RDkit.

GNNs only use monomer level features.
"""

import json
import numpy as np
import random
import re
from iteround import saferound
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from pathlib import Path

root_dir = Path(__file__).parent.parent


class sample:
    def __init__(
        self, monomers, DP, mol_dist, sampling_method, n, batch=False, encoded=False
    ):
        self._monomers = monomers
        self._DP = DP
        self._dist = mol_dist
        self._sampling_method = sampling_method
        self._n = n
        self._batch = batch

        self._cum_dist = np.cumsum(self._dist)

        self._num_of_monomers = {
            self._monomers[i]: self._dist[i] * self._DP
            for i in range(len(self._monomers))
        }
        self._num_of_monomers = saferound(self._num_of_monomers, 0)

        if self._batch:
            self._num_of_monomers = {
                key: val * self._n for key, val in self._num_of_monomers.items()
            }

        self.samples = self.generate_n_samples(encoded=encoded)

    def determine_monomer(self, x, cum_dist):
        for i, partition in enumerate(cum_dist):
            if x < partition:
                return i

    def uniform_sample(self):
        polymer = []
        for i in range(self._DP):
            x = np.random.uniform(0, 1)
            polymer.append(self._monomers[self.determine_monomer(x, self._cum_dist)])

        polymer = "".join(polymer)

        return polymer

    def sample_wo_replacement(self):
        dist = self._num_of_monomers

        all_monomers = []
        for key, value in dist.items():
            all_monomers.extend([key] * int(value))

        random.shuffle(all_monomers)

        polymer = "".join(all_monomers)

        return polymer

    def generate_n_samples(self, encoded=False):
        if self._sampling_method == "wo_replacement" and self._batch:
            sample = self.sample_wo_replacement()

            sample_split = re.findall("[A-Z][^A-Z]*", sample)
            x = len(sample_split) // self._n

            if encoded:
                monomer_encoding = json.load(
                    open(
                        root_dir
                        / "shoshana_polymers/monomer_data/monomer_encoding.json",
                        "r",
                    )
                )
                sample_split = [monomer_encoding.get(s) for s in sample_split]

            samples = [
                "".join(sample_split[i : i + x]) for i in range(0, len(sample_split), x)
            ]

            return samples
        else:
            samples = set()

            while len(samples) < self._n:
                if self._sampling_method == "uniform":
                    samples.add(self.uniform_sample())
                elif self._sampling_method == "wo_replacement" and not self._batch:
                    samples.add(self.sample_wo_replacement())

            return list(samples)


class polymerize(object):
    def __init__(self, sample, SMILES):
        self._sample = sample  # string of monomers

        sample_split = re.findall(
            "[A-Z][^A-Z]*", self._sample
        )  # split sample by capital letters
        self._DP = len(sample_split)

        sample_smiles = [SMILES[mon] for mon in sample_split]
        self._sample_mol = [Chem.MolFromSmiles(reactant) for reactant in sample_smiles]

    def reinitialize_polymer(self, product):
        if len(product) > 1:
            return "Too many products"
        else:
            product_smiles = Chem.MolToSmiles(product[0][0])
            reinit_product = Chem.MolFromSmiles(product_smiles)
            return reinit_product

    def initialize_reaction(self):
        rxn1 = AllChem.ReactionFromSmarts(
            "[C:0]=[CH2:1].[CH2:2]=[C:3]>>[Kr][C:0][C:1][C:3][C:2][Xe]"
        )

        A = self._sample_mol.pop(0)
        B = self._sample_mol.pop(0)
        product = rxn1.RunReactants((A, B))
        self.polymer = self.reinitialize_polymer(product)

        return product

    def propagate_reaction(self):
        rxn2 = AllChem.ReactionFromSmarts(
            "[C:0][C:1][C:2][C:3][Xe].[C:4]=[CH2:5]>>[C:0][C:1][C:2][C:3][C:4][C:5][Xe]"
        )

        A = self.polymer
        B = self._sample_mol.pop(0)
        product = rxn2.RunReactants((A, B))
        self.polymer = self.reinitialize_polymer(product)

        return product

    def terminate_reaction(self):
        if self._DP > 2:
            # terminate & remove Xe
            rxn3 = AllChem.ReactionFromSmarts(
                "[C:0][C:1][C:2][C:3][Xe].[C:4]=[CH2:5]>>[C:0][C:1][C:2][C:3][C:4][C:5]"
            )

            A = self.polymer
            B = self._sample_mol.pop(0)

            product = rxn3.RunReactants((A, B))
            self.polymer = self.reinitialize_polymer(product)

            # (removes Kr)
            rxn4 = AllChem.ReactionFromSmarts(
                "[C:0][C:1][C:2][C:3][Kr]>>[C:0][C:1][C:2][C:3]"
            )

            A = self.polymer
            product = rxn4.RunReactants((A,))
            self.polymer = product[0][0]
        else:
            rxn3 = AllChem.ReactionFromSmarts(
                "[C:0]=[CH2:1].[CH2:2]=[C:3]>>[C:0][C:1][C:3]=[C:2]"
            )

            A = self._sample_mol.pop(0)
            B = self._sample_mol.pop(0)
            product = rxn3.RunReactants((A, B))
            self.polymer = product[0][0]

    def run_reaction(self):
        if self._DP > 2:
            self.initialize_reaction()

        if self._DP > 3:
            for i in range(self._DP - 3):
                self.propagate_reaction()

        self.terminate_reaction()

    def get_smiles(self):
        return Chem.MolToSmiles(self.polymer)

    def draw_diagram(self):
        return Draw.MolToImage(self.polymer, size=(500, 500))
