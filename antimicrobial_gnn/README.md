# Generating Polymer Sequences
Generate polymer compositions, sequences, and features.

***TOC***
- [Generating Polymer Sequences](#generating-polymer-sequences)
  - [compositions](#compositions)
  - [Generating polymer sequences from compositions](#generating-polymer-sequences-from-compositions)
    - [Composition to sequence](#composition-to-sequence)
    - [Sequence to SMILES](#sequence-to-smiles)


## Generating Polymer Compositions

> Compositions are the relative proportions of monomers in a polymer.  
    (monA: 0.2, monB: 0.3, monC: 0.5)

Compositions are generated using the notebook at `shoshana_polymers/generate_polymer_combinations.ipynb`.  

This generates a `.csv` file named `polymer_combinations.csv` with 7,026,137 possible polymer compositions at time of writing (09/14/24).

### Composition Heuristics
- Considered Monomers
  - Tma,C[N+](C)(C)CCCNC(=O)C=C,
  - Aeg,NC(=[NH2+])NCCNC(=O)C=C,
  - Mo,C=CC(=O)N1CCOCC1,
  - Mep,COCCCNC(=O)C=C,
  - Ni,CC(C)NC(=O)C=C,
  - Phe,C=CC(=O)Nc1ccccc1,
  - Do,CCCCCCCCCCCCNC(=O)C=C,
  - Bam,CCCCNC(=O)C=C,
  - Oct,CCCCCCCCNC(=O)C=C,
  - Olam,CCCCCCCC/C=C\CCCCCCCCNC(=O)C(=C),
  - Bmam,CCCCOCNC(=O)C=C,
  - Tmb,CC(C)(C)CC(C)(C)NC(=O)C=C,
- 5 wt% increments
- XYZ Hydrophobicty
- XYZ ABC

## Generating polymer sequences from compositions
> Converting compositions to sequence of monomers. Sequences represented as concatenated monomer acronyms.

In reality, polymer sequences are dictated by the reaction conditions and kinetics of the polymerization reaction.  

Here, we generate polymer sequences using sampling without replacement from a bag of monomers with the prescribed composition of monomers.


### Generating Sequences
`src/make_text_sequences.py` contains a script that will generate polymer sequences from compositions in `polymer_combinations.csv`.
- parallelizes the process across cpus
- splits the polymer_combinations.csv evenly across each cpu
- applies `sample` class to generate polymer sequences
  - `src/polymerization.py` contains a sample class that will generate polymer sequences when provided with:
    ```python
    monomers: list of monomer acronyms
    mol_dist: list of floats representing the mol fractions of each monomer
    sampling_method: sampling without replacement or uniform
    n: number of sequences to generate per composition
    ```

- save polymer sequences to disk in parquet format

#### Tests
Test sequence generation with:
```bash
python -m unittest -v tests/test_polymer_data_preparation.py  
```

## Sequence to SMILES


