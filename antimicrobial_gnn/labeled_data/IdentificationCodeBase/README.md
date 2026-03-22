# Readme

## Requirements

1. numpy==1.19.4  
1. pandas==0.25.3  
1. python==3.6.12  
1. scikit-learn==0.23.2  
1. torch==1.7.0+cu101  
1. torchvision==0.8.0+cu101  
1. xgboost==1.3.0  
1. tqdm==4.54.1  

## Installation

```python
pip install -r requirements.txt
```

## Data preparation

### Generate training and testing files
Will generate the training data from GRAMPA dataset (POSITIVES) and from UniProt Database (NEGATIVES)
```python
python generate_sample.py
```

### Generate sequences for searching  

> Use ```sequence_generated.py``` in ```./sequence_generated``` to generate the sequence for customized searching space, we offered sequences for peptides which length is 6 and the script to generate peptide sequences of length 7 in folder ```./sequence_generated```.

### Generate strutual data for sequences  

> Use ```cal_pep_des.py``` in ```./featured_data_generated``` to generate structual data for Classification and Ranking stage from the sequences derived in the last step.

## Model Training  

### Pipeline training

> Use ```train.py``` to get all the params for the three models(Classifcation, Ranking, Regressing). You can use customized training data or data generated from Grampa dataset.

### Incremental learning  

> Use ```lstm_fine_tune.py``` for incremental learning. The augmented data was provided in folder ```./data/origin_data```. Using customized data validated in other wet-lab settings is optional.

## Searching for antimicrobial sequences

> Use ```predict.py``` to get the final searching result. For a vast searching space, you may use 'chunk' mechanism to avoid RAM shortage.
