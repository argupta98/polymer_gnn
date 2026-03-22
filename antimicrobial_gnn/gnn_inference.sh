#!/bin/bash

# Define the list of model numbers
model_numbers=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24")

target_dir="no_dups_samples_one_replicate"
# Iterate over each model number
for model in "${model_numbers[@]}"; do
  # Construct the model path by replacing the model number
  model_path="inference/trained_models/round3_models/trial_${model}/"

  echo "($model_path)"
  # Execute the Python command with the current model path
  python -m inference.inference --model_path "$model_path" --target_dir "$target_dir" --num_gpus 4
done
