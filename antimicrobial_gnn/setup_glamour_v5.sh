#!/bin/bash

# If a marker file exists, skip the initialization
if [ -f ~/.bashrc_executed ]; then
  # Remove the marker file
  rm ~/.bashrc_executed

  # Continue with the second part of the script

  # Ensure conda is available by sourcing the updated bashrc
  source ~/.bashrc

  # Create the Conda environment
  conda create -n GLAMOUR python=3.9 -y
  conda activate GLAMOUR

  # Use conda run to ensure all subsequent commands run in the environment

  # Use environment pip (not global)
  export PATH=/home/ubuntu/miniconda3/envs/GLAMOUR/bin:$PATH

  # Install Jupyter kernel
  conda install ipykernel -y
  python -m ipykernel install --user --name GLAMOUR --display-name "GLAMOUR"

  # Install DGL with pip (using the specified URL for wheels)
  pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html

  # Install PyTorch with CUDA support
  conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y

  # Install RDKit with Conda
  conda install -c conda-forge rdkit -y

  # Install other packages with pip
  pip install dgllife
  pip install GraKeL
  pip install pydantic
  pip install captum
  pip install matplotlib
  pip install seaborn
  pip install optuna
  pip install optunahub
  pip install HEBO
  pip install pyarrow

  # Get the version of numpy-base
  # NUMPY_BASE_VERSION=\$(conda list numpy-base | grep numpy-base | awk '{print \$2}')

  # Uninstall any pip-installed numpy
  pip uninstall numpy -y

  # Install the matching version of numpy using pip
  # pip install numpy==\$NUMPY_BASE_VERSION
  pip install numpy==1.23.0

  # Clone the GitHub repository
  git clone https:// <YOUR-USERNAME >: <YOUR-TOKEN >@github.com/GabrielGreenstein01/antimicrobial_GNN.git

  # Source the updated bashrc again to make conda available immediately
  source ~/.bashrc

  # Exit the script after the second part is complete
  exit 0
fi

# First part of the script

# Pre-configure the msttcorefonts package to auto-accept the EULA
echo "msttcorefonts msttcorefonts/accepted-mscorefonts-eula select true" | sudo debconf-set-selections

# Install Microsoft TrueType core fonts (to fix the sans-serif issue)
sudo apt-get update
sudo apt-get install msttcorefonts -qq

# Remove matplotlib cache to apply new fonts
rm ~/.cache/matplotlib -rf

# Download Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install Miniconda
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# Initialize Conda
source ~/miniconda3/bin/activate
conda init

# Remove the Miniconda installer to clean up
rm Miniconda3-latest-Linux-x86_64.sh

# Create a marker file to indicate the script has run once
touch ~/.bashrc_executed

# Restart the shell and run this script again
exec bash "$0"
