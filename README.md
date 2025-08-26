# Project Overview

This project implements a customized weighted view pipeline incorporating a Transformer Encoder for molecular property prediction using the QM7 dataset. The pipeline processes molecular data through atomic embeddings, weights, and a transformer-based architecture to generate predictions.

# Data
The data/ directory contains the QM7 dataset, stored in the qm7.mat file. This dataset includes molecular structures and properties used for training and evaluating the Transformer Encoder model.

# Notebooks and Scripts

The project includes several Python scripts and Jupyter notebooks for data processing, model implementation, and analysis:





qm7_transformercode.py: Implements the core functionality for splitting molecular views, creating a dictionary of single atom properties, calculating Coulomb matrices, and generating atomic embeddings.



qm7_weightedviews.py: Contains methods to create weighted views, with configurable switches to switch between full and truncated views.



qm7_work_full_split.ipynb: Provides analysis of full and truncated views. Requires activating the switch in qm7_weightedviews.py to generate truncated views.



qm7_coulomb_only.ipynb: Contains experiments incorporating single and pair properties in the full split configuration.



qm7_transformer_padded_zero.ipynb: Implements a customized Transformer Encoder with zero-padding for handling variable-sized inputs.

# Usage

To reproduce the results:





Ensure the QM7 dataset is available in the data/ directory.



Run the scripts or notebooks in the order described above, starting with data preprocessing (qm7_transformercode.py and qm7_weightedviews.py).



Use the notebooks for specific analyses or experiments, ensuring any required switches are set appropriately.
