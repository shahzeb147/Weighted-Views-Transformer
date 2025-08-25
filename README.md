data:

data folder contain qm7.mat dataset.

Notebook: 

1) qm7_transformercode.py contain views splitter, single atom properties dictionary, Coulomb matrices calculations and atomic embeddings creation.
2) qm7_weightedviews.py contain method to create views. It also contain switches to get truncated views. 
3) qm7_work_full split.ipynb contain analysis about Full and truncated views, require to activate/deactivate a switch from qm7_weightedviews.py to get truncated views.
4) qm7_coulomb only.ipnb contain code for the experiemnts when we added sinle and pair properties in full split. 
5) qm7_transformer_padded_zero.ipynb has customized encoder implimentation.
