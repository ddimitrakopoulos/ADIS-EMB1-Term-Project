The **Graph Isomorphism Network (GIN)** code is split into two Python Notebooks:
- **gin\_original.ipyn** contains the entire implementation of the GIN architecture, the dataset loading and esge case handling and the Classification, Clustering and Stability Analysis tasks all in separate cells, for better understanding of the code and its structure. 
- **gin\_experiments.ipynb** is a modified version of the above where all the tasks are in the same cell and nested in side a for loop in order to be able to test all embedding sizes for a single dataset with a single run of the notebook and save the results efficiently.

To run either of the files simply download the corresponding notebook and upload it into either Kaggle or Google Colab and set the environment in order to use on of the available GPUs and select Run All.
