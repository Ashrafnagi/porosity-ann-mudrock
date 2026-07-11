# porosity-ann-mudrock
The source codes developed and used in this study for porosity prediction using artificial neural networks (ANN), clustering-assisted modeling, and feature reduction techniques are publicly available in this repository.

The repository includes MATLAB and Python implementations corresponding to:

ANN models for the original dataset
Cluster-based models (CG-A and CG-B)
PCA-based dimensionality reduction models
Correlation-based feature selection models
Data preprocessing and analysis scripts

The ANN models are implemented using MATLAB Neural Network Toolbox functions (e.g., fitnet, Levenberg–Marquardt and quasi-Newton training algorithms), with configurable hidden-layer sizes (30–40 neurons) and transfer functions (purelin and tansig), consistent with the methodology described in this work.

Repository Contents

The repository contains:

Individual MATLAB scripts (.m, .mlx) for each modeling case
Python notebook for data preprocessing and analysis
Example dataset (or sample input format)
Output examples for validation
A README.md file with instructions.

How to Run the Code
1. Download or clone the repository
2. Open data description and analysis jupyter file using python software.
3. Load the dataset (CSV format) and run the code to generate analysis figures and tables, and to create csv files:

      a. clustered dataset for CG-A ANN code.
   
      b. clustered dataset for for CG-B ANN code.
   
      c. dimensional reduced dataset for PCA dataset ANN code.
   
      d. CC reduced dataset for CC-based dataset.
   
5. Open MATLAB or Python environment
6. Load the dataset (CSV format) for each specific code
7. Run the desired script:
   
     ANN_original dataset.mlx
  
     ANN_CG_A.mlx
  
     ANN_CG_B.mlx
  
     ANN_PCA dataset.m
  
     ANN CC-based dataset.mlx
  
6. Execute training and evaluation
   

A test example is included to verify correct execution.


**Requirements**

MATLAB (R2020 or later recommended)

MATLAB Neural Network Toolbox

Python:
- Python 3.x
- Jupyter Notebook
- Libraries:
  - numpy
  - pandas
  - matplotlib
  - scikit-learn
