# Neural networks from scratch
### Project for the Machine Learning course @ University of Pisa<br>

### Overview
1. [Short description](#description)
2. [Directory structure](#directory-structure-only-main-elements)
3. [Quick start](#quick-start)
---

### Description
This project contains the implementation from scratch of neural networks for classification and regression
trained with _Stochastic Gradient Descent_ with _back-propagation_.<br>
For more detailed information check the [report](report.pdf)

### Directory structure (only main elements)
```
ML-project
  |-- src
      |-- network.py
      |-- layer.py
      |-- functions.py
      |-- optimizers.py
      |-- model_selection.py
      |-- weights_initializations.py
      |-- demo.py
  |-- datasets
      |-- cup
          |-- ML-CUP20-TR.csv
          |-- ML-CUP20_TS.csv
          |-- CUP-DEV-SET.csv
          |-- CUP-INTERNAL-TEST.csv
      |-- monks
          |-- monks.names       # description file
          |-- monks-x.train     # the 'x' is the number of the dataset (1, 2, 3)
          |-- monks-x.test
  |-- plots
      |-- ensemble      # where the plots of the constituent models go
      |-- monks         # where the plots of the monks are
  |-- results           # json files with the results of grid searches
  |-- ensemble_models   # json files with the constituent models of the ensemble
```

## Quick start
Install Python:<br>
`sudo apt install python3`

Install pip:<br>
`sudo apt install --upgrade python3-pip`

Install requirements:<br>
`python -m pip install --requirement requirements.txt`

MONKS demo: open the [script](src/demo.py) to find information and instructions.
Then execute it with
```
cd src/
python demo.py
```
