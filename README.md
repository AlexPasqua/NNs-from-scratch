# ML-project
Project for the Machine Learning course @ University of Pisa
---

### Overview
1. [Directory structure](#directory-structure)
2. [Quick start](#quick-start)

### Directory structure
```
ML-project
  |-- src
      |-- network.py
      |-- functions.py
```

## Quick start
Install Python:<br>
`sudo apt install python3`

Install pip:<br>
`sudo apt install --upgrade python3-pip`

Install requirements:<br>
`python -m pip install --requirement requirements.txt`

Create the NN and execute an inference:<br>
`python src/network.py [--input_dim INPUT_DIM] [--inputs INPUTS] [--units_per_layer UNITS_PER_LAYER] [--act_funcs ACT_FUNCS]`<br>
Arguments:
* `INPUT_DIM`: integer representing the length of each data record
* `INPUTS`: sequence of floats representing a single data record for inference
* `UNITS_PER_LAYER`:  sequence of integers. The i-th number represents the number of units in the i-th layer
* `ACT_FUNCS`: sequence of strings. The i-th string represents the activation function of the units in the i-th layer
    * This parameter's values have to be chosen among {`relu`, `sigmoid`}
