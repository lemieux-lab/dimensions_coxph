# Impacts of dimensionality reduction on Cox-Proportional Hazard in Acute Myeloid Leukemia  

## 1. Introduction
In this report, we will investigate a subset of Gene Expression profiles coming from the Leucegene dataset. We will use both PCA, and t-SNE to perform dimensionality reduction on the data. This will provide visualizations of the data as well as highlighting putative cancer subgroups by eye. By correlating the most contributing genes to the PCA, we will assign each PC to a major ontology if it exists. 

## 2. Generating the Data

### 2.0 Initializing the program, setting up environment variables (taken from [Source](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) )

To install venv via pip
```{bash}
python3 -m pip install --user virtualenv
```

Then, create  activate the environment (Only first time)
```
python3 -m venv env
```

**Activate environment (everytime to run)**

**On windows**

do this before activating. (in a powershell)*
```
Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope CurrentUser
```
Then, to activate the environment. One of the options.
```
./env/Scripts/Activate.ps1
./env/Scripts/activate
```

**On Unix**
```
source env/bin/activate
```

**Install required packages (Only first time)**
```
python3 -m pip install -r requirements.txt
```

Then finally, to run the program, run :
```{python3}
python3 main.py 
```
The other commands will be explained.

### 2.0.1: Experiment Book

## FIG1/FIG4b,c,e,f
```
# generate scores data, cross-validation and bootstrapping concordance indices
python3 main.py --run_experiment 1 -BN 10000 -N_FOLDS 10 -O FIG1
```

## FIG2 
```
# generate Pearson-moment correlation logistic regression from GE to CF heatmaps results 
python3 main.py --run_experiment 2 -C lgn_pronostic -O FIG2
```

## FIG3/FIG4a,d 
```
## performance by dimension sweep (Leucegene)
python3 main.py --run_experiment 1 -C lgn_pronostic -P PCA CF-PCA RSelect RPgauss_var -IN_D 1 50 -N_REP 1000 -O RES/FIGS/FIG3 
## performance of LSC17
python3 main.py --run_experiment 1 -C lgn_pronostic -P LSC17 -IN_D 17 18 -N_REP 1000 -O RES/FIGS/FIG3
```
