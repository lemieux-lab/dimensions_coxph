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
## FIG1
```
# generate scores data, leave-one-out bootstraps c_index
python3 main.py --run_experiment 2 -C lgn_pronostic -P CF LSC17 PCA17 CF-PCA17 CF-LSC17 -M CPH -N_REP 10000 -CYT -O RES/FIGS/FIG4
```

## FIG2 
```
# generate log_reg GE to CF results (leave-one-out)
python3 main.py --run_experiment 2 -C lgn_pronostic -P PCA17 LSC17 PCA300 -M LOG_REG -O RES/FIGS/FIG2
```

## FIG3
```
## performance by dimension sweep (Leucegene)
python3 main.py --run_experiment 1 -C lgn_pronostic -P PCA CF-PCA RSelect RPgauss_var -IN_D 1 50 -N_REP 1000 -O RES/FIGS/FIG3 
## performance of LSC17
python3 main.py --run_experiment 1 -C lgn_pronostic -P LSC17 -IN_D 17 18 -N_REP 1000 -O RES/FIGS/FIG3
```

## FIG4
```
## performance by dimension sweep (Leucegene Intermediate)
python3 main.py --run_experiment 1 -C lgn_intermediate -P PCA -N_REP 1000 -IN_D 1 50 -O RES/FIGS/FIG4

## performance by dimension sweep (TCGA)
python3 main.py --run_experiment 1 -C TCGA -P PCA -N_REP 10000 -IN_D 1 5 -O RES/FIGS/FIG4

## performance of LSC17, PCAX (found precedently)
python3 main.py --run_experiment 2 -C lgn_intermediate -P PCAX LSC17 -M CPH -N_REP 10000 -O RES/FIGS/FIG4

```