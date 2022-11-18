# imports
import pdb
from typing import DefaultDict
from collections import OrderedDict
from lifelines import CoxPHFitter
import pandas as pd
import numpy as np
import functions
from tqdm import tqdm 
import os
from torch import nn
import torch
import utils 
## OWN
from optimisers.base_optimisers import Optimiser

class Model:
    def __init__(self):
        self._init_default_F()
        self._init_default_Optimiser()
        pass
    
    def _init_defautlt_Optimiser(self):
        # a suite of action to get optimiser 
        self.Optimiser = Optimiser()
        return self.Optimiser

    def _init_default_F(self):
        self.F = None
    
    def _optimise(self):
        self.Optimiser(self.X, self.F)

    def _train(self):
        self._optimise()
        return 0    

    def fit(self,X):
        self.X = X
        self._train(self.F)
        return X

    def fit_transform(self, X):
        self.X = X
        self.train()
        return self.F(X)