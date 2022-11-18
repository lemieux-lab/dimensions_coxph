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
# classes 
class CPH():
    def __init__(self, data, nepochs = 1):

        self.data = data
        self.params = DefaultDict()

    def set_random_params(self, input_size = None):
        # weight decay or L2
        self.params["wd"] = np.power(10, np.random.uniform(-10,-9))
        input_size = min(np.random.randint(2, self.data.folds[0].train.x.shape[1]), 50)
        # set number of input PCs
        self.params["input_size"] = input_size
    
    def set_fixed_params(self, params):
        self.params = params
    
    def _train(self):
        # create lifelines dataset
        ds = pd.DataFrame(self.data.x.iloc[:,:self.params["input_size"]])
        ds["T"] = self.data.y["t"]
        ds["E"] = self.data.y["e"]
        CPH = CoxPHFitter(penalizer = self.params["wd"], l1_ratio = 0.)
        self.model = CPH.fit(ds, duration_col = "T", event_col = "E")
        l = self.model.log_likelihood_
        c = self.model.concordance_index_

    def _train_cv(self, fold_index):
        # create lifelines dataset
        ds = pd.DataFrame(self.data.folds[fold_index].train.x.iloc[:,:self.params["input_size"]])
        ds["T"] = self.data.folds[fold_index].train.y["t"]
        ds["E"] = self.data.folds[fold_index].y["e"]
        CPH = CoxPHFitter(penalizer = self.params["wd"], l1_ratio = 0.)
        self.model = CPH.fit(ds, duration_col = "T", event_col = "E")
        l = self.model.log_likelihood_
        c = self.model.concordance_index_
        return c
    def _valid_cv(self, fold_index):
        test_data = self.data.folds[fold_index].test
        test_features = test_data.x
        test_t = test_data.y["t"]
        test_e = test_data.y["e"]
        out = self.model.predict_log_partial_hazard(test_features)
        l = self.loss(out, test_t, test_e)
        c = functions.compute_c_index(test_t, test_e, out)
        return out, l, c
        
    def set_fixed_params(self, hp_dict):

        self.params = hp_dict

    def _test(self, test_data):
        test_features = test_data.x
        test_t = test_data.y["t"]
        test_e = test_data.y["e"]
        out = self.model.predict_log_partial_hazard(test_features)
        l = self.loss(out, test_t, test_e)
        c = functions.compute_c_index(test_t, test_e, out)
        return out, l, c
    
    def loss(self, out, T, E): 
        return 999 

class CPHDNN(nn.Module):
    def __init__(self, data, nepochs = 1):
        super(CPHDNN, self).__init__()
        self.data = data
        self.params = DefaultDict()
        self.params["device"] = "cuda:0"
        self.params["crossval_nfolds"] = 5
        self.params["epochs"] = nepochs
        self.params["opt_nepochs"] = 1000
        self.params["lr"] = 1e-5
        self.params["c_index_cross_val"] = 0
        self.params["c_index_training"] = 0
        self.params["machine"] = os.uname()[1]
        self.params["process_id"] = os.getpid() 
        self.cols = ["process_id", "crossval_nfolds", "lr", "epochs","input_size","nInPCA",  "wd", "W", "D", "nL","c_index_training", "c_index_cross_val", "cpt_time"]    

        
        self.loss_training = []
        self.loss_valid = []
        self.c_index_training = []
        self.c_index_valid = []
    
    def set_fixed_params(self, hp_dict):

        self.params = hp_dict
        self.setup_stack()
        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.params["lr"], weight_decay = self.params["wd"])        

    def set_random_params(self):
        # weight decay or L2
        self.params["input_size"] = self.data.folds[0].train.x.shape[1] # dataset dependent!
        self.params["wd"] = np.power(10, np.random.uniform(-10, -9)) # V2 reasonable range for WD after analysis on V1 
        self.params["W"] = np.random.randint(3,2048) # V2 Reasonable
        self.params["D"] = np.random.randint(2,4) # V2 Reasonable
        # self.params["dp"] = np.random.uniform(0,0.5) # cap at 0.5 ! (else nans in output)
        self.params["nL"] = np.random.choice([nn.ReLU()]) 
        self.params["ARCH"] = {
            "W": np.concatenate( [[  ## ARCHITECTURE ###
            self.params["input_size"]], ### INSIZE
            np.ones(self.params["D"] - 1) * self.params["W"], ### N hidden = D - 1  
            [1]]).astype(int), ### OUTNODE 
            "nL": np.array([self.params["nL"] for i in range(self.params["D"])]),
            # "dp": np.ones(self.params["D"]) * self.params["dp"] 
        }
        self.params["nInPCA"] = np.random.randint(2,26)
        self.setup_stack()
        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.params["lr"], weight_decay = self.params["wd"])        

    def forward(self, x):
        risk = self.stack(x)
        return risk # torch.clamp(risk, min = -1000, max = 10)
    
    def loss(self, out, T, E): 
        uncensored_likelihood = torch.zeros(E.size())# list of uncensored likelihoods
        for x_i, E_i in enumerate(E): # cycle through samples
            if E_i == 1: # if uncensored ...
                log_risk = torch.log(torch.sum(torch.exp(out[:x_i +1])))
                uncensored_likelihood[x_i] = out[x_i] - log_risk # sub sum of log risks to hazard, append to uncensored likelihoods list
        
        loss = - uncensored_likelihood.sum() / (E == 1).sum() 
        return loss 
    
    def _train(self):
        bs = 24
        N = self.data.x.shape[0]
        self.nbatch = int(np.ceil(N / bs))
        self.data.to(self.params["device"])
        d = self.data
        for e in tqdm(range(self.params["opt_nepochs"]), desc="TRAINING FINAL MODEL"): # add timer 
            n = 0
            loss = 0
            c_index = 0
            for i in range(self.nbatch):
                train_ids = np.arange(i * bs , (i + 1) * bs)
                sorted_ids = torch.argsort(d.y[train_ids,0], descending = True) 
                train_features, train_T, train_E = d.x[sorted_ids], d.y[sorted_ids,0], d.y[sorted_ids,1]
                # train
                #print (f"train features: {train_features.size()}")
                #print (f"train T: {train_T.size()}")
                #print (f"train E: {train_E.size()}")
                #print(f"epoch: {e + 1} [{i+1}/{self.nbatch}]") 
                self.optimizer.zero_grad()
                try:  ### WORKAROUND, NANS in output of model 
                    out = self.forward(train_features)
                    l = self.loss(out, train_T, train_E)
                    if np.isnan(out.detach().cpu().numpy()).any():
                        raise ValueError("NaNs detected in forward pass")  ### WORKAROUND, NANS in output of model 
                    c = functions.compute_c_index(train_T.detach().cpu().numpy(), train_E.detach().cpu().numpy(), out.detach().cpu().numpy())
                    #print(f"c_index: {c}")
                    l.backward() 
        
                    self.optimizer.step()
                    loss += l
                    c_index += c
                    n += 1
                except ValueError:
                    return 0
                # print(f"loss: {loss/n}")
                # print(f"c_index: {c_index/n}")
            self.loss_training.append(loss.item() / n)
            self.c_index_training.append(c_index / n)
    def _train_cv(self, foldn):
        d =  self.data.folds[foldn].train
        
        bs = 24
        N = d.x.shape[0]
        self.nbatch = int(np.ceil(N / bs))
        for e in range(self.params["epochs"]): # add timer 
            n = 0
            loss = 0
            c_index = 0
            for i in range(self.nbatch):
                train_ids = np.arange(i * bs , (i + 1) * bs)
                sorted_ids = torch.argsort(d.y[train_ids,0], descending = True) 
                train_features, train_T, train_E = d.x[sorted_ids], d.y[sorted_ids,0], d.y[sorted_ids,1]
                # train
                #print (f"train features: {train_features.size()}")
                #print (f"train T: {train_T.size()}")
                #print (f"train E: {train_E.size()}")
                #print(f"epoch: {e + 1} [{i+1}/{self.nbatch}]") 
                self.optimizer.zero_grad()
                try:  ### WORKAROUND, NANS in output of model 
                    out = self.forward(train_features)
                    l = self.loss(out, train_T, train_E)
                    if np.isnan(out.detach().cpu().numpy()).any():
                        raise ValueError("NaNs detected in forward pass")  ### WORKAROUND, NANS in output of model 
                    c = functions.compute_c_index(train_T.detach().cpu().numpy(), train_E.detach().cpu().numpy(), out.detach().cpu().numpy())
                    #print(f"c_index: {c}")
                    l.backward() 
        
                    self.optimizer.step()
                    loss += l
                    c_index += c
                    n += 1
                except ValueError:
                    return 0
                # print(f"loss: {loss/n}")
                # print(f"c_index: {c_index/n}")
            self.loss_training.append(loss.item() / n)
            self.c_index_training.append(c_index / n)
            # test
            # for i, valid_data in enumerate(valid_dataloader):
            #     sorted_ids = torch.argsort(valid_data["t"], descending = True)
            #     valid_features, valid_T, valid_E = valid_data["data"][sorted_ids], valid_data["t"][sorted_ids], valid_data["e"][sorted_ids]
            #     l, c = self.test(valid_features, valid_T, valid_E)
            #     print(f"valid loss: {l}")
            #     print(f"valid c_index: {c}")
            #     self.loss_valid.append(l.item())
            #     self.c_index_valid.append(c)
    
    def _valid_cv(self, foldn):
        # forward prop
        # loss
        # c_index
        d = self.data.folds[foldn].test
        valid_features = d.x
        valid_t = d.y[:,0]
        valid_e = d.y[:,1]
        out = self.forward(valid_features)
        l = self.loss(out, valid_t, valid_e)
        c = functions.compute_c_index(valid_t.detach().cpu().numpy(),valid_e.detach().cpu().numpy(), out.detach().cpu().numpy())
        return out, l , c
    
    def _test(self, test_data):
        test_data.to(self.params["device"])
        test_features = test_data.x
        test_t = test_data.y[:,0]
        test_e = test_data.y[:,1]
        out = self.forward(test_features)
        l = self.loss(out, test_t, test_e)
        c = functions.compute_c_index(test_t.detach().cpu().numpy(), test_e.detach().cpu().numpy(), out.detach().cpu().numpy())
        return out, l, c

    def setup_stack(self):
        
        stack = []
        print("Setting up stack... saving to GPU")
        for layer_id in range(self.params["D"]):
            stack.append([
            [f"Linear_{layer_id}", nn.Linear(self.params["ARCH"]["W"][layer_id], self.params["ARCH"]["W"][layer_id + 1])],
            [f"Non-Linearity_{layer_id}", self.params["ARCH"]["nL"][0]]])            
        stack = np.concatenate(stack)
        stack = stack[:-1]# remove last non linearity !!
        self.stack = nn.Sequential(OrderedDict(stack)).to(self.params["device"])
        
class Factorized_Embedding():
    def __init__(self) -> None:
        pass

def train_test(data, model_type, input):
    # define data
    data.set_input_targets(input)
    data.shuffle()
    data.split_train_test(0.2)
    if model_type == "CPH":
        model = CPH(data)
    elif model_type == "CPHDNN":
        model = CPHDNN(data)
    else: model = None
    model._train()
    out = model._test()
    c_index = functions.compute_c_index(data.test.y["t"], data.test.y["e"], out)
    print (f"C index for model {model_type}, input: {input}: {c_index}")
    pdb.set_trace()

model_picker = {"CPH": CPH, "CPHDNN": CPHDNN}
def hpoptim(data, model_type, n = 100, nfolds = 5, nepochs = 1, input_size_range = None):
    
    # choose correct model, init
    model = model_picker[model_type](data, nepochs = nepochs)
    # split train / test (5)
    model.data.split_train_test(nfolds = 5) # 5 fold cross-val
    if model_type == "CPHDNN": model.data.folds_to_cuda_tensors()
    res = []
    best_params = None
    best_c_index = 0
    rep_params_list = functions.set_params_list(n, input_size_range)
    models = []
    # for each replicate (100)
    for rep_n, params in enumerate(rep_params_list):
        
        # fix (choose at random) set of params
        model.set_fixed_params(params)
        tr_c_index = []
        scores = []
        # cycle through folds
        for fold_n in tqdm(range (nfolds), desc = f"{model_type} - N{rep_n + 1} - Internal Cross Val"):
                
            # train
            tr_c = model._train_cv(fold_n)
            # test 
            out,l,c = model._valid_cv(fold_n)
            scores.append(out)
            # record accuracy
            tr_c_index.append(tr_c)
        c_ind_agg = functions.compute_aggregated_c_index(scores, model.data)
        c_ind_tr = np.mean(tr_c_index)
        if c_ind_agg > best_c_index:
            best_c_index = c_ind_agg
            best_params = model.params
        # compute aggregated c_index
        # print(model.params, round(score, 3))
        if model_type == "CPHDNN":
            res.append(np.concatenate([[model.params[key] for key in ["wd", "input_size", "D","W"]], [c_ind_tr, c_ind_agg]] ))
        elif model_type == "CPH":
           res.append(np.concatenate([[model.params[key] for key in ["wd", "input_size"]], [c_ind_tr, c_ind_agg]] )) 
        # for each fold (5)
            # train epochs (400)
            # test
        # record agg score, params
        
    if model_type == "CPHDNN":
        res = pd.DataFrame(res, columns = ["wd", "nIN", "D", "W", "c_index_train", "c_index_vld"] )
    elif model_type == "CPH":
        res = pd.DataFrame(res, columns = ["wd", "nIN", "c_index_train", "c_index_vld"] ) 
    res = res.sort_values(["c_index_vld"], ascending = False)
    # RERUN model with best HPs
    opt_model = model_picker[model_type](data)
    opt_model.set_fixed_params(best_params)
    opt_model._train()
    # return model
    return res, opt_model


def main():
    # some test funcs
    pass

if __name__ == "__main__":
    main()