# custom
from re import I
from engines.datasets.base_datasets import SurvivalGEDataset
import engines.datasets.FE_datasets as FE_Datasets
import engines.models.functions as functions 
import engines.models.dimredox_models as models
from engines.optimisers.base_optimisers import HPOptimiser
from engines.models import cox_models
from engines import utils
from engines.datasets.base_datasets import Data
# base
from torch.autograd import Variable
from datetime import datetime
import pandas as pd
import pdb 
import numpy as np
from tqdm import tqdm 
import os 
import torch 
import monitoring 
import time 

class Engine:
    def __init__(self, params):
        self._params = params

class Benchmark(Engine):
    def __init__(self, data, params):
        super().__init__(params)
        self.cohort = data["cohort"]
        self.data = data
        self.OUTDIR = utils.assert_mkdir(os.path.join("RES", f"EXP_{self._params.EXP}", self.cohort))
        
    def _perform_projection(self, proj_type, data, input_size = 17):    
        # set data
        if proj_type == "PCA":
            data = data["CDS"].clone()
            data.generate_PCA(input_size)
        elif proj_type == "SVD":
            data = data["CDS"].clone()
            data.generate_SVD(input_size)
        elif proj_type == "RPgauss":
            data = data["CDS"].clone()
            data.generate_RP("gauss", input_size)
        elif proj_type == "RPgauss_var":
            data = data["CDS"].clone()
            data.generate_RP("gauss", input_size, var_frac = 0.5)
        elif proj_type == "RPsparse":
            data = data["CDS"].clone()
            data.generate_RP("sparse", input_size)
        elif proj_type == "RSelect":
            data = data["CDS"].clone()
            data.generate_RS(input_size, var_frac = 0.5)
        elif proj_type == "CF-PCA":
            y = data['CDS'].y
            gi = data["CDS"].gene_info
            red_bl_cf = data["CF_bin"].iloc[:,[0,1,2,3,4,5,7,9,11]]
            pca = data["CDS"].clone()
            pca.generate_PCA()
            red_bl_cf_pca = red_bl_cf.merge(pca.x.iloc[:,:input_size], left_index = True, right_index = True)
            data = Data(red_bl_cf_pca, y, gi , name = f"red_bl_PCA_d_{input_size}" )
        else:
            data = data[proj_type].clone()
        return data 
    
    def _dump(self, line):
        with open(self.OUTFILE, "a") as o:
            o.writelines(line)
    
    def run(self, in_D):
        self.OUTFILE = os.path.join(self.OUTDIR, f"d_{in_D}_n_{self._params.NREP_TECHN}_{datetime.now()}.csv")
        # init results
        tst_res = []
        tr_res = [] 
        agg_c_index = []
        header = ",".join(["cohort", "rep_n", "proj_type", "input_d", "c_ind_tr", "c_ind_tst"]) + "\n"
        self._dump(header)
        for rep_n in tqdm(range(self._params.NREP_TECHN), desc = f"input D - {in_D}"):
            idx = np.arange(self.data["CDS"].x.shape[0])
            np.random.shuffle(idx) # shuffle dataset! 

            for proj_type in self._params.PROJ_TYPES:
                data = self._perform_projection(proj_type, self.data, in_D)
                data.reindex(idx) # shuffle 
                data.split_train_test(self._params.NFOLDS)
                # width    
                tst_scores = [] # store risk prediction scores for agg_c_index calc
                tr_c_ind_list = [] # store risk prediction scores for agg_c_index calc 
                # a data frame containing training optimzation results

                for foldn in range(self._params.NFOLDS):
                    test_data = data.folds[foldn].test
                    train_data = data.folds[foldn].train
                    # choose model type, hps and train
                    model = cox_models.CPH(data = train_data)
                    model.set_fixed_params({"input_size": in_D, "wd": 1e-10})
                    tr_metrics = model._train()
                    # test
                    tst_metrics = model._test(test_data)
                    tst_scores.append(tst_metrics["out"])
                    tr_c_ind_list.append(tr_metrics["c"])
                c_ind_tr = np.mean(tr_c_ind_list)
                c_ind_tst = functions.compute_c_index(data.y["t"], data.y["e"], np.concatenate(tst_scores))
                line = ",".join(np.array([self.cohort, rep_n, proj_type, in_D, c_ind_tr, c_ind_tst]).astype(str)) + "\n"
                self._dump(line)
        
        return self.OUTFILE

class RP_BG_Engine(Benchmark):
    """
    Class that computes accuracy with random projection of data 
    """
    def __init__(self, params):
        super().__init__(params)
        
    def _evaluate(self, data, label):
        tst_scores = [] # store risk prediction scores for agg_c_index calc
        tr_c_ind_list = [] # store risk prediction scores for agg_c_index calc 
        # a data frame containing training optimzation results
        for foldn in tqdm(range(self.NFOLDS), desc = label):
            test_data = data.folds[foldn].test
            train_data = data.folds[foldn].train
            # choose model type, hps and train
            model = cox_models.CPH(data = train_data)
            model.set_fixed_params({"input_size": self.INPUT_DIMS, "wd": 1e-10})
            tr_metrics = model._train()
            # test
            tst_metrics = model._test(test_data)
            tst_scores.append(tst_metrics["out"])
            tr_c_ind_list.append(tr_metrics["c"])
        return tst_scores, tr_c_ind_list

    def run(self):
        # select cohort data
        cohort_data = self.datasets[self.COHORT]
        header = ",".join(["rep_n", "proj_type", "k", "c_ind_tr", "c_ind_tst"]) + "\n"
        self._dump(header)
        for proj_type in self.PROJ_TYPES:
            for rep_n in range(self.REP_N):
                idx = np.arange(cohort_data.data["CDS"].x.shape[0])
                np.random.shuffle(idx) # shuffle dataset! 

                data = self._perform_projection(proj_type, cohort_data, self.INPUT_DIMS)
                data.reindex(idx) # shuffle 
                
                data.split_train_test(self.NFOLDS)
                # width    
                tst_scores, tr_c_ind_list = self._evaluate(data, label = f"{rep_n + 1}-{proj_type}")

                c_ind_tr = np.mean(tr_c_ind_list)
                c_ind_tst = functions.compute_c_index(data.y["t"], data.y["e"], np.concatenate(tst_scores))
                line = ",".join(np.array([rep_n, proj_type, self.INPUT_DIMS, c_ind_tr, c_ind_tst]).astype(str)) + "\n"
                self._dump(line)
    
        return self.OUTFILE

class FE_Engine:
    def __init__(self, params) -> None:
        self.params = params

    def run_fact_emb(self):
        cohort = "pronostic"
        opt = self.params
        seed = opt.seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)

        exp_dir = opt.load_folder
        if exp_dir is None: # we create a new folder if we don't load.
            exp_dir = monitoring.create_experiment_folder(opt)

        # creating the dataset
        print ("Getting the dataset...")
        dataset = FE_Datasets.get_dataset(opt,exp_dir)
        ds = Leucegene_Dataset(cohort).data["CDS"]

        # Creating a model
        print ("Getting the model...")

        my_model, optimizer, epoch, opt = monitoring.load_checkpoint(exp_dir, opt, dataset.dataset.input_size(), dataset.dataset.additional_info())

        # Training optimizer and stuff
        criterion = torch.nn.MSELoss()

        if not opt.cpu:
            print ("Putting the model on gpu...")
            my_model.cuda(opt.gpu_selection)

        # The training.
        print ("Start training.")
        #monitoring and predictions
        predictions =np.zeros((dataset.dataset.nb_patient,dataset.dataset.nb_gene))
        indices_patients = np.arange(dataset.dataset.nb_patient)
        indices_genes = np.arange(dataset.dataset.nb_gene)
        xdata = np.transpose([np.tile(indices_genes, len(indices_patients)),
                            np.repeat(indices_patients, len(indices_genes))])
        progress_bar_modulo = len(dataset)/100




        monitoring_dic = {}
        monitoring_dic['train_loss'] = []

        for t in range(epoch, opt.epoch):

            start_timer = time.time()

            thisepoch_trainloss = []

            with tqdm(dataset, unit="batch") as tepoch:
                for mini in tepoch:
                    tepoch.set_description(f"Epoch {t}")


                    inputs, targets = mini[0], mini[1]

                    inputs = Variable(inputs, requires_grad=False).float()
                    targets = Variable(targets, requires_grad=False).float()

                    if not opt.cpu:
                        inputs = inputs.cuda(opt.gpu_selection)
                        targets = targets.cuda(opt.gpu_selection)

                    # Forward pass: Compute predicted y by passing x to the model
                    y_pred = my_model(inputs).float()
                    y_pred = y_pred.squeeze()

                    targets = torch.reshape(targets,(targets.shape[0],))
                    # Compute and print loss

                    loss = criterion(y_pred, targets)
                    to_list = loss.cpu().data.numpy().reshape((1, ))[0]
                    thisepoch_trainloss.append(to_list)
                    tepoch.set_postfix(loss=loss.item())

                    np.save(os.path.join(exp_dir, 'pixel_epoch_{}'.format(t)),my_model.emb_1.weight.cpu().data.numpy() )
                    np.save(os.path.join(exp_dir,'digit_epoch_{}'.format(t)),my_model.emb_2.weight.cpu().data.numpy())

                    # Zero gradients, perform a backward pass, and update the weights.
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            monitoring.save_checkpoint(my_model, optimizer, t, opt, exp_dir)
            monitoring_dic['train_loss'].append(np.mean(thisepoch_trainloss))
            np.save(f'{exp_dir}/train_loss.npy',monitoring_dic['train_loss'])
            functions.plot_factorized_embedding(ds, my_model.emb_2.weight.cpu().data.numpy(), 
                loss.data.cpu().numpy().reshape(1,)[0], 
                self.params.emb_size, 
                t,
                method = "TSNE",
                cohort = "pronostic"
            )