from operator import xor
from re import I, L
import gzip
import numpy as np
import pandas as pd
from tqdm import tqdm
import pdb
import os 
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn import random_projection
import torch
from engines import utils
import xml.etree.ElementTree as ET
from sklearn.preprocessing import LabelBinarizer
class Data:
    def __init__(self,x, y ,gene_info, name = "data", reindex = True, device = "cpu", learning = True) -> None:
        self.name = name
        self.x = x
        self.y = y
        self.gene_info = gene_info
        self.device = device
        if reindex: self._reindex_targets()
        if learning and len(self.y.columns) > 2:
            self.y = self.y[["Overall_Survival_Time_days", "Overall_Survival_Status"]]
            self.y.columns = ["t", "e"]
    
    def clone(self):
        return Data(self.x, self.y, self.gene_info, name = self.name)
      
    def folds_to_cuda_tensors(self, device = "cuda:0"):
        if "cuda" in self.device : return 
        for i in range(len(self.folds)):
            train_x = torch.Tensor(self.folds[i].train.x.values).to(device)
            train_y = torch.Tensor(self.folds[i].train.y.values).to(device)
            test_x = torch.Tensor(self.folds[i].test.x.values).to(device)
            test_y = torch.Tensor(self.folds[i].test.y.values).to(device)
            train = Data(x = train_x, y = train_y, gene_info = None, reindex = False)
            test = Data(x = test_x, y = test_y, gene_info = None, reindex =False)
            self.folds[i].train = train
            self.folds[i].test = test

    def split_train_test(self, nfolds, device = "cpu", loo = False):
         # do nothing if dataset is already split! 
        self.x = self.x.sample(frac = 1)
        
        n = self.x.shape[0]
        fold_size = int(float(n)/nfolds)
        fold_size = fold_size if loo else fold_size + 1
        self.folds = []
        for i in range(nfolds):
            fold_ids = np.arange(i * fold_size, min((i + 1) * fold_size, n))
            test_x = self.x.iloc[fold_ids,:]
            test_y = self.y.loc[test_x.index]
            train_x = self.x.loc[~self.x.index.isin(test_x.index)]
            train_y = self.y.loc[train_x.index]
            self.folds.append(Data(self.x, self.y,  self.gene_info, name = self.name))
            self.folds[i].train = Data(train_x, train_y, self.gene_info, name = self.name)
            self.folds[i].train.to(device)
            self.folds[i].test = Data(test_x,test_y, self.gene_info, name = self.name)
            self.folds[i].test.to(device)
        # reorder original data
        self._reindex_targets()

    def to(self, device):    
        self.device = device
        if device == "cpu": return
        self.x = torch.Tensor(self.x.values).to(device)
        self.y = torch.Tensor(self.y.values).to(device)
        
    def to_DF(self):
        if "cuda" in self.device :
            pdb.set_trace()
            self.x = pd.DataFrame(self.x.detach().cpu().numpy())
            self.y = pd.DataFrame(self.y.detach().cpu().numpy(), columns = ["t", "e"])
    
    def generate_PCA(self, input_size):
        self.name = f"PCA-{input_size}"
        # init object
        self._pca = PCA()
        # fit to data
        self._xpca = pd.DataFrame(self._pca.fit_transform(self.x), index = self.x.index)
        # get loadings
        # 
        # transform in self.NS dimensions
        # 
        # Writes to file 
        # 
        self.x = self._xpca.iloc[:,:input_size]
        return {"proj_x":self._xpca, "pca":self._pca }
        
    def generate_RP(self, method, n = 17, var_frac = 0):
        self.name = f"random projection var frac = {var_frac}"
        #print("Running Random Projection...")
        high_v_cols = self.x.columns[self.x.var() >= self.x.var().sort_values()[int(self.x.shape[1] * float(var_frac))]]
        self._x_var_frac = self.x[high_v_cols] 
        if method == "gauss":
            self.transformer = random_projection.GaussianRandomProjection(n_components=n)
            self._xrp = pd.DataFrame(self.transformer.fit_transform(self._x_var_frac), index = self.x.index)
        elif method == "sparse":
            self.transformer = random_projection.SparseRandomProjection(n_components=n)
            self._xrp = pd.DataFrame(self.transformer.fit_transform(self._x_var_frac), index = self.x.index)
        self.x = self._xrp
    
    def generate_RS(self, n, var_frac = 0.5):
        self.name = "random selection"
        #print(f"Generating Random signature (genes with var_frac: {var_frac})...")
        high_v_cols = self.x.columns[self.x.var() >= self.x.var().sort_values()[int(self.x.shape[1] * float(var_frac))]]
        col_ids = np.arange(len(high_v_cols))
        np.random.shuffle(col_ids)
        # assert enough variance
        self.x = self.x[high_v_cols[:n]]

    def generate_SVD(self, n):
        #print("Running Singular Value Decomposition SVD ...")
        svd = TruncatedSVD(n_components = n)
        self.x = pd.DataFrame(svd.fit_transform(self.x), index = self.x.index)

    def shuffle(self):
        self.x = self.x.sample(frac = 1)
        self._reindex_targets()

    def remove_unexpressed_genes(self, verbose = 0):
        """ removes all genes with no expression across all samples"""
        d = self.x.shape[1]
        n_rm = self.x.sum(0) != 0
        self.x = self.x.loc[:,n_rm]
        if verbose:
            print(f"removed {d - n_rm.sum()} genes with null expression across samples ")
            print(f"Now datataset hase shape {self.x.shape}")


    def create_shuffles(self, n):
        print(f"Creates {n} data shuffles ...")
        self.shuffles = [self.x.sample(frac =1).index for i in range(n)]
    
    def select_shuffle(self, n):
        self.x = self.x.loc[self.shuffles[n]]
        self._reindex_targets()
    def reindex(self, idx):
        self.x = self.x.iloc[idx]
        self.y = self.y.loc[self.x.index]

    def _reindex_targets(self):
        self.y = self.y.loc[self.x.index]
    
    def _reindex_expressions(self):
        self.x = self.x.loc[self.y.index]

class SurvivalGEDataset():
    def __init__(self) -> None:
        self.learning = True
        self.gene_repertoire = self.process_gene_repertoire_data()
    
    def new(self, clinical_factors, gene_expressions):
        if clinical_factors is None:
            if "LSC17+PCA" in gene_expressions:
                LSC17_features = self.data["LSC17"]
                train_features = LSC17_features.x.merge(self.data["CDS"].x, left_index = True, right_index = True)

            elif "LSC17" in gene_expressions :
                LSC17_features = self.data["LSC17"]
                train_features = LSC17_features.x
            elif "PCA" in gene_expressions:
                ge_features = self.data["CDS"]
                train_features = ge_features.x
            else: return None
            data_name = self.COHORT + "_" + gene_expressions
        else:
            # manage clinical factors
            clinical_features = self.data["CF_bin"][clinical_factors]
            # manage gene expressions
            if gene_expressions and  "LSC17+PCA" in gene_expressions:
                LSC17_features = self.data["LSC17"].x.merge(self.data["CDS"].x, left_index = True, right_index = True)
                train_features = clinical_features.merge(LSC17_features, left_index = True, right_index = True)

            elif gene_expressions and "LSC17" in gene_expressions :
                LSC17_features = self.data["LSC17"]
                train_features = clinical_features.merge(LSC17_features.x, left_index = True, right_index = True)
            elif gene_expressions and  "PCA" in gene_expressions:
                ge_features = self.data["CDS"]
                train_features = clinical_features.merge(ge_features.x, left_index = True, right_index = True) 
            else : train_features = clinical_features 
            # manage target features
            
            data_name = self.COHORT + "_clin_factors_"+ gene_expressions if gene_expressions is not None else "clin_factors" 
        target_features = self.data["CDS"].y   
        data = Data(x = train_features, y = target_features, gene_info=self.gene_repertoire, name= data_name)
        if (data.y.index != data.x.index).sum() > 0: raise(Exception, 'error: unmatch in index')
        return data
        
    def process_gene_repertoire_data(self):
        print("Loading and assembling Gene Repertoire...")
        # load in Gencode 37 repertoire (NOTE: no biotype present !!) 
        Gencode37 = pd.read_csv("/u/leucegene/data/Homo_sapiens.GRCh38_H32/annotations/Homo_sapiens.GRCh38.Gencode37.genes.tsv", sep = "\t")
        # load in Esembl99 repertoire (NOTE: ensembl id incomplete (no version id))
        Ensembl99 = pd.read_csv("/u/leucegene/data/Homo_sapiens.GRCh38_H32/annotations/Homo_sapiens.GRCh38.Ensembl99.genes.tsv", sep = "\t")
        # extract gene infos and store
        gene_info = Gencode37.merge(Ensembl99, on = "SYMBOL") 
        return gene_info
    
    def get_data(self, cohort):
        if cohort == "lgn_pronostic": 
            DS = Leucegene_Dataset(self.gene_repertoire)
            DS.load()
            self._set_data(DS, rm_unexpr=True)
        # load 
        elif cohort == "tcga_target_aml":
            DS = TCGA_Dataset(self.gene_repertoire)
            DS.load()
            self._set_data(DS, rm_unexpr=True)

        elif cohort == "lgn_pronostic_intermediate":
            DS = Leucegene_Dataset(self.gene_repertoire)
            DS.load(cytogenetic_groups= ["intermediate cytogenetics"])
            DS.COHORT = cohort
            self._set_data(DS, rm_unexpr=True)
        return self.data
    
    def _binarize_clin_infos(self):
        ret_df = pd.DataFrame((self.CF["Age_at_diagnosis"] > 60).astype(int))
        ret_df.columns = ["Age_gt_60"]
        binarized_features = ['WHO classification','FLT3-ITD mutation', 'IDH1-R132 mutation','NPM1 mutation', 'Sex', 'Cytogenetic risk']
        # self.CF["Cytogenetic group"] = ["cytg_" + s for s in self.CF["Cytogenetic group"]] 
        for feature in binarized_features :
            lb = LabelBinarizer()
            bin = lb.fit_transform(self.CF[feature])
            if bin.shape[1] == 1:
                bin = np.hstack((1 - bin, bin))
                bin_labels = pd.DataFrame(bin, columns = [f"{feature}_{c}" for c in lb.classes_], index = self.CF.index)
            else: bin_labels = pd.DataFrame(bin, columns = lb.classes_, index = self.CF.index)
            ret_df = ret_df.merge(bin_labels,  left_index = True, right_index = True)

        self.CF_bin = ret_df.drop(["FLT3-ITD mutation_0", "NPM1 mutation_0.0", "IDH1-R132 mutation_0.0", "Sex_M"], axis = 1)
        self.CF_bin.columns = [c.replace("_1.0", "").replace("_1", "") for c in self.CF_bin.columns]

    def _set_data(self, DS, rm_unexpr = False):
         
        self._GE_TPM = DS._GE_TPM
        self.CF = DS._CLIN_INFO
        if DS.COHORT in ["tcga_target_aml", "lgn_pronostic_intermediate"]:
            self.CF_bin = None
        else: 
            self._binarize_clin_infos() 
        self.COHORT = DS.COHORT
        # select cds
        ### select based on repertoire
        # filtering if needed, merge with GE data  
        self._GE_CDS_TPM = self._GE_TPM.merge(self.gene_repertoire[self.gene_repertoire["gene_biotype_y"] == "protein_coding"], left_index = True, right_on = "featureID_y")
        # clean up
        self._GE_CDS_TPM.index = self._GE_CDS_TPM.SYMBOL
        self._GE_CDS_TPM = (self._GE_CDS_TPM.iloc[:,:-self.gene_repertoire.shape[1]]).T
        self.GE_CDS_LOG = np.log(self._GE_CDS_TPM + 1)
        self.GE_TRSC_LOG = np.log(self._GE_TPM.T + 1)
        # set CDS data
        cds_data = Data(self.GE_CDS_LOG, self.CF, self.gene_repertoire, name = f"{self.COHORT}_CDS", learning = self.learning)
        if rm_unexpr :  cds_data.remove_unexpressed_genes(verbose=1)
        # set TRSC data
        trsc_data = Data(self.GE_TRSC_LOG, self.CF, self.gene_repertoire, name = f"{self.COHORT}_TRSC", learning = self.learning) 
        # set LSC17 data
        lsc17_data = Data(self._get_LSC17(), self.CF ,self.gene_repertoire, name = f"{self.COHORT}_LSC17", learning = self.learning )
        # set the data dict
        self.data = {"cohort": self.COHORT,"CDS": cds_data, "TRSC": trsc_data, "LSC17": lsc17_data, "CF": self.CF, "CF_bin": self.CF_bin} #, "LSC17":lsc17_data, "FE": FE_data}

    
    def _get_LSC17(self):
        if 0: # f"LSC17_{self.COHORT}_expressions.csv" in os.listdir("Data/SIGNATURES"):
            LSC17_expressions = pd.read_csv(f"Data/SIGNATURES/LSC17_{self.COHORT}_expressions.csv", index_col = 0)
        else: 
            lsc17 = pd.read_csv("Data/SIGNATURES/LSC17.csv")
            LSC17_expressions = self.GE_TRSC_LOG[lsc17.merge(self.gene_repertoire, left_on = "ensmbl_id_version", right_on = "featureID_x").featureID_y]
            LSC17_expressions.to_csv(f"Data/SIGNATURES/LSC17_{self.COHORT}_expressions.csv")
        return LSC17_expressions
    
    def _get_embedding(self): # bugged for now...
        print("Fetching embedding file...")
        if self.EMB_FILE.split(".")[-1] == "npy":
            emb_x = np.load(self.EMB_FILE)
        elif self.EMB_FILE.split(".")[-1] == "csv":
            emb_x = pd.read_csv(self.EMB_FILE, index_col=0)
        x = pd.DataFrame(emb_x, index = self.GE_CDS_LOG.index)
        return x 

class TCGA_Dataset():
    def __init__(self, gene_repertoire):
        # hardocre the name of the cohort
        self.COHORT = "tcga_target_aml"
        # setup paths
        self.data_path = "Data"
        self.tcga_data_path = os.path.join(self.data_path, "TCGATARGETAML")
        self.tcga_manifests_path = os.path.join(self.tcga_data_path, "MANIFESTS")
        self.tcga_counts_path = os.path.join(self.tcga_data_path, "COUNTS")
        self.tcga_cd_path  = os.path.join(self.tcga_data_path, "CLINICAL")
        # store the gene repertoire
        self.gene_repertoire = gene_repertoire
        # init Clinical infos files
        self._init_CF_files()
        self.NS = self._CLIN_INFO.shape[0]   

    def _init_CF_files(self):
        if "TCGA_CF.assembled.csv" in os.listdir(self.tcga_data_path):
            self._CLIN_INFO_RAW = pd.read_csv(os.path.join(self.tcga_data_path, "TCGA_CF.assembled.csv"), index_col = 0)
        else: 
            ## FETCH CLINICAL DATA ##
            CD_manifest_file = os.path.join(self.tcga_manifests_path, 'gdc_manifest.2020-07-23_CD.txt')
            if not self._assert_load_from_manifest(CD_manifest_file, self.tcga_cd_path):
                self._load_tcga_aml(CD_manifest_file, target_dir = self.tcga_cd_path)
            else : print('OUT TCGA + TARGET - AML Clinical Raw data already loaded locally on disk')   
            
            ## ASSEMBLE CLINICAL DATA ## 
            CD_tcga_profile = self._parse_clinical_xml_files()
            print ('OUT Assembled TCGA clinical data')
            # select target-aml most up to date clinical file 
            CD_target_profile = pd.read_excel(os.path.join(self.tcga_cd_path, "TARGET_AML_ClinicalData_Validation_20181213.xlsx"), engine='openpyxl') 
            print ('OUT Assembled TARGET clinical data')
            
            ## GET TARGET to CASE ID FILE ##
            filepath = os.path.join(self.tcga_data_path, 'filename_to_caseID.csv')
            fileuuid_caseuuid = pd.read_csv(filepath) 
            
            ## MERGE TARGET + TCGA CLINICAL FEATURES ##
            tcga_target_clinical_features = self._merge_tcga_target_clinical_features(CD_tcga_profile, CD_target_profile)
            info_data = tcga_target_clinical_features.merge(fileuuid_caseuuid)
            # process filenames for easier fetching
            info_data['filepath'] = [os.path.join(self.tcga_counts_path,  filename_x) for filename_x in info_data.filename_x]
            # add a dataset tag
            info_data['dataset'] = 'TCGA'
            info_data['sequencer'] = 'Hi-seq'
            info_data['Overall Survival Time in Days'] = info_data['Overall Survival Time in Days'].astype(float)
            # select proper columns
            self._CLIN_INFO_RAW = info_data[['TARGET USI', 'submitter_id', 'filepath','dataset', 'sequencer', 'gender', 'age_at_diagnosis_in_years' , 'cytogenetic_risk', 'FLT3-ITD', 'NPM1', 'Induction_Type', 'Overall Survival Time in Days', 'vital_status']]
            print("OUT Merged TCGA and TARGET clinical features")
            # remove duplicates 
            counts = self._CLIN_INFO_RAW.groupby("TARGET USI").count()
            duplicates = counts.index[counts["filepath"] > 1]
            self._CLIN_INFO_RAW = self._CLIN_INFO_RAW[~self._CLIN_INFO_RAW["TARGET USI"].isin(duplicates)]
            self._CLIN_INFO_RAW.to_csv(os.path.join(self.tcga_data_path, "TCGA_CF.assembled.csv"))
        
        # preprocess clinical info file
        self._CLIN_INFO_RAW.columns = ['sampleID', 'submitter_id', 'filepath','dataset', 'sequencer', 'gender', 'age_at_diagnosis_years' , 'Cytogenetic risk', 'flt3_itd', 'npm1', 'induction_type', 'Overall_Survival_Time_days','Overall_Survival_Status']
        # format censorship state 
        self._CLIN_INFO_RAW['Overall_Survival_Status'] = (self._CLIN_INFO_RAW['Overall_Survival_Status'] == "Dead").astype(int)
        # remove samples marked as "dead" without recorded time (8 samples)
        self._CLIN_INFO = self._CLIN_INFO_RAW[(self._CLIN_INFO_RAW['Overall_Survival_Time_days'] == self._CLIN_INFO_RAW['Overall_Survival_Time_days'])]


    def load(self, cytogenetic_groups = ["Low", "Standard", "Favorable"]):
        # unstable
        # retriteves tpm transformed expression data 
        self._compute_tpm()
        # make sure all the ids in CF file and GE files are the same!
        common_ids = np.intersect1d(self._GE_TPM.columns, self._CLIN_INFO["sampleID"])
        self._CLIN_INFO.loc[:,"Cytogenetic risk"] = self._CLIN_INFO["Cytogenetic risk"].replace("Intermediate/Normal","Standard").replace("Poor", "Low")
        # perform cyto group filtration 
        filtered = np.intersect1d(common_ids, self._CLIN_INFO["sampleID"][self._CLIN_INFO["Cytogenetic risk"].isin(cytogenetic_groups)])
        self._GE_TPM = self._GE_TPM[filtered]
        self._CLIN_INFO = self._CLIN_INFO[self._CLIN_INFO["sampleID"].isin(filtered)]
        self._CLIN_INFO.index = self._CLIN_INFO["sampleID"]
        self._CLIN_INFO = self._CLIN_INFO.loc[self._GE_TPM.columns]
    
    def _compute_tpm(self):
        # computes tpm from raw matrix, if already computed, just load
        outfile = f"{self.COHORT}_GE_TRSC_TPM.csv"
        if outfile in os.listdir("Data") :
            self._GE_TPM = pd.read_csv(f"Data/{outfile}", index_col = 0)
        else:
            self._compute_ge_raw()
            print(f"TPM normalized Gene Expression (CDS only) file not found in Data/{outfile}\nNow performing tpm norm ...")
            print("Processing TPM computation...")
            # get gene lengths
            M = self._RAW_COUNTS.iloc[:,5:].T # remove header columns, then transpose
            GL = np.matrix(self._RAW_COUNTS["gene.length_x"]) / 1000 # get gene lengths in KB
            # tpm norm
            GE_RPK = M.values / GL
            per_million = GE_RPK.sum(1) / 1e6
            self._GE_TPM =  pd.DataFrame(GE_RPK / per_million)
            # clean up 
            self._GE_TPM.columns = self._RAW_COUNTS.ensmbl_id 
            self._GE_TPM.index = self._RAW_COUNTS.columns[5:]
            self._GE_TPM = self._GE_TPM.T 
            # write to file 
            print(f"Writing to Data/{outfile}...")
            self._GE_TPM.to_csv(f"Data/{outfile}")

    def _compute_ge_raw(self):
        # computes the count matrix if it doesn't already exist, else load
        if "TCGA_GE.assembled.csv" in os.listdir(self.tcga_data_path):
            print("TCGA loading data matrices ...")
            self._RAW_COUNTS = pd.read_csv(os.path.join(self.tcga_data_path, "TCGA_GE.assembled.csv"))
        else: 
            self._assemble_load_tcga_data()

    def _assemble_load_tcga_data(self):
        ## FETCH GENE EXPRESSIONS ##
        GE_manifest_file = os.path.join(self.tcga_manifests_path, 'gdc_manifest.2020-07-23_GE.txt')
        if not self._assert_load_from_manifest(GE_manifest_file, self.tcga_counts_path):
            self._load_tcga_aml(GE_manifest_file, target_dir = self.tcga_counts_path)
        else : print('OUT TCGA + TARGET -AML Gene Expression Raw data already loaded locally on disk')
        ## ASSEMBLE GE PROFILES FROM REPERTOIRE ##
        # unzip and merge files 
        
        # relabel repertoire column , store into matrix
        self.gene_repertoire['ensmbl_id'] = [g.split(".")[0] for g in self.gene_repertoire.featureID_x]
        count_matrix = self.gene_repertoire[['ensmbl_id', 'SYMBOL', 'gene_biotype_y', "gene.length_x"]]

        for i,r in self._CLIN_INFO.iterrows():
            if i % 10 == 0 : print('OUT Assembled TCGA + TARGET (C) {} / {} HT-Seq data'.format(i, self._CLIN_INFO.shape[0] ))
            filename =  r['filepath']
            GE_profile = pd.read_csv(filename, sep = '\t', names = ['ensmbl_id', r['TARGET USI']])
            GE_profile['ensmbl_id'] = [e.split('.')[0] for e in GE_profile.ensmbl_id]
            count_matrix = count_matrix.merge(GE_profile, on = 'ensmbl_id') 
        print('OUT finished assembling TCGA + TARGET raw COUNT matrix of {} with {} samples'.format(count_matrix.shape[0], i + 1)) 
        self._RAW_COUNTS = count_matrix
        # writing to files 
        print("writing to files ... ")
        self._RAW_COUNTS.to_csv(os.path.join(self.tcga_data_path, "TCGA_GE.assembled.csv"))

    def _assert_load_from_manifest(self, manifest_file, tcga_path):
        """ 
        returns false if number of files in manifest and tcga_path unequal
        returns false if manifest doesnt exist
        """
        utils.assert_mkdir(tcga_path)
        manifest_exists = os.path.exists(manifest_file)
        tcga_manifest = len(os.listdir(tcga_path)) == len(open(manifest_file).readlines(  )) -1    
        return manifest_exists and tcga_manifest
        
    def _parse_clinical_xml_files(self):
        # HARDCODED features to extract
        patient_features = ['batch_number', 'project_code', 'tumor_tissue_site', 'leukemia_specimen_cell_source_type', 'gender', 'vital_status','bcr_patient_barcode', 'days_to_death', 'days_to_last_known_alive', 'days_to_last_followup', 'days_to_initial_pathologic_diagnosis','days_to_birth', 'age_at_initial_pathologic_diagnosis', 'year_of_initial_pathologic_diagnosis', "acute_myeloid_leukemia_calgb_cytogenetics_risk_category" ]
        mutation_features = ['NPMc Positive', 'FLT3 Mutation Positive', 'Activating RAS Positive']
        header_array = np.concatenate((patient_features, mutation_features))
        clinical_data_matrix = []
        # assemble all .xml tcga-laml files
        for f in os.listdir(self.tcga_cd_path):
            if '.xml' in f:
                tree = ET.parse(os.path.join(self.tcga_cd_path, f))
                root = tree.getroot()
                # assemble a dict of all patient, clin features 
                xml_patient_dict = dict([(e.tag.split('}')[-1], e.text) for i in root for e in i])
                patient_features_array = [xml_patient_dict[f] for f in patient_features]
                mutation_features_array = []
                for e in root:
                    for i in e :
                        if i.tag.split('}')[-1] == "molecular_analysis_abnormality_testing_results":
                            mutation_profile = [(elem[0].text) for elem in i]
                            for mut in mutation_features:
                                mutation_features_array.append(int(mut in mutation_profile))	    
            clinical_data_matrix.append(np.concatenate((patient_features_array, mutation_features_array)))
        clinical_data = pd.DataFrame(clinical_data_matrix, columns = header_array)

        return clinical_data

    def _merge_tcga_target_clinical_features(self, CD_tcga_profile, CD_target_profile):
        target_features = ['TARGET USI', 'Gender', 'Age at Diagnosis in Days' ,'FLT3/ITD positive?', 'NPM mutation', 'Overall Survival Time in Days', 'Vital Status', 'Risk group']
        target = CD_target_profile[target_features]
        target['Age at Diagnosis in Days'] = np.array(target['Age at Diagnosis in Days'] / 365, dtype = int)
        tcga_features = ['bcr_patient_barcode','gender', 'age_at_initial_pathologic_diagnosis' ,'FLT3 Mutation Positive', 'NPMc Positive', 'Overall Survival Time in Days' , 'vital_status', 'acute_myeloid_leukemia_calgb_cytogenetics_risk_category']
        CD_tcga_profile["Overall Survival Time in Days"] = CD_tcga_profile["days_to_death"] 
        censored = CD_tcga_profile["days_to_death"] != CD_tcga_profile["days_to_death"]
        CD_tcga_profile["Overall Survival Time in Days"][censored] = CD_tcga_profile["days_to_last_followup"][censored]
        CD_tcga_profile = CD_tcga_profile[tcga_features]
        CD_tcga_profile.columns = target_features
        # uniformize values 
        target.Gender = target.Gender.str.upper()
        target['FLT3/ITD positive?'] = np.asarray(target['FLT3/ITD positive?'] == 'Yes', dtype = int)
        target['NPM mutation'] = np.asarray(target['NPM mutation'] == 'Yes', dtype = int)
        tcga_target_clinical_features = pd.DataFrame(np.concatenate((CD_tcga_profile, target)), columns = ["TARGET USI", "gender", "age_at_diagnosis_in_years", "FLT3-ITD","NPM1", 'Overall Survival Time in Days' , 'vital_status', "Cytogenetic risk"])
        # add an unknown induction type column
        tcga_target_clinical_features['Induction_Type'] = 'unknown'
        # rename columns 
        return tcga_target_clinical_features 
    
    def _load_tcga_aml(self, manifest_file, target_dir = "OUT"): #### TO BE UPDATED!
        # target directory
        utils.assert_mkdir(target_dir)
        # import manifest
        manifest = pd.read_csv(manifest_file, sep = '\t')
        # cycle through filenames
        for i, row in manifest.iterrows():
            if not row['filename'] in os.listdir(target_dir) : 
                # store file_id
                file_id = row['id']
                # store data endpoint
                data_endpt = "https://api.gdc.cancer.gov/data/{}".format(file_id)
                # get response
                response = requests.get(data_endpt, headers = {"Content-Type": "application/json"})
                # The file name can be found in the header within the Content-Disposition key.
                response_head_cd = response.headers["Content-Disposition"]
                # store filename
                file_name = re.findall("filename=(.+)", response_head_cd)[0]
                output_file_name = os.path.join(target_dir, file_name)
                with open(output_file_name, "wb") as o:
                    o.write(response.content)
                    print('{} written to {}'.format( file_name, output_file_name))
            else : print ('{} Already in data base'.format(row['filename']))

class Leucegene_Dataset():
    def __init__(self, gene_repertoire, learning = True):
        self._init_CF_files()
        self.gene_repertoire = gene_repertoire
        self.COHORT = "lgn_pronostic"
        self.learning = learning # for machine learning data processing
        print(f"Loading ClinF {self.COHORT} file ...")
        self.CF_file = f"Data/{self.COHORT}_CF"
        self._CLIN_INFO = pd.read_csv(self.CF_file, index_col = 0)  # load in and preprocess Clinical Features file
        self.NS = self._CLIN_INFO.shape[0]

    def _init_CF_files(self):
            
        infos = pd.read_csv("Data/lgn_ALL_CF", sep = "\t").T
        infos.columns = infos.iloc[0,:] # rename cols
        infos = infos.iloc[1:,:] # remove 1st row
        features = ["Prognostic subset", "Age_at_diagnosis", 
        "Sex", "Induction_Type",
        'HSCT_Status_Type' ,'Cytogenetic risk', 
        'FAB classification','Tissue', 
        'RNASEQ_protocol',"IDH1-R132 mutation" ,
        "Relapse",'FLT3-ITD mutation', 
        "WHO classification", "NPM1 mutation", 
        "Overall_Survival_Time_days", "Overall_Survival_Status",
        "Cytogenetic group"] # select features
        infos = infos[features]
        for cohort in ["lgn_public", "lgn_pronostic"]:
            samples = pd.read_csv(f"Data/{cohort}_samples", index_col = 0)
            CF_file = infos.merge(samples, left_index = True, right_on = "sampleID")
            CF_file.index = CF_file.sampleID
            CF_file = CF_file[np.setdiff1d(CF_file.columns, ["sampleID"])]
            CF_file.to_csv(f"Data/{cohort}_CF")

    def load(self,  cytogenetic_groups = ["adverse cytogenetics", "intermediate cytogenetics", "favorable cytogenetics"]):
        self._compute_tpm()
        # make sure all the ids in CF file and GE files are the same!
        common_ids = np.intersect1d(self._GE_TPM.columns, self._CLIN_INFO.index)
        # perform cyto group filtration 
        common_ids = np.intersect1d(common_ids, self._CLIN_INFO.index[self._CLIN_INFO["Cytogenetic risk"].isin(cytogenetic_groups)] )

        self._GE_TPM = self._GE_TPM[common_ids]
        self._CLIN_INFO = self._CLIN_INFO[self._CLIN_INFO.index.isin(common_ids)]
        self._CLIN_INFO = self._CLIN_INFO.loc[self._GE_TPM.columns]
        
    def _compute_tpm(self):
        outfile = f"{self.COHORT}_GE_TRSC_TPM.csv"
        if outfile in os.listdir("Data") :
            self._GE_TPM = pd.read_csv(f"Data/{outfile}", index_col = 0)
        else:
            self._compute_ge_raw()
            print(f"TPM normalized Gene Expression (CDS only) file not found in Data/{outfile}\nNow performing tpm norm ...")
            self._GE_raw_T = self._RAW_COUNTS.T 
            self._GE_raw_T["featureID_x"] = self._GE_raw_T.index
            self._GE_raw_T["featureID_y"] = self._GE_raw_T["featureID_x"].str.split(".", expand = True)[0].values
            
            print("Processing TPM computation...")
            # get gene infos
            gene_info = self.gene_repertoire
            self._GE = self._GE_raw_T.merge(gene_info, on = "featureID_y") 
            gene_lengths = np.matrix(self._GE["gene.length_x"]).T / 1000 # get gene lengths in KB
            # tpm norm
            GE_RPK = self._GE.iloc[:,:self.NS].astype(float) / gene_lengths 
            per_million = GE_RPK.sum(0) / 1e6
            self._GE_TPM =  GE_RPK / per_million 
            # clean up 
            self._GE_TPM.index = self._GE.featureID_y
            # write to file 
            print(f"Writing to Data/{outfile}...")
            self._GE_TPM.to_csv(f"Data/{outfile}")
    
    def _compute_ge_raw(self):
        
        outfile = f"{self.COHORT}_GE.assembled.csv"
        if outfile in os.listdir("Data") :
            print(f"Loading Raw Gene Expression file from {outfile}...")
            self._RAW_COUNTS = pd.read_csv(f"Data/{outfile}", index_col = 0)
        else : 
            print(f"Gene Expression file not found... in Data/{outfile} \nLoading {self.NS} samples GE readcounts from files ...")
            samples = []
            for sample in tqdm(self.CF.index): 
                samples.append( pd.read_csv(f"/u/leucegene/data/sample/{sample}/transcriptome/readcount/star_GRCh38/star_genes_readcount.unstranded.xls", sep = "\t", index_col=0).T) 
            print("Concatenating ...")
            df = pd.concat(samples)
            print(f"writing to Data/{outfile} ...")
            df.to_csv(f"Data/{outfile}")
            self._RAW_COUNTS = df 
   
    
    
