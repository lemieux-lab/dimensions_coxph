from engines.utils import assert_mkdir
from experiments.plotting_functions import *
from engines.models.functions import *
from engines.functions import * 
from engines.datasets.base_datasets import SurvivalGEDataset
from engines.models import cox_models
from engines.hp_dict.base import HP_dict
from collections import Counter
import os
import pdb 

def run(args):

    assert_mkdir(args.OUTPATH)
    
    SGE1 = SurvivalGEDataset()
    lgn_pronostic = SGE1.get_data("lgn_pronostic")
    LGN_PCA17 = SGE1.new(None, "PCA")
    LGN_LSC17 = SGE1.new(None, "LSC17")
    

    SGE2 = SurvivalGEDataset()
    SGE2.get_data("lgn_pronostic_intermediate")
    LGN_INT_PCA17 = SGE2.new(None, "PCA")
    LGN_INT_LSC17 = SGE2.new(None, "LSC17")


    SGE3 = SurvivalGEDataset()
    tcga = SGE3.get_data("tcga_target_aml")
    TCGA_PCA17 = SGE3.new(None, "PCA")
    TCGA_LSC17 = SGE3.new(None, "LSC17")
    lgn_cyt_levels = [{"intermediate cytogenetics":1, "adverse cytogenetics": 2, "favorable cytogenetics":0 }[level] for level in lgn_pronostic["CF"]["Cytogenetic risk"]]
    tcga_cyt_levels = [{"Standard":1, "Low": 2, "Favorable":0 }[level] for level in tcga["CF"]["Cytogenetic risk"]]
    
    SGE = SurvivalGEDataset()
    SGE.get_data("lgn_pronostic")
    mutations = ["NPM1 mutation", "FLT3-ITD mutation", "IDH1-R132 mutation"]
    age_sex = ["Sex_F", "Age_gt_60"] # Age is bugged
    who = ['Therapy-related myeloid neoplasms',
       'AML with minimal differentiation', 'AML without maturation',
       'AML with myelodysplasia-related changes',
       'Acute monoblastic and monocytic leukaemia',
       'AML with t(8;21)(q22;q22); RUNX1-RUNX1T1',
       'AML with inv(16)(p13.1q22) or t(16;16)(p13.1;q22); CBFB-MYH11',
       'Acute erythroid leukaemia',
       'AML with inv(3)(q21q26.2) or t(3;3)(q21;q26.2); RPN1-EVI1',
       'AML with maturation', 'Acute myeloid leukaemia, NOS',
       'AML with t(9;11)(p22;q23); MLLT3-MLL',
       'Acute myelomonocytic leukaemia',
       'AML with t(6;9)(p23;q34); DEK-NUP214',
       'Acute megakaryoblastic leukaemia']

    clinical_features = np.concatenate([mutations, who, age_sex])
    LGN_CF = SGE1.new(clinical_features, gene_expressions=None)
    
    LGN_CF_LSC17 = SGE.new(clinical_features, gene_expressions="LSC17")
    LGN_CF_PCA17 = SGE.new(clinical_features, gene_expressions="PCA")

    # filter on variance
    var = LGN_CF.x.var(0)
    LGN_CF.x = LGN_CF.x[LGN_CF.x.columns[np.where( var > 0.01)]]
    var = LGN_CF_LSC17.x.var(0)
    LGN_CF_LSC17.x = LGN_CF_LSC17.x[LGN_CF_LSC17.x.columns[np.where( var > 0.01)]]
    var = LGN_CF_PCA17.x.var(0)
    LGN_CF_PCA17.x = LGN_CF_PCA17.x[LGN_CF_PCA17.x.columns[np.where( var > 0.01)]]

    LGN_CF.split_train_test(args.NFOLDS)
    LGN_LSC17.split_train_test(args.NFOLDS)
    LGN_PCA17.split_train_test(args.NFOLDS)
    LGN_CF_LSC17.split_train_test(args.NFOLDS)
    LGN_CF_PCA17.split_train_test(args.NFOLDS)
    
    LGN_INT_LSC17.split_train_test(args.NFOLDS)
    LGN_INT_PCA17.split_train_test(args.NFOLDS)

    TCGA_LSC17.split_train_test(args.NFOLDS)
    TCGA_PCA17.split_train_test(args.NFOLDS)



    HyperParams = HP_dict(args.WEIGHT_DECAY, args.NEPOCHS, args.bootstr_n, args.NFOLDS)
    LGN_LSC17_params = HyperParams.generate_default("ridge_cph_lifelines_LSC17", LGN_LSC17)  
    LGN_CF_LSC17_params = HyperParams.generate_default("ridge_cph_lifelines_CF_LSC17", LGN_CF_LSC17)      
    LGN_PCA17_params = HyperParams.generate_default("ridge_cph_lifelines_PCA", LGN_PCA17) 
    LGN_PCA_params = {"min_col": 0, "max_col": LGN_PCA17.x.shape[1], "pca_n": 17}       
    LGN_CF_PCA17_params = HyperParams.generate_default("ridge_cph_lifelines_CF_PCA", LGN_CF_PCA17) 

    LGN_INT_LSC17_params = HyperParams.generate_default("ridge_cph_lifelines_LSC17", LGN_INT_LSC17)        
    LGN_INT_PCA17_params = HyperParams.generate_default("ridge_cph_lifelines_PCA", LGN_INT_PCA17)        
    LGN_INT_PCA_PARAMS = {"min_col": 0, "max_col": LGN_INT_PCA17.x.shape[1], "pca_n": 17}   
    TCGA_LSC17_params = HyperParams.generate_default("ridge_cph_lifelines_LSC17", TCGA_LSC17)        
    TCGA_PCA17_params = HyperParams.generate_default("ridge_cph_lifelines_PCA", TCGA_PCA17)   
    TCGA_PCA_params = {"min_col": 0, "max_col": TCGA_PCA17.x.shape[1], "pca_n": 17}   
    LGN_CYT_params = HyperParams.generate_default("cytogenetic_risk", LGN_CF)
    LGN_CF_params = HyperParams.generate_default("ridge_cph_lifelines_CF", LGN_CF)

    # CYT ONLY
    cyt = pd.DataFrame(dict([("t", lgn_pronostic["CF"]["Overall_Survival_Time_days"]),("e",  lgn_pronostic["CF"]["Overall_Survival_Status"]), ("pred_risk", lgn_cyt_levels)]))
    cyt_c_scores, cyt_metrics = compute_cyto_risk_c_index(cyt["pred_risk"], cyt, gamma = 0.001, n = HyperParams.bootstr_n)
    LGN_CYT_params["c_index_metrics"] = cyt_metrics
    
    # plot_c_surv_3_groups([0,0, cyt, LGN_CYT_params], args.OUTPATH, group_weights = Counter(lgn_cyt_levels))
    
    # # CF 
    plot_c_surv_3_groups(cox_models.evaluate(LGN_CF, LGN_CF_params), args.OUTPATH, group_weights = Counter(lgn_cyt_levels))
      

    # # GE WITH CF 
    plot_c_surv_3_groups(cox_models.evaluate(LGN_CF_LSC17, LGN_CF_LSC17_params), args.OUTPATH, group_weights = Counter(lgn_cyt_levels))
    plot_c_surv_3_groups(cox_models.evaluate(LGN_CF_PCA17, LGN_CF_PCA17_params, pca_params = {"min_col": 34, "max_col": LGN_CF_PCA17.x.shape[1], "pca_n": 17} ), args.OUTPATH, group_weights = Counter(lgn_cyt_levels))

    # # GE NO CF  
    plot_c_surv_3_groups(cox_models.evaluate(TCGA_LSC17, TCGA_LSC17_params), args.OUTPATH, group_weights = Counter(tcga_cyt_levels))
    plot_c_surv_3_groups(cox_models.evaluate(TCGA_PCA17, TCGA_PCA17_params, pca_params = TCGA_PCA_params), args.OUTPATH, group_weights = Counter(tcga_cyt_levels))
    plot_c_surv_3_groups(cox_models.evaluate(LGN_LSC17, LGN_LSC17_params), args.OUTPATH, group_weights = Counter(lgn_cyt_levels))
    plot_c_surv_3_groups(cox_models.evaluate(LGN_PCA17, LGN_PCA17_params, pca_params = LGN_PCA_params), args.OUTPATH, group_weights = Counter(lgn_cyt_levels))
    plot_c_surv(cox_models.evaluate(LGN_INT_LSC17, LGN_INT_LSC17_params), args.OUTPATH, group_weights = Counter(lgn_cyt_levels))
    plot_c_surv(cox_models.evaluate(LGN_INT_PCA17, LGN_INT_PCA17_params, pca_params = LGN_INT_PCA_PARAMS), args.OUTPATH, group_weights = Counter(lgn_cyt_levels))
    
    
    
    # TO DO
    # plot_c_surv(cox_models.evaluate(LGN_INT_LSC17, LGN_INT_LSC17_params), args.OUTPATH, group_weights = Counter(cyt_levels))
    # plot_c_surv(cox_models.evaluate(LGN_INT_PCA17, LGN_INT_PCA17_params, pca_params = LGN_INT_PCA_PARAMS), args.OUTPATH, group_weights = Counter(cyt_levels))
    


    
    