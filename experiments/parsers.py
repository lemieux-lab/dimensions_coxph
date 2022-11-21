import argparse 

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_experiment", dest = "EXP", type = str, default = "4", help = "experiment number and version")
    # add control arguments
    parser.add_argument("-O", dest = "OUTPATH", type = str, default = "RES/FIGS/", help = "generated output filepath, (will override existing)")
    parser.add_argument("-C", dest = "COHORT", type = str, default = "lgn_pronostic", help = "public: \tThe Leucegene public subset = 403 samples. Curated samples subset to complete different feature prediction on.\n pronostic: The Leucegene pronostic subset = 300 samples. Curated samples subset that was selected to perform survival analysis on. \n lgn_intermediate: From The Leucegene pronostic subset, intermediate cytogenetic risk, n samples = 177. \n tcga_target_aml")
    parser.add_argument("-N_REP", dest = "NREP_TECHN", default = 10, type = int, help = "number of technical replicates") 
    parser.add_argument("-IN_D", dest = "INPUT_DIMS", default = [17, 18, 1], type = int, nargs = "+", help = "range of number of input dimensions to test (min,max, step) default= [17,18,1]")
    parser.add_argument("-WD", dest = "WEIGHT_DECAY", default = 1e-4, type = float, help = "l2 regularization strength, default = 0.001")
    parser.add_argument("-BN", dest = "bootstr_n", default = 1000, type = int, help = "bootstrap ")
    # for the input projection types
    help = "list of projection types for survival prediction and further analyses default = [PCA, SVD, LSC17, RPgauss, RPsparse, RSelect]"
    parser.add_argument("-P", dest = "PROJ_TYPES", type = str, nargs = "+", default = ["PCA", "SVD", "LSC17", "RPgauss", "RPsparse", "RSelect"], help= help)
    parser.add_argument("-M", dest = "MODEL_TYPES", type = str, nargs = "+", default = ["cphdnn"], help= "list of models to perform survival modelling.")
    parser.add_argument("-CYT", dest = "PLOT_CYTO_RISK", action = "store_true", help= "exp 2: run the km fitting from cyto groups.")
     
    # OLD
    parser.add_argument("-d", dest = "debug", action="store_true", help = "debug")
    parser.add_argument("-W", dest = "WIDTHS", nargs = "+", type = str,  default = ["CDS", "TRSC"], help = "Dimensionality of input features space. \n CDS: Small transcriptome ~= 19,500 variables \nTRSC: Full transcriptome ~= 54,500 transcripts. Can put as many jobs in this queue. Jobs will be done sequentially" )
    parser.add_argument("-N_TSNE", dest = "N_TSNE", type = int,  default = 1, help = "Number of T-SNE replicates done if TSNE selected. To check reproducibility." )
    parser.add_argument("-MAX_PC", dest = "MAX_PC", type = int,  default = 10, help = "Number of PC to be analysed (GO enrichment, figures)" )
    parser.add_argument("-GO_TOP_N", dest = "GO_TOP_N", type = int,  default = 1000, help = "Number of genes in gene set when performing GO enrichment analysis" )
    parser.add_argument("-FIXED_EMB", dest = "EMB_FILE", type = str, default = "Data/emb125_MLP25.csv", help = "name of embedding file used for fixed embedding mode CPH training prediction.")
    parser.add_argument("-N_OPTIM", dest = "NREP_OPTIM", default = 1, type = int, help = "number of optimizations for Hyper parameters")
    parser.add_argument("-N_FOLDS", dest = "NFOLDS", default = 5, type = int, help = "number of crossvalidation folds") 
    parser.add_argument("-E", dest = "NEPOCHS", default = 200, type = int, help = "number of epochs for optim of DNN models")
    