# other
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import lifelines
import numpy as np
import pandas as pd
from tqdm import tqdm 
import umap
# base
import os
import pdb
import itertools as it
# custom 
import engines.models.utils as utils 
from engines.models.obo.read import read_obo 


proj_picker = {"TSNE": TSNE, "UMAP":umap.UMAP}
def plot_factorized_embedding(ds, embedding, MSEloss, emb_size, e, cohort = "public", method = "UMAP"):
    # manage colors 
    #colors = ["b", "g", "c", "y", "k","lightgreen", "darkgrey", "darkviolet"]
    r = lambda: np.random.randint(0,255)
    flatui = []
    for i in range(53):
        flatui.append('#%02X%02X%02X' % (r(),r(),r()))

    # plot cyto group
    markers = ['<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']*5
    np.random.shuffle(markers)
    # fix label
    feature = "Cytogenetic group"
    #perplexity = 30
    #data_folder = "/u/sauves/FactorizedEmbedding/run_LS_emb17/50934a761eefc71f19f324b1953fc056/"
    print(f"Epoch {e} - Plotting with {method} ...")
    
    #tsne = TSNE(n_components = 2, perplexity= perplexity, verbose =1, init = "pca")
    #proj_x = tsne.fit_transform(data)
    reducer = proj_picker[method]()
    proj_x = reducer.fit_transform(embedding)
    fig = plt.figure(figsize = (20,10))
    for i, cyto_group in enumerate(np.unique(ds.y[feature])):
        plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)     
        X = proj_x[ds.y[feature] == cyto_group][:,0]
        Y = proj_x[ds.y[feature] == cyto_group][:,1]
        c = flatui[i%len(flatui)]
        m = markers[i%len(markers)]
            
        plt.scatter(X, Y, label = cyto_group[:50] + "...", color = c , marker = m)
    caption = f"Leucegene {cohort} - {emb_size}-D Embedding - Epoch {e} - by {feature} - Factorized Embedding {method} -\n From CDS - With {ds.x.shape[0]} Samples and {ds.x.shape[1]} features"
    plt.title(caption)
    plt.xlabel(f"{method}- EMBEDDING")
    plt.ylabel(f"{method} - EMBEDDING")
    MIN = np.array(proj_x).flatten().min() - 1
    MAX = np.array(proj_x).flatten().max() + 1
    plt.xlim([MIN,MAX])
    plt.ylim([MIN,MAX])
    plt.box(False)
    plt.legend(bbox_to_anchor=(-0.5, 1.0), loc='upper left')
    plt.gca().set_aspect('equal')
    fname = f"RES/FIGS/proj_by_epoch_FE/embedding{emb_size}_{method}_epoch_{e}"
    plt.savefig(f"{fname}.svg")
    plt.savefig(f"{fname}.png")
    plt.savefig(f"{fname}.pdf")

def compute_cyto_risk_c_index(scores, target, gamma = 0.01, n = 10):
    c_scores = []
    for i in tqdm(range(n)):
        noise_risk_scores =  scores + np.random.normal(0, gamma, size = len(scores))
        c_scores.append(compute_c_index(noise_risk_scores, target, method = "own"))
    c_scores = np.sort(c_scores)
    c_ind_med = np.median(c_scores)
    c_ind_min = c_scores[int(n*0.05)]
    c_ind_max = c_scores[int(n*0.95)]
    return c_scores, (c_ind_med, c_ind_min, c_ind_max)

def compute_pca_loadings(data, pca_params):
    """
    From sklearn source code 
    """
    if pca_params is None: return None
    pca = PCA()
    pca.fit(data.iloc[:,pca_params["min_col"]:pca_params["max_col"]])

    return {"components": pca.components_[:pca_params["pca_n"]], "mean" : pca.mean_}

def compute_aggregated_bootstrapped_c_index(scores, data, n=10000):
    c_scores = []
    nsamples=data.shape[0]
    for i in tqdm(range (n), desc = f"bootstraping {n}..."):
        idx = np.random.choice(np.arange(nsamples),nsamples, True)
        c_ind_vld = compute_c_index(scores[idx], data.iloc[idx,:], method = "lifelines")
        c_scores.append(c_ind_vld)
    c_scores = np.sort(c_scores)
    c_ind_med = np.median(c_scores)
    c_ind_min = c_scores[int(n*0.05)]
    c_ind_max = c_scores[int(n*0.95)]
    return c_scores, (c_ind_med, c_ind_min, c_ind_max)

def compute_c_index(scores, data, method = "lifelines"):
    def _lifelines(scores,data):
        aggr_c_ind = lifelines.utils.concordance_index(data["t"], scores, data["e"])
        if aggr_c_ind < 0.5:
            aggr_c_ind = lifelines.utils.concordance_index(data["t"], -scores, data["e"])
            
        return aggr_c_ind
    
    def _own_it(scores, data):
        # iterative version of c index computations
        ### VERY SLOW
        n = len(scores)
        s = 0
        ap = 0
        for i in range(n):
            for j in range(n):
                # check if admissible pair
                if (data["t"][i] < data["t"][j]) and (data["e"][i]):
                    ap += 1
                    if scores[i] > scores[j]:
                        s += 1
                    if scores[i] == scores[j]:
                        s += 0.5
        c_index = float(s) / ap
        return c_index

    def _own_vec(scores, data):
        # vector computations of c index 
        # FAST
        n = len(scores)
        censoring = np.array(data["e"] == 1, dtype = int).reshape((1,n))
        surv_times = np.array(data["t"]).reshape((1,n))
        partial_hazards = np.array(scores).reshape((1,n))
        concordant_pairs = partial_hazards > partial_hazards.T
        tied_pairs = (partial_hazards == partial_hazards.T).sum(0) - np.ones(n)
        admissable_pairs = surv_times < surv_times.T
        c_ind = (censoring * admissable_pairs * concordant_pairs).sum() + (0.5 * tied_pairs.sum())
        c_ind = float(c_ind) / ((censoring * admissable_pairs.sum(0)).sum() + tied_pairs.sum())
        
        # censoring * surv_times > surv_times.T
        return c_ind

    if method == "lifelines": return _lifelines(scores, data)
    elif method == "own": return _own_vec(scores, data)

def compute_cox_nll(t, e, out): 
    # sort indices 
    sorted_indices = np.argsort(t)
    E = e[sorted_indices] # event obs. sorted
    OUT = out[sorted_indices] # risk scores sorted
    uncensored_likelihood = np.zeros(E.shape[0])# list of uncensored likelihoods
    for x_i, E_i in enumerate(E): # cycle through samples
        if E_i == 1: # if uncensored ...
            log_risk = np.log(np.sum(np.exp(OUT[:x_i +1])))
            uncensored_likelihood[x_i] = OUT[x_i] - log_risk # sub sum of log risks to hazard, append to uncensored likelihoods list
    
    loss = - uncensored_likelihood.sum() / (E == 1).sum() 
    return loss 

def XL_mHG_test(gene_set, go_matrix):
    pdb.set_trace()
    pass

def get_enrichment(loading_scores, GO_terms, top_n = 10):
    pdb.set_trace()
    # rank scores
    gene_set = loading_scores.sort_values()[np.concatenate([np.arange(top_n), np.arange(loading_scores.shape[0] - top_n, loading_scores.shape[0])])]
    # make enrichment test
    go_scores = []
    for GO_term in GO_terms.keys():
        go_scores.append(XL_mHG_test(gene_set, GO_terms))
    # filter go terms 
    go_scores_filtered = go_scores[go_scores["adjusted_p_val"] < 0.05]
    # store 
    return go_scores_filtered

def preselection_of_GO_terms(gene_info):
    # load GO struct
    if not os.path.exists("Data/goa_human.gaf"):
        print ("loading and parsing gaf file ...")
        os.system("rm *.gaf*")
        os.system("wget http://geneontology.org/gene-associations/goa_human.gaf.gz")
        os.system("gunzip goa_human.gaf.gz ")
        os.system("mv goa_human.gaf Data")
    columns = np.concatenate([["db_name", "db_id", "SYMBOL", "rship", "GO_term", "ref",  "EV_code", "PMID", "undef", "descr", "alt_names"], np.arange(11, 17).astype(str)])
    assoc = pd.read_csv("Data/goa_human.gaf", sep = "\t", skiprows = 41, header = None)
    assoc.columns = columns 
    # further parse assoc.... 
    assoc = assoc[["SYMBOL","GO_term", "EV_code", "descr", "alt_names"]]
    evidences = ["IDA","IGI","IMP","ISO","ISS","IC","NAS","TAS"]
    assoc = assoc[assoc.EV_code.isin(evidences)]
    # get counts by GO 
    counts = assoc.groupby("GO_term").count().sort_values("SYMBOL", ascending = False)
    # remove too broad GO
    counts = counts[counts.SYMBOL < 200]
    # remove too specific GO
    counts = counts[counts.SYMBOL > 5]
    pdb.set_trace()
    if not os.path.exists("Data/go-basic.obo"):
        print("loading and parsing obo file ...")
        os.system("rm *.obo")
        os.system("wget http://purl.obolibrary.org/obo/go/go-basic.obo")
        os.system("mv go-basic.obo Data")
    obo = read_obo(open("Data/go-basic.obo"))
    # understand how the structure works.
    for n, nbrsdict in obo.adjacency():
        print (n)
        for nbr, keydict in nbrsdict.items():
                pdb.set_trace()



    # filter CDS if needed
    # propagate through struct
    # filter to get about 7,500 terms
    go_terms_filtered = go_terms
    # return G x m matrix with G number of selected genes, by m number of go terms retained. where each entry is presence / absence
    M = []
    go_terms = []
    for go_term in go_terms_filtered.keys():
        go_terms.append(go_term)
        G = []
        for gene in gene_info.SYMBOL:
            G.append(int(gene in go_terms_filtered[go_term]))
        M.append(G)
    return pd.DataFrame(np.matrix(M), index = go_terms, columns = gene_info.SYMBOL) 

def get_ontologies(datasets, pca,  cohort, width, outpath, max_pc, top_n):
    print("performing Gene Ontology analysis... ")
    GO_terms = preselection_of_GO_terms(datasets[cohort].gene_info)
    print(f"\tcohort: {cohort}, width: {width}, max_pcs: {max_pc}, top_n:{top_n} ...")
    # get pc loadings
    loadings = pd.DataFrame(pca["pca"].components_, columns = datasets[cohort].data[width].x.columns).T
    # store ??
    # get enrichments
    # verify ref genome is there
    RES = []
    for PC in range(max_pc):
        
        enrichm = get_enrichment(loadings[PC], GO_terms, top_n = top_n)
        RES.append(enrichm)

def generate_tsne(datasets, cohort, width, outpath, N =1):
    
    data = []
    p = np.random.randint(15,30) # select perplexity
    datasets[cohort].data[width].create_shuffles(N) # create shuffles data
    for n in range(N): 
        datasets[cohort].data[width].select_shuffle(n)
        X = datasets[cohort].data[width].x 
        print(f"Running TSNE Rep {n +1} ...")
        tsne_engine = TSNE(perplexity=p, init = "pca", verbose = 1) # random perplexity
        proj_x = tsne_engine.fit_transform(X)
        data.append({"proj_x": proj_x, "tsne": tsne_engine})
    return data


def pca_plotting(datasets, pca_data, cohort, width, base_outpath):
    
    maxPCs = 3

    for pc1, pc2 in [x for x in it.combinations(np.arange(maxPCs),2)]: 
        outpath = utils.assert_mkdir(os.path.join(base_outpath, cohort, width, "PCA",  f"_[{pc1 + 1},{pc2 +1 }]" ))
        
        # for features in datasets do smthing...
        X = pca_data["proj_x"].values
        Y = datasets[cohort].data[width].y
        var_expl_pc1 = round(pca_data["pca"].explained_variance_ratio_[pc1], 4) * 100
        var_expl_pc2 = round(pca_data["pca"].explained_variance_ratio_[pc2] , 4) * 100
        markers = [".", "v",">", "<","^","p", "P"]
        colors = ["b", "g", "c", "y", "k","lightgreen", "darkgrey", "darkviolet"]
        features = ["WHO classification","Sex", "Cytogenetic group","Induction_Type",'HSCT_Status_Type' ,'Cytogenetic risk', 'FAB classification','Tissue', 'RNASEQ_protocol',"IDH1-R132 mutation" ,'FLT3-ITD mutation',  "NPM1 mutation"]
        print("Plotting figures ...")
        for feature in tqdm(features, desc = f"[{pc1 +1 }, {pc2 + 1}]"):
            # plot pca
            fig = plt.figure(figsize = (20,10))
            plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)     
            np.random.shuffle(markers)
            np.random.shuffle(colors) 
            for i, level in enumerate(np.unique(Y[feature])):
                x1,x2 = X[Y[feature]==level,pc1], X[Y[feature]==level,pc2]
                c = colors[i%len(colors)]
                m = markers[i%len(markers)]
                plt.scatter(x1, x2, c = c, marker = m, label = level)
            plt.legend()
            fname = f"lgn_{cohort}_GE_PCA_{width}_TPM_{feature}_[{pc1+1},{pc2+1}]"
            # Some aesthetics
            plt.xlabel(f"PCA{pc1 + 1} ({var_expl_pc1}%)")
            plt.ylabel(f"PCA{pc2 + 1} ({var_expl_pc2}%)")
            caption = f"Leucegene - {cohort} - by {feature} - With Gene Expression PCA [{pc1 + 1}({var_expl_pc1}%),{pc2 + 1}({var_expl_pc2}%)]-\n From {width} - With {datasets[cohort].NS} Samples and {datasets[cohort].data[width].x.shape[1]} features"
            # center text
            plt.title(caption)
            plt.box(False)
            plt.legend(bbox_to_anchor=(.9, 1.0), loc='upper left')
            #MIN = np.array(X.iloc[:,[pc1,pc2]]).flatten().min() - 1
            #MAX = np.array(X.iloc[:,[pc1,pc2]]).flatten().max() + 1
            #plt.xlim([MIN,MAX])
            #plt.ylim([MIN,MAX])
            plt.gca().set_aspect('equal')
            plt.savefig(f"{outpath}/{fname}.svg")
            plt.savefig(f"{outpath}/{fname}.png")
            plt.savefig(f"{outpath}/{fname}.pdf")
    # loadings
    # var explained 
    # set up fnames, paths
    # titles axes

    # write

def tsne_plotting(datasets, tsne_data, cohort, width, outpath):
    outpath = utils.assert_mkdir(os.path.join(outpath, cohort, width, "TSNE"))
    markers = [".", "v",">", "<","^","p", "P"]
    colors = ["b", "g", "c", "y", "k","lightgreen", "darkgrey", "darkviolet"]
    features = ["WHO classification","Sex", "Cytogenetic group","Induction_Type",'HSCT_Status_Type' ,'Cytogenetic risk', 'FAB classification','Tissue', 'RNASEQ_protocol',"IDH1-R132 mutation" ,'FLT3-ITD mutation',  "NPM1 mutation"]
    print("Plotting figures ...")
    for rep_n, tsne in enumerate(tsne_data):
        datasets[cohort].data[width].select_shuffle(rep_n)
        for feature in tqdm(features, desc =f"PLOTTING REP: [{rep_n +1}] ..."):
            fig = plt.figure(figsize = (20,10))
            plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)     
            np.random.shuffle(markers)
            np.random.shuffle(colors) 
            for i, level in enumerate(np.unique(datasets[cohort].data[width].y[feature])):
                c = colors[i%len(colors)]
                m = markers[i%len(markers)]
                X = tsne["proj_x"][:,0][datasets[cohort].data[width].y[feature] == level]
                Y = tsne["proj_x"][:,1][datasets[cohort].data[width].y[feature] == level]
                
                plt.scatter(X, Y, color = c, marker=m, label = str(level).replace(" or ", "/\n"))
            # Some aesthetics
            plt.xlabel("TSNE-1")
            plt.ylabel("TSNE-2")
            caption = f"Leucegene - {cohort} - by {feature} - With Gene Expression t-SNE p={tsne['tsne'].perplexity} rep#{rep_n+1} -\n From {width} - With {datasets[cohort].NS} Samples and {datasets[cohort].data[width].x.shape[1]} features"
            # center text
            plt.title(caption)
            plt.box(False)
            plt.legend(bbox_to_anchor=(.9, 1.0), loc='upper left')
            MIN = np.array(tsne["proj_x"]).flatten().min() - 1
            MAX = np.array(tsne["proj_x"]).flatten().max() + 1
            plt.xlim([MIN,MAX])
            plt.ylim([MIN,MAX])
            plt.gca().set_aspect('equal')
            fname = f"{outpath}/lgn_{cohort}_GE_TSNE_{width}_TPM_{feature}_[{rep_n+1}]"
            plt.savefig(f"{fname}.svg")
            plt.savefig(f"{fname}.png")
            plt.savefig(f"{fname}.pdf")