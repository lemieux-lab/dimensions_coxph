import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pdb 

def get_corr_to_cf(X, bin_cf):
    bin_cf = bin_cf.loc[X.index]
    matrix = []
    for name, i in enumerate(range(len(X.columns))):
        row = [name]
        for feature in bin_cf.columns:
            row.append(np.corrcoef(X.iloc[:,i], bin_cf[feature])[0,1])
        matrix.append(row)
    # get corr to features with Lsc17
    return  pd.DataFrame(matrix, columns = np.concatenate([["name"], bin_cf.columns]), index = X.columns)

def compute_pca_loadings(data, pca_n):
    """
    From sklearn source code 
    """
    if pca_n is None: return None
    pca = PCA()
    pca.fit(data)

    return {"components": pca.components_[:pca_n], "mean" : pca.mean_}

def PCA_transform(data, n_components):
    pca = compute_pca_loadings(data, pca_n = n_components) 
    new_df = pd.DataFrame(np.dot(data.values - pca["mean"], pca["components"].T), index = data.index)
    return new_df

def TSNE_transform(data):
    tsne = TSNE(n_components = 2, perplexity= 15, verbose =1, init = "random")
    new_df = pd.DataFrame(tsne.fit_transform(data), index = data.index)
    return new_df