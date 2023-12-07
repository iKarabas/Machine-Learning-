import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.decomposition import PCA 


dataset = pickle.load(open("../data/part3_dataset.data", "rb"))



def perform_and_plot_dimensionality_reductions(reducer, dataset, title):
    # do the reduction 
    reduced_data = reducer.fit_transform(dataset)
    # plot the reduced data
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], marker='.')
    plt.title(title)
    plt.show()
    
# PCA visualization
pca = PCA(n_components=2)  
perform_and_plot_dimensionality_reductions(pca, dataset, 'PCA Visualization - Dataset')
 

# t-SNE visualization
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
perform_and_plot_dimensionality_reductions(tsne, dataset, 't-SNE Visualization - Dataset')

# UMAP visualization
umap_model = UMAP(n_neighbors=5, min_dist=0.3)
perform_and_plot_dimensionality_reductions(umap_model, dataset, 'UMAP Visualization - Dataset')



