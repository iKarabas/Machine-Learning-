import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.decomposition import PCA 

dataset1 = pickle.load(open("../data/part2_dataset_1.data", "rb"))
dataset2 = pickle.load(open("../data/part2_dataset_2.data", "rb"))






def perform_and_plot_dimensionality_reductions(reducer, dataset, title):
    # do the reduction 
    reduced_data = reducer.fit_transform(dataset)
    # plot the reduced data
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], marker='.')
    plt.title(title)
    plt.show()
    
# PCA visualization
pca = PCA(n_components=2)  
perform_and_plot_dimensionality_reductions(pca, dataset1, 'PCA Visualization - Dataset 1')
perform_and_plot_dimensionality_reductions(pca, dataset2, 'PCA Visualization - Dataset 2')    

# t-SNE visualization
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
perform_and_plot_dimensionality_reductions(tsne, dataset1, 't-SNE Visualization - Dataset 1')
perform_and_plot_dimensionality_reductions(tsne, dataset2, 't-SNE Visualization - Dataset 2')

# UMAP visualization
umap_model = UMAP(n_neighbors=5, min_dist=0.3)
perform_and_plot_dimensionality_reductions(umap_model, dataset1, 'UMAP Visualization - Dataset 1')
perform_and_plot_dimensionality_reductions(umap_model, dataset2, 'UMAP Visualization - Dataset 2')


