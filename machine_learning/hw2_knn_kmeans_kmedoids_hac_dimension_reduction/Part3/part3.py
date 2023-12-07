import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# load the dataset
dataset = pickle.load(open("../data/part3_dataset.data", "rb"))

# hyperparameters and their range
linkage_types = ['single', 'complete']
distance_types = ['euclidean', 'cosine']
K_values = [2, 3, 4, 5]

# main loop for all configurations
for linkage_type in linkage_types:
    for distance_type in distance_types:
        # plotting the dendograms
        plt.figure(figsize=(8, 4))
        plt.title(f"Dendrogram with Linkage type: {linkage_type}, Distance type: {distance_type}")
        dendrogram(linkage(dataset, method=linkage_type, metric=distance_type), truncate_mode='level', p=5)
        plt.xlabel("Data Points")
        plt.ylabel("Distance")
        plt.show()

        # silhouette analysis for K values
        silhouette_scores = []
        for k in K_values:
            # create the clusters
            cluster_calculater = AgglomerativeClustering(n_clusters=k, linkage=linkage_type, metric=distance_type)
            clusters = cluster_calculater.fit_predict(dataset)
            silhouette_avg = silhouette_score(dataset, clusters)
            silhouette_scores.append(silhouette_avg)
            print(f"Linkage type: {linkage_type}, Distance type: {distance_type}, K: {k}, Silhouette Score: {silhouette_avg:.4f}")
            
        # plotting silhouette scores
        plt.plot(K_values, silhouette_scores, marker='o')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Silhouette Score')
        plt.title(f'Silhouette Analysis - Linkage type: {linkage_type}, Distance type: {distance_type}')
        plt.show()
