from Distance import Distance
import numpy as np 

class KMeans:

    
    def __init__(self, dataset, K=2):
        """
        :param dataset: 2D numpy array, the whole dataset to be clustered
        :param K: integer, the number of clusters to form
        """
        self.K = K
        self.dataset = dataset
        # each cluster is represented with an integer index
        # self.clusters stores the data points of each cluster in a dictionary
        self.clusters = {i: [] for i in range(K)}
        # self.cluster_centers stores the cluster mean vectors for each cluster in a dictionary
        self.cluster_centers = {i: None for i in range(K)}
        # other variables 
        self.max_iterations= 6  # it should be much higher than this for more accuracy but it would have take to too much time , 
                                #since I am using a basic laptop for calculations, I chose a small value 
        self.convergence_threshold=1e-4

    def euclidean_norm(self, vector):
        return np.sqrt(np.sum(np.square(vector)))

    def calculateLoss(self):
        """Loss function implementation of Equation 1"""
        total_loss = 0
        for k in range(self.K):
            cluster_center = self.cluster_centers[k]
            cluster_points = self.clusters[k]
            for point in cluster_points:
                total_loss += self.euclidean_norm(point - cluster_center)
        return total_loss
        

    def run(self):
        """Kmeans algorithm implementation"""
        
        # INITIALIZATION 
        # Randomly choose K data points as initial cluster centers
        self.cluster_centers = {}
        # we replace=False so that no point is sampled more than once
        for i, idx in enumerate(np.random.choice(len(self.dataset), self.K, replace=False)):
            self.cluster_centers[i] = self.dataset[idx]

        # LOOP TILL CONVERGENCE OR MAX NUMBER OF ITERATIONS
        for iteration in range(self.max_iterations):
            # Create new clusters
            new_clusters = {}
            for i in range(self.K):
                new_clusters[i] = []
            
            # Assign each data point to closest cluster
            for data_point in self.dataset:
                # Find the nearest cluster center
                nearest_cluster = None
                min_distance = float('inf')

                for i in range(self.K):
                    distance = self.euclidean_norm(data_point - self.cluster_centers[i])
    
                    if distance < min_distance:
                        min_distance = distance
                        nearest_cluster = i
                
                new_clusters[nearest_cluster].append(data_point)

            # Calculate new cluster centers as the mean of assigned data points
            new_centers = {}
            for i in range(self.K):
                # if the new cluster has no points, keep the center the same
                if(len(new_clusters[i])):
                    new_centers[i] = np.mean(new_clusters[i], axis=0)
                else:
                    new_centers[i] = self.cluster_centers[i]
            
            # Check for convergence: Stop if cluster centers have not changed significantly
            center_changes = [self.euclidean_norm(new_centers[i] - self.cluster_centers[i]) for i in range(self.K)]
            
            significant_change_flag = False
            for change in center_changes:
                if change > self.convergence_threshold:
                    significant_change_flag = True
            # If there were no significant change stop the iteration        
            if significant_change_flag == False:
                break
            
            # Update cluster assignments and centers
            self.clusters = new_clusters
            self.cluster_centers = new_centers
        
        return self.cluster_centers, self.clusters, self.calculateLoss()
