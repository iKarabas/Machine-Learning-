import numpy as np
from Distance import Distance

class KMedoids:
    def __init__(self, dataset, K=2, distance_metric="cosine"):
        """
        :param dataset: 2D numpy array, the whole dataset to be clustered
        :param K: integer, the number of clusters to form
        """
        self.K = K
        self.dataset = dataset
        self.distance_metric = distance_metric
        # each cluster is represented with an integer index
        # self.clusters stores the data points of each cluster in a dictionary
        # In this dictionary, you can keep either the data instance themselves or their corresponding indices in the dataset (self.dataset).
        self.clusters = {i: [] for i in range(K)}
        # self.cluster_medoids stores the cluster medoid for each cluster in a dictionary
        # # In this dictionary, you can keep either the data instance themselves or their corresponding indices in the dataset (self.dataset).
        self.cluster_medoids = {i: None for i in range(K)}
        # you are free to add further variables and functions to the class
        self.max_iterations = 10
        # Create the distance matrix
        self.num_instances, self.num_features = self.dataset.shape
        self.distance_table = None
    
    

    
    def calculateLoss(self):
        """Loss function implementation of Equation 2"""
        total_loss = 0
        for cluster_index, medoid_index in self.cluster_medoids.items():
            cluster_points = self.clusters[cluster_index]  
            loss= 0
            for point_index in cluster_points:
                loss += self.distance_table[medoid_index][point_index]
            total_loss += loss 
        return total_loss


    def run(self, distance_table):
        """Kmedoids algorithm implementation"""
        self.distance_table = distance_table
        # Initialize medoids randomly
        # we replace=False so that no point is sampled more than once
        medoid_indices = np.random.choice(self.num_instances, self.K, replace=False)
        for i, medoids_index in enumerate(medoid_indices):
            self.cluster_medoids[i] = medoids_index

        for _ in range(self.max_iterations):
            
            # Create empty clusters
            self.clusters = {i: [] for i in range(self.K)}
            
            # Assign each point to nearest cluster
            
            for point_index in range(self.num_instances):
                
                # find the cluster that is closest to the current point
                min_distance = float('inf')
                nearest_cluster = None

                for k in range(self.K):
                    medoid_index = self.cluster_medoids[k]
                    distance = self.distance_table[point_index][medoid_index]

                    if distance < min_distance:
                        min_distance = distance
                        nearest_cluster = k
                # Assign current point to the nearest cluster
                self.clusters[nearest_cluster].append(point_index)    
            
             
            # Update medoids
            for cluster_index, cluster_points in self.clusters.items():
                cluster_distances = []

                for point1 in cluster_points:
                    distance_sum = sum(self.distance_table[point1][point2] for point2 in cluster_points)
                    cluster_distances.append(distance_sum)

                min_distance_index = None
                min_distance = float('inf')

                for i, distance_sum in enumerate(cluster_distances):
                    if distance_sum < min_distance:
                        min_distance = distance_sum
                        min_distance_index = cluster_points[i]

                self.cluster_medoids[cluster_index] = min_distance_index


        return self.cluster_medoids, self.clusters, self.calculateLoss()




# the original code without faster version which calculates the distance_table only once
"""   

import numpy as np
from Distance import Distance

class KMedoids:
    def __init__(self, dataset, K=2, distance_metric="cosine"):
        
        #:param dataset: 2D numpy array, the whole dataset to be clustered
        #param K: integer, the number of clusters to form
        
        self.K = K
        self.dataset = dataset
        self.distance_metric = distance_metric
        # each cluster is represented with an integer index
        # self.clusters stores the data points of each cluster in a dictionary
        # In this dictionary, you can keep either the data instance themselves or their corresponding indices in the dataset (self.dataset).
        self.clusters = {i: [] for i in range(K)}
        # self.cluster_medoids stores the cluster medoid for each cluster in a dictionary
        # # In this dictionary, you can keep either the data instance themselves or their corresponding indices in the dataset (self.dataset).
        self.cluster_medoids = {i: None for i in range(K)}
        # you are free to add further variables and functions to the class
        self.max_iterations = 10
        # Create the distance matrix
        self.num_instances, self.num_features = self.dataset.shape
        self.distance_table = [[0 for _ in range(self.num_instances)] for _ in range(self.num_instances)]
    
    
    def calculate_distance(self, point1, point2):
        if self.distance_metric == "cosine":
            # Cosine distance
            return Distance.calculateCosineDistance(point1, point2)
        else:
            # Implement other distance metrics if needed
            pass
    
    def calculateLoss(self):
        #Loss function implementation of Equation 2
        total_loss = 0
        for cluster_index, medoid_index in self.cluster_medoids.items():
            cluster_points = self.clusters[cluster_index]  
            for point_index in cluster_points:
                total_loss += self.distance_table[medoid_index][point_index]

        return total_loss


    def run(self):
        #Kmedoids algorithm implementation
        # Fill the distance table
        for i in range(self.num_instances):
            for j in range(self.num_instances):
                self.distance_table[i][j] = self.calculate_distance(self.dataset[i], self.dataset[j])
        # Initialize medoids randomly
        # we replace=False so that no point is sampled more than once
        medoid_indices = np.random.choice(self.num_instances, self.K, replace=False)
        for i, medoids_index in enumerate(medoid_indices):
            self.cluster_medoids[i] = medoids_index

        for _ in range(self.max_iterations):
            
            # Create empty clusters
            self.clusters = {i: [] for i in range(self.K)}
            
            # Assign each point to nearest cluster
            
            for point_index in range(self.num_instances):
                
                # find the cluster that is closest to the current point
                min_distance = float('inf')
                nearest_cluster = None

                for k in range(self.K):
                    medoid_index = self.cluster_medoids[k]
                    distance = self.distance_table[point_index][medoid_index]

                    if distance < min_distance:
                        min_distance = distance
                        nearest_cluster = k
                # Assign current point to the nearest cluster
                self.clusters[nearest_cluster].append(point_index)    
            
             
            # Update medoids
            for cluster_index, cluster_points in self.clusters.items():
                cluster_distances = []

                for point1 in cluster_points:
                    distance_sum = sum(self.distance_table[point1][point2] for point2 in cluster_points)
                    cluster_distances.append(distance_sum)

                min_distance_index = None
                min_distance = float('inf')

                for i, distance_sum in enumerate(cluster_distances):
                    if distance_sum < min_distance:
                        min_distance = distance_sum
                        min_distance_index = cluster_points[i]

                self.cluster_medoids[cluster_index] = min_distance_index


        return self.cluster_medoids, self.clusters, self.calculateLoss()


"""