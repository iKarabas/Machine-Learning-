class KNN:
    def __init__(self, dataset, data_label, similarity_function, similarity_function_parameters=None, K=1):
        """
        :param dataset: dataset on which KNN is executed, 2D numpy array
        :param data_label: class labels for each data sample, 1D numpy array
        :param similarity_function: similarity/distance function, Python function
        :param similarity_function_parameters: auxiliary parameter or parameter array for distance metrics
        :param K: how many neighbors to consider, integer
        """
        self.K = K
        self.dataset = dataset
        self.dataset_label = data_label
        self.similarity_function = similarity_function
        self.similarity_function_parameters = similarity_function_parameters
    
    def get_distance(self, item):
        return item[1]    
        
    def calculate_distances(self, y):
        distances = []
        for x in self.dataset:
            distance = self.similarity_function(x, y, self.similarity_function_parameters)
            distances.append(distance)
        return distances    
        
    def predict(self, instance):
        # calculate all the distances 
        distances =  self.calculate_distances(instance)
        
        # sort according to their distance
        enumerated_distances = enumerate(distances)
        sorted_distances = sorted(enumerated_distances, key=self.get_distance)
        
        # find the labels of closest K points
        k_nearest_points_labels = []
        for i in range(self.K):
            k_nearest_points_labels.append( self.dataset_label[sorted_distances[i][0]])
        
        
        # count occurrences of each label
        label_counts = {}
        for label in k_nearest_points_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # the label with the maximum count
        max_label = None
        max_count = -1
        for label, count in label_counts.items():
            if count > max_count:
                max_label = label
                max_count = count
        
        # return the label with the maximum count
        return max_label
        
