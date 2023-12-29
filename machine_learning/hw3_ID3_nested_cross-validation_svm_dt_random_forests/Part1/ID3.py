import numpy as np
import math
# In the decision tree, non-leaf nodes are going to be represented via TreeNode
class TreeNode:
    def __init__(self, attribute):
        self.attribute = attribute
        # dictionary, k: subtree, key (k) an attribute value, value is either TreeNode or TreeLeafNode
        self.subtrees = {}

# In the decision tree, leaf nodes are going to be represented via TreeLeafNode
class TreeLeafNode:
    def __init__(self, data, label):
        self.data = data
        self.labels = label

class DecisionTree:
    def __init__(self, dataset: list, labels, features, criterion="information gain"):
        """
        :param dataset: array of data instances, each data instance is represented via an Python array
        :param labels: array of the labels of the data instances
        :param features: the array that stores the name of each feature dimension
        :param criterion: depending on which criterion ("information gain" or "gain ratio") the splits are to be performed
        """
        self.dataset = dataset
        self.labels = labels
        self.features = features
        self.criterion = criterion
        # it keeps the root node of the decision tree
        self.root = None

        # further variables and functions can be added...


    def calculate_entropy__(self, dataset, labels):
        """
        :param dataset: array of the data instances
        :param labels: array of the labels of the data instances
        :return: calculated entropy value for the given dataset
        """
        # Formula: H(S) = -Σ (p_i * log2(p_i))
         
        # Create an empty dictionary to store label counts
        label_counts = {}

        # Iterate over unique labels in the 'labels' list, use set to be on the safe side
        for label in set(labels):
            # Count occurrences of the current label in the 'labels' list
            count = dataset.count(label)

            # Assign the count to the label in the 'label_counts' dictionary
            label_counts[label] = count

        entropy_value = 0.0
        len_dataset = len(dataset)
        for count in label_counts.values():
            if count != 0:
                probability = count / len_dataset
                entropy_value -= probability * np.log2(probability)

        return entropy_value  

    def calculate_average_entropy__(self, dataset, labels, attribute):
        """
        :param dataset: array of the data instances on which an average entropy value is calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute an average entropy value is going to be calculated...
        :return: the calculated average entropy value for the given attribute
        """
        #Formula: Ave_Entropy(S, A) = Σ (|S_v| / |S| * Entropy(S_v))

        # Get the index of the attribute in the feature list
        attribute_index = self.features.index(attribute)
        
         # Create a dictionary to store subsets of instances based on the attribute values
        subsets = {}
                
        average_entropy = 0.0

        # Fill the sets in the subsets, for every distinct value create a new set
        for i in range(len(dataset)):
            value = dataset[i][attribute_index]
            if value not in subsets:
                subsets[value] = {'data': [], 'labels': []}
            # Assign data to a set    
            subsets[value]['data'].append(dataset[i])
            subsets[value]['labels'].append(labels[i])

        # Calculate the average entropy
        total_num_of_instances = len(dataset)
        for value, subset in subsets.items():
            num_of_subset_instances = len(subset['data'])
            ratio = num_of_subset_instances / total_num_of_instances

            # Calculate entropy and then add it to average entropy according to its ratio
            subset_entropy = self.calculate_entropy__(subset['data'], subset['labels'])
            average_entropy += ratio * subset_entropy

        return average_entropy

    def calculate_information_gain__(self, dataset, labels, attribute):
        """
        :param dataset: array of the data instances on which an information gain score is going to be calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute the information gain score is going to be calculated...
        :return: the calculated information gain score
        """
        # Formula: Info_Gain(S, A) = Entropy(S) - Average_Entropy(S, A)
        
        information_gain = 0.0
        
        # Calculate the entropy of the entire dataset (Entropy(S))
        entropy_s = self.calculate_entropy__(dataset, labels)

        # Calculate the average entropy for the given attribute (Ave_Entropy(S, A))
        ave_entropy_s_a = self.calculate_average_entropy__(dataset, labels, attribute)

        # Calculate information gain using the formula 
        information_gain = entropy_s - ave_entropy_s_a
        
        return information_gain

    def calculate_intrinsic_information__(self, dataset, labels, attribute):
        """
        :param dataset: array of data instances on which an intrinsic information score is going to be calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute the intrinsic information score is going to be calculated...
        :return: the calculated intrinsic information score
        """
        # Formula: Intrinsic_Info(S, A) = - Sum((|Vi| / |S|) * log2(|Vi| / |S|))
        # |Vi|: The number of instances with the attribute value 
        # ∣S∣: The total number of instances in the dataset.
    
        intrinsic_info = None
        
        #  Get unique values (Vi) of the specified attribute in the dataset
        attribute_values = set(instance[self.features.index(attribute)] for instance in dataset)
        
         # Repeatedly used variables
        intrinsic_info = 0.0
        total_num_of_instances = len(dataset)
        attribute_index = self.features.index(attribute)
        
        for value in attribute_values:
            # Count instances with the current attribute value
            value_count = 0
            for instance in dataset:
                instance_attribute_value = instance[attribute_index]      
                if instance_attribute_value == value:
                    value_count += 1

            # Calculate the fraction of instances with the current attribute value
            fraction = value_count / total_num_of_instances

            # Add the current value to the summation
            intrinsic_info -= fraction * math.log2(fraction) if fraction > 0 else 0

        return intrinsic_info
    
    
    
    def calculate_gain_ratio__(self, dataset, labels, attribute):
        """
        :param dataset: array of data instances with which a gain ratio is going to be calculated
        :param labels: array of labels of those instances
        :param attribute: for which attribute the gain ratio score is going to be calculated...
        :return: the calculated gain ratio score
        """
        """
        Note:in the context of decision trees and information gain-based measures, if an attribute has zero intrinsic information (i.e., all instances falling under a specific attribute value have the same class label), it implies that splitting on that attribute will not introduce any impurity. In such cases, the attribute is considered as having a perfect split, and setting the Gain Ratio to zero reflects this.
        """
        # Formula: Gain Ratio = Information Gain / Intrinsic Information
        
        # Calculate the information gain using the specified attribute
        information_gain = self.calculate_information_gain__(dataset, labels, attribute)
        
        # Calculate the intrinsic information for the specified attribute
        intrinsic_info = self.calculate_intrinsic_information__(dataset, labels, attribute)
        # Calculate the ratio while avoiding division by zero by checking 
        if intrinsic_info != 0:
            gain_ratio = information_gain / intrinsic_info
        else:
            # If intrinsic_info is zero, set gain_ratio to zero to avoid division errors
            gain_ratio = 0.0

        return gain_ratio

    
    def choose_best_attribute__(self, dataset, labels, used_attributes):
        """
        Choose the best attribute to split on based on information gain or gain ratio.

        :param dataset: array of data instances
        :param labels: array of labels of the data instances
        :param used_attributes: set of attributes already used in the tree
        :return: the best attribute to split on
        """
        best_attribute = None
        best_score = -1  

        for attribute in self.features:
            if attribute not in used_attributes:
                # Calculate information gain or gain ratio based on the chosen criterion
                if self.criterion == "information gain":
                    score = self.calculate_information_gain__(dataset, labels, attribute)
                elif self.criterion == "gain ratio":
                    score = self.calculate_gain_ratio__(dataset, labels, attribute)
                else:
                    raise ValueError("Error: criterion is not specified")

                # Update the best attribute if the current score is higher
                if score > best_score:
                    best_score = score
                    best_attribute = attribute

        return best_attribute

    
    
    
    
    
    
    def ID3__(self, dataset, labels, used_attributes):
        """
        Recursive function for ID3 algorithm
        :param dataset: data instances falling under the current  tree node
        :param labels: labels of those instances
        :param used_attributes: while recursively constructing the tree, already used labels should be stored in used_attributes
        :return: it returns a created non-leaf node or a created leaf node
        """
        # Base case: all of the remaining dataset have one label, put them in a leaf node
        if len(set(labels)) == 1:
            return TreeLeafNode(dataset, labels[0])

        # Another ending case: if run out of features , take the majority and create a leaf node
        if not self.features:
            majority_label = max(set(labels), key=labels.count)
            return TreeLeafNode(dataset, majority_label)
        
        # Main algorithm 
        # Choose the best attribute to split the current node
        best_attribute = self.choose_best_attribute__(dataset, labels, used_attributes)

        # Create a tree node
        node = TreeNode(best_attribute)

        # Update used_attributes
        used_attributes.append(best_attribute)

        # Find the atribute index and then find all the distinct values at that index 
        attribute_index = self.features.index(best_attribute)
        attribute_values = set(instance[attribute_index] for instance in dataset)

        #Create a new node for each value 
        for value in attribute_values:
            # Split dataset and labels based on the chosen attribute value
            subset_indices = [i for i, instance in enumerate(dataset) if instance[attribute_index] == value]
            subset_dataset = [dataset[i] for i in subset_indices]
            subset_labels = [labels[i] for i in subset_indices]

            # Recursive call to create subtree
            subtree = self.ID3__(subset_dataset, subset_labels, used_attributes.copy())

            # Update the subtree in the non-leaf node
            node.subtrees[value] = subtree

        return node
    def predict(self, x):
        """
        :param x: a data instance, 1 dimensional Python array 
        :return: predicted label of x
        
        If a leaf node contains multiple labels in it, the majority label should be returned as the predicted label
        """
        predicted_label = None
        # Start from the root node and go till you reach a leaf node
        current_node = self.root
        while isinstance(current_node, TreeNode):
            # Find the attribute index of the current node
            attribute_index = self.features.index(current_node.attribute)
            # Find the attribute value of the new data point
            attribute_value = x[attribute_index]
            # Move to the next node according to decision
            current_node = current_node.subtrees[attribute_value]

        # Return the leaf node label
        if isinstance(current_node, TreeLeafNode):
            predicted_label = current_node.labels
        else:
            # In case there were a problem in the implementation
            raise ValueError("Node does not have any labels")

        return predicted_label

    def train(self):
        self.root = self.ID3__(self.dataset, self.labels, [])
        print("Training completed")