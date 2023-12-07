import pickle
from Distance import Distance
from Knn import KNN
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np

# Load dataset
dataset, labels = pickle.load(open("../data/part1_dataset.data", "rb"))

# Define the k values
k_values = [1, 2, 3, 4, 5, 7]

# Define the number of iterations (shuffling and k-fold cross-validation)
num_iterations = 5

# Define distance measure types
distance_measures = ['cosine', 'minkowski', 'mahalanobis']

# Perform hyperparameter tuning and confidence interval calculation
for distance_measure in distance_measures:
    print(f"Testing KNN with DISTANCE MEASURE: {distance_measure}")
    
    for iteration in range(num_iterations):
        # Set a different random state for each iteration
        random_state = iteration + 1

        # Create the stratifier
        stratifier = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)

        k_with_max_accuracy = -1
        max_accuracy = 0
        accuracies_per_k = {}

        for k in k_values:
            local_accuracies = []

            # Create the k folded data and use each one directly
            for train_indices, test_indices in stratifier.split(dataset, labels):
                X_train, X_test = dataset[train_indices], dataset[test_indices]
                y_train, y_test = labels[train_indices], labels[test_indices]

                # Train and predict using the KNN model with the specified distance measure
                if distance_measure == 'cosine':
                    knn = KNN(dataset=X_train, data_label=y_train, similarity_function=Distance.calculateCosineDistance, K=k)
                elif distance_measure == 'minkowski':
                    knn = KNN(dataset=X_train, data_label=y_train, similarity_function=Distance.calculateMinkowskiDistance,similarity_function_parameters=2, K=k)
                elif distance_measure == 'mahalanobis':
                    covariance_matrix = np.cov(dataset, rowvar=False) 
                    S_minus_1 = np.linalg.inv(covariance_matrix)
                    knn = KNN(dataset=X_train, data_label=y_train, similarity_function=Distance.calculateMahalanobisDistance, similarity_function_parameters=S_minus_1, K=k)

                predictions = [knn.predict(instance) for instance in X_test]
                accuracy = accuracy_score(y_test, predictions)
                local_accuracies.append(accuracy)

            # Calculate the average accuracy and standard deviation
            average = np.mean(local_accuracies)
            std_dev = np.std(local_accuracies, ddof=1)  # ddof=1 for sample standard deviation

            # Calculate the critical value from the t-distribution table
            confidence_level = 0.95  # Change as needed
            degrees_of_freedom = len(local_accuracies) - 1
            t_critical_value = 2.262  # For a two-tailed 95% confidence interval with 9 degrees of freedom, using students table 

            # Calculate the margin of error
            margin_of_error = t_critical_value * (std_dev / np.sqrt(len(local_accuracies)))
            confidence_interval = (average - margin_of_error, average + margin_of_error)

            accuracies_per_k[k] = {
                'average': average,
                'std_dev': std_dev,
                'confidence_interval': confidence_interval
            }

            # Update max accuracy and corresponding k value
            if average > max_accuracy:
                max_accuracy = average
                k_with_max_accuracy = k

        print(f"Iteration {iteration + 1} , k value with max accuracy: {k_with_max_accuracy}, accuracy: {max_accuracy:.4f}")
        print("Accuracies per k:")
        for k, info in accuracies_per_k.items():
            print(f"  k={k}, average={info['average']:.4f}, std_dev={info['std_dev']:.4f}, confidence_interval={info['confidence_interval']}")
        print()
    print("##############################################")
