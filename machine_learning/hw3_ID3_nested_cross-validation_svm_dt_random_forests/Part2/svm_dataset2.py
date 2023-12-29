import pickle
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


dataset, labels = pickle.load(open("data/part2_dataset2.data", "rb"))

# Data preprocessing with StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(dataset)

# Define parameter grid for SVM
param_grid = [(0.1, 'linear'), (1, 'linear'), (10, 'linear'), (100, 'linear'),
              (0.1, 'rbf'), (1, 'rbf'), (10, 'rbf'), (100, 'rbf')]


best_accuracy = 0
best_configuration = param_grid[0]
# Go over every configuration
for configuration in param_grid:
    regularization_parameter, kernel = configuration
    mean_accuracies_of_cross_val = []
    # Repeat 5 times for each configuration
    for iteration in range(5):
        kfolds = KFold(n_splits=10, shuffle=True, random_state= iteration + 1)
        
        # Store accuracy scores for each iteration
        scores = []
        # Perform 10-fold cross-validation
        for train_index, test_index in kfolds.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            
            # Initialize a new SVM classifier for each fold
            svc = SVC(C=regularization_parameter, kernel=kernel)
            # Fit the classifier on the training data
            svc.fit(X_train, y_train)
            # Evaluate on the test data
            accuracy = svc.score(X_test, y_test)
            scores.append(accuracy)

        # Calculate and print the mean accuracy for this configuration
        mean_accuracies_of_cross_val.append(np.mean(scores))
    
    overall_average_accuracy = float(np.mean(mean_accuracies_of_cross_val))
    if overall_average_accuracy > best_accuracy:
        best_accuracy = overall_average_accuracy
        best_configuration = configuration
        
    print(f"Configuration: Regularization parameter = {regularization_parameter} | Kernel = {kernel} || Accuracy = {overall_average_accuracy}")

print()
print(f"Best Configuration: Regularization parameter = {best_configuration[0]} | Kernel = {best_configuration[1]} || Accuracy = {best_accuracy}")    