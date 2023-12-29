import numpy as np
from DataLoader import DataLoader
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

# Helper functions
def append_performance_metrics(overall_performance, avg_f1, avg_accuracy, best_params):
    overall_performance['f1'].append(avg_f1)
    overall_performance['accuracy'].append(avg_accuracy)
    overall_performance['best hyperparameters'].append(best_params)

def calculate_and_append_confidence_intervals(overall_performance):
    confidence_interval_f1 = np.percentile(overall_performance['f1'], [2.5, 97.5]).tolist()
    confidence_interval_acc = np.percentile(overall_performance['accuracy'], [2.5, 97.5]).tolist()

    overall_performance['confidence_interval_f1'] = confidence_interval_f1
    overall_performance['confidence_interval_acc'] = confidence_interval_acc

def print_performance_metrics(overall_performance, model_name):
    print(f"### {model_name} Performance ###")
    print()
    print("Hyperparameter Configuration\t\tF1\tAccuracy")
    print("-" * 50)

    for i, (best_params, f1, accuracy) in enumerate(zip(overall_performance['best hyperparameters'],
                                                        overall_performance['f1'],
                                                        overall_performance['accuracy'])):
        print(f"{i + 1}. {best_params}\t\t{f1:.4f}\t{accuracy:.4f}")

    print()
    print(f"Average F1: {np.mean(overall_performance['f1']):.4f}")
    print(f"Average Accuracy: {np.mean(overall_performance['accuracy']):.4f}")
    print(f"Confidence Interval (F1): {overall_performance['confidence_interval_f1']}")
    print(f"Confidence Interval (Accuracy): {overall_performance['confidence_interval_acc']}")
    print("\n" + "=" * 50)


# Data loading
data_path = "data/credit.data"
dataset, labels = DataLoader.load_credit_with_onehot(data_path)


# Define parameter grids
knn_parameter_grid = {
    "n_neighbors": [3, 5, 10],
    "metric": ["euclidean", "manhattan"]
}

svm_parameter_grid = {
    "C": [0.1, 0.5],
    "kernel": ["linear", "rbf"]
}

decision_tree_parameter_grid = {
    "max_depth": [10, 20, None]
}

random_forest_parameter_grid = {
    "n_estimators": [5 , 20],
    "max_depth": [10, None]
}

# Number of times to repeat the Random Forest model within the inner loop
num_repeats_inner = 5


# scoring_type = "accuracy"
scoring_types = ["f1", "accuracy"]

for scoring_type in scoring_types:
    print(f"#### scoring used for inner cross validation: {scoring_type} ####")
    print()
    # Set up cross-validation
    outer_cross_validation = RepeatedStratifiedKFold(n_splits=3, n_repeats=5)
    inner_cross_validation = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)
    # mitigating the effects of random nature of random forests algorithm by repeating each inner loop 5 times more
    inner_cross_validation_random_forest = RepeatedStratifiedKFold(n_splits=5, n_repeats=5*num_repeats_inner)
    
    # Performance storage
    knn_overall_performance = {'best hyperparameters': [], 'f1': [], 'accuracy': [], 'confidence_interval_f1': [], 'confidence_interval_acc': []}
    svm_overall_performance = {'best hyperparameters': [],'f1': [], 'accuracy': [], 'confidence_interval_f1': [], 'confidence_interval_acc': []}
    decision_tree_overall_performance = {'best hyperparameters': [],'f1': [], 'accuracy': [], 'confidence_interval_f1': [], 'confidence_interval_acc': []}
    random_forest_overall_performance = {'best hyperparameters': [],'f1': [], 'accuracy': [], 'confidence_interval_f1': [], 'confidence_interval_acc': []}

    # Outer cross-validation loop
    for train_indices, test_indices in outer_cross_validation.split(dataset, labels):
        
        current_training_part = dataset[train_indices]
        current_training_part_label = labels[train_indices]
        current_test_part = dataset[test_indices]
        current_test_part_label = labels[test_indices]
        
        # Step 1: Create and fit the scaler on the training data for this fold
        scaler = MinMaxScaler().fit(current_training_part)

        # Step 2: Transform the training data
        current_training_part = scaler.transform(current_training_part)

        # Step 3: Transform the test data using the same scaler
        current_test_part = scaler.transform(current_test_part)
        
        # KNN
        knn_grid_search = GridSearchCV(KNeighborsClassifier(), param_grid=knn_parameter_grid, refit=True, cv=inner_cross_validation, scoring=scoring_type)
        knn_grid_search.fit(current_training_part, current_training_part_label)
        knn_predicted = knn_grid_search.predict(current_test_part)
        knn_f1 = f1_score(current_test_part_label, knn_predicted, average="weighted")
        knn_accuracy = accuracy_score(current_test_part_label, knn_predicted)
        append_performance_metrics(knn_overall_performance, knn_f1, knn_accuracy,knn_grid_search.best_params_)

        # SVM
        svm_grid_search = GridSearchCV(SVC(), param_grid=svm_parameter_grid, refit=True, cv=inner_cross_validation, scoring=scoring_type)
        svm_grid_search.fit(current_training_part, current_training_part_label)
        svm_predicted = svm_grid_search.predict(current_test_part)
        svm_f1 = f1_score(current_test_part_label, svm_predicted, average="weighted")
        svm_accuracy = accuracy_score(current_test_part_label, svm_predicted)
        append_performance_metrics(svm_overall_performance, svm_f1, svm_accuracy,svm_grid_search.best_params_)

        # Decision Tree
        dt_grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid=decision_tree_parameter_grid, refit=True, cv=inner_cross_validation, scoring=scoring_type)
        dt_grid_search.fit(current_training_part, current_training_part_label)
        dt_predicted = dt_grid_search.predict(current_test_part)
        dt_f1 = f1_score(current_test_part_label, dt_predicted, average="weighted")
        dt_accuracy = accuracy_score(current_test_part_label, dt_predicted)
        append_performance_metrics(decision_tree_overall_performance, dt_f1, dt_accuracy,dt_grid_search.best_params_)
        
        # Random Forest
        rf_grid_search = GridSearchCV(RandomForestClassifier(), param_grid=random_forest_parameter_grid, refit=True, cv=inner_cross_validation_random_forest, scoring=scoring_type)
        rf_grid_search.fit(current_training_part, current_training_part_label)
        rf_predicted = rf_grid_search.predict(current_test_part)
        rf_f1 = f1_score(current_test_part_label, rf_predicted, average="weighted")
        rf_accuracy = accuracy_score(current_test_part_label, rf_predicted)
        append_performance_metrics(random_forest_overall_performance, rf_f1, rf_accuracy, rf_grid_search.best_params_)
    
        
    
    # Calculate and append confidence intervals for F1 and accuracy
    calculate_and_append_confidence_intervals(knn_overall_performance)
    calculate_and_append_confidence_intervals(svm_overall_performance)
    calculate_and_append_confidence_intervals(decision_tree_overall_performance)
    calculate_and_append_confidence_intervals(random_forest_overall_performance)
    
    # Print results
    print_performance_metrics(knn_overall_performance, "KNN")
    print_performance_metrics(svm_overall_performance, "SVM")
    print_performance_metrics(decision_tree_overall_performance, "Decision Tree")
    print_performance_metrics(random_forest_overall_performance, "Random Forest")



### Second Question ###

# Load the credit application dataset without one-hot encoding
dataset, labels = DataLoader.load_credit("data/credit.data")

# List the feature names 
feature_names = [
    "Status of existing checking account",
    "Duration in month",
    "Credit history",
    "Purpose",
    "Credit amount",
    "Savings account / bonds",
    "Present employment since",
    "Installment rate in percentage of disposable income",
    "Personal status and sex",
    "Other debtors / guarantors",
    "Present residence since",
    "Property",
    "Age in years",
    "Other installment plans",
    "Housing",
    "Number of existing credits at this bank",
    "Job",
    "Number of people being liable to provide maintenance for",
    "Telephone",
    "Foreign worker"
]


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)

# Scale the features using Min-Max Scaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train_scaled, y_train)

# Get feature importances
feature_importances = dt_classifier.feature_importances_

# Sorting 
enum_feature_importances = list(enumerate(feature_importances))
sorted_feature_importances = sorted(enum_feature_importances, key=lambda x: x[1], reverse=True)
top5_feature_indices = [index for index, _ in sorted_feature_importances[:5]]


# Print the top 5 most important features
print("Top 5 Most Important Features:")
for idx in top5_feature_indices:
    print(f"{feature_names[idx]} - Importance: {feature_importances[idx]}")
    
    

### Third Question ###

# Load one-hot encoded dataset
data_path = "data/credit.data"
dataset, labels = DataLoader.load_credit_with_onehot(data_path)


# Train an SVC
svm_classifier = SVC(kernel="rbf", C=1)
svm_classifier.fit(dataset, labels)

# Get support vectors
positive_class_sv_indices = svm_classifier.support_[svm_classifier.dual_coef_[0] > 0]
negative_class_sv_indices = svm_classifier.support_[svm_classifier.dual_coef_[0] < 0]

positive_class_svs = dataset[positive_class_sv_indices]
negative_class_svs = dataset[negative_class_sv_indices]

def reverse_one_hot(encoded_data, one_hot_lengths):
    reversed_data = []
    current_index = 0
    
    for row in encoded_data:
        reversed_row = []
        current_index = 0
        for length in one_hot_lengths:
            one_hot_part = row[current_index:current_index+length]
            
            if length == 1:
                # Numeric value
                reversed_row.append(one_hot_part[0])
            else:
                # One-hot encoding
                reversed_row.append(np.argmax(one_hot_part))
            
            current_index += length
        
        reversed_data.append(reversed_row)
    
    return np.array(reversed_data)

# The length of each feature in the one hot encoded data form 
one_hot_lengths = [4, 1, 5, 11, 1, 5, 5, 1, 5, 3,
                   1, 4, 1 ,3, 3, 1, 4, 1, 2, 2]
positive_class_svs_original = reverse_one_hot(positive_class_svs, one_hot_lengths)
negative_class_svs_original = reverse_one_hot(negative_class_svs, one_hot_lengths)

# Transpose the support vectors to group by feature
positive_class_svs_transposed = positive_class_svs_original.T
negative_class_svs_transposed = negative_class_svs_original.T

# Analyze each feature separately
for feature_index in range(positive_class_svs_transposed.shape[0]):
    feature_name = feature_names[feature_index]  # Replace with actual feature names if available
    positive_values = positive_class_svs_transposed[feature_index]
    negative_values = negative_class_svs_transposed[feature_index]

    # Count occurrences for each class
    positive_counts = Counter(positive_values)
    negative_counts = Counter(negative_values)

    # Print common values for positive class
    print(f"\nCommon values for {feature_name} in Positive Class:")
    for value, count in positive_counts.items():
        print(f"Value {value}: Count {count}")

    # Print common values for negative class
    print(f"\nCommon values for {feature_name} in Negative Class:")
    for value, count in negative_counts.items():
        print(f"Value {value}: Count {count}")
