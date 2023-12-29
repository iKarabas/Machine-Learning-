import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split

# Data loading
dataset, labels = pickle.load(open("data/part2_dataset1.data", "rb"))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)

# Implement SVM
def create_and_train_svm(X, y, C, kernel):
    classifier = svm.SVC(C=C, kernel=kernel)
    classifier.fit(X, y)
    return classifier

# Plotting the outputs (decision boundaries)
def plot_decision_boundaries(X, y, clf, title):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="Paired")
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                         np.linspace(ylim[0], ylim[1], 50))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    plt.title(title)
    plt.show()

# Main part

# Run the SVM for different configurations and plot decision boundaries accordingly
# Linear Kernel ('linear'), Radial Basis Function (RBF) Kernel ('rbf')
# For each kernel type, one small and one large regularization parameter
configurations = [(2, 'linear'), (10, 'linear'), (2, 'rbf'), (10, 'rbf')]

# Test the configurations
for configuration in configurations:
    regularization_parameter, kernel = configuration
    classifier = create_and_train_svm(X_train, y_train, regularization_parameter, kernel)

    # Test the model on the testing set
    accuracy = classifier.score(X_test, y_test)
    print(f"Accuracy on the test set: {accuracy}")

    plot_decision_boundaries(X_train, y_train, classifier, f"SVM Decision Boundary Plot - when C={regularization_parameter}, Kernel={kernel}")
