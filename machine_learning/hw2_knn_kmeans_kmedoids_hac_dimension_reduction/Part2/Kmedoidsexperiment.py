
from Kmedoids import KMedoids
import pickle
import matplotlib.pyplot as plt
import numpy as np
from Distance import Distance

# Load the datasets
dataset1 = pickle.load(open("../data/part2_dataset_1.data", "rb"))
dataset2 = pickle.load(open("../data/part2_dataset_2.data", "rb"))



# FOR THE FIRST DATASET
# Define the range of K values to test, 1 to 10
k_values_1 = range(2, 11)

# Initialize lists to store average loss values
average_losses_1 = []

num_instances, num_features = dataset1.shape
distance_table = [[0 for _ in range(num_instances)] for _ in range(num_instances)]
# Fill the distance table
for i in range(num_instances):
    for j in range(num_instances):
        distance_table[i][j] = Distance.calculateCosineDistance(dataset1[i], dataset1[j])

# Run KMedoids for each K value and calculate average losses
for k in k_values_1:
    losses = []
    for i in range(10):  # Run 10 times for randomness
        kmedoids = KMedoids(dataset1, K=k)
        _, _, min_loss = kmedoids.run(distance_table)
        for j in range(9):
            kmedoids = KMedoids(dataset1, K=k)
            _, _, loss = kmedoids.run(distance_table)
            if loss < min_loss:
                min_loss = loss

        losses.append(min_loss)
    average_loss = np.mean(losses)
    std_dev = np.std(losses)
    confidence_interval = 1.96 * (std_dev / np.sqrt(10))  # 1.96 is the z-value for 95% confidence interval

    print(f"Dataset1, k = {k}, average loss: {average_loss:.4f}, for 95% confidence interval: ±{confidence_interval:.4f}")
    average_losses_1.append(average_loss)

# Plot the K versus Loss graph
plt.plot(k_values_1, average_losses_1, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Average Loss for Dataset1')
plt.title('KMedoids Elbow Method with dataset 1')
plt.show()




# FOR THE SECOND DATASET 
# Define the range of K values to test, 1 to 10
k_values_2 = range(2, 11)

# Initialize lists to store average loss values
average_losses_2 = []


num_instances, num_features = dataset2.shape
distance_table2 = [[0 for _ in range(num_instances)] for _ in range(num_instances)]
# Fill the distance table
for i in range(num_instances):
    for j in range(i,num_instances):
        distance_table2[i][j] = Distance.calculateCosineDistance(dataset2[i], dataset2[j])
        distance_table2[j][i] = distance_table2[i][j]

# Run KMedoids for each K value and calculate average losses
for k in k_values_2:
    losses = []
    for i in range(10):  # Run 10 times for randomness
        kmedoids = KMedoids(dataset2, K=k)
        _, _, min_loss = kmedoids.run(distance_table2)
        for j in range(9):
            kmedoids = KMedoids(dataset2, K=k)
            _, _, loss = kmedoids.run(distance_table2)
            if loss < min_loss:
                min_loss = loss

        losses.append(min_loss)
    average_loss = np.mean(losses)
    std_dev = np.std(losses)
    confidence_interval = 1.96 * (std_dev / np.sqrt(10))  # 1.96 is the z-value for 95% confidence interval

    print(f"Dataset2, k = {k}, average loss: {average_loss:.4f}, for 95% confidence interval: ±{confidence_interval:.4f}")
    average_losses_2.append(average_loss)

# Plot the K versus Loss graph
plt.plot(k_values_2, average_losses_2, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Average Loss for Dataset2')
plt.title('KMedoids Elbow Method with dataset 2')
plt.show()





# the original code without distance table 
""" 
from Kmedoids import KMedoids
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load the datasets
dataset1 = pickle.load(open("../data/part2_dataset_1.data", "rb"))
dataset2 = pickle.load(open("../data/part2_dataset_2.data", "rb"))

# FOR THE FIRST DATASET
# Define the range of K values to test, 1 to 10
k_values_1 = range(2, 11)

# Initialize lists to store average loss values
average_losses_1 = []
# Run KMedoids for each K value and calculate average losses
for k in k_values_1:
    losses = []
    for i in range(10):  # Run 10 times for randomness
        kmedoids = KMedoids(dataset1, K=k)
        _, _, min_loss = kmedoids.run()
        for j in range(9):
            kmedoids = KMedoids(dataset1, K=k)
            _, _, loss = kmedoids.run()
            if loss < min_loss:
                min_loss = loss

        losses.append(min_loss)
    average_loss = np.mean(losses)
    std_dev = np.std(losses)
    confidence_interval = 1.96 * (std_dev / np.sqrt(10))  # 1.96 is the z-value for 95% confidence interval

    print(f"Dataset1, k = {k}, average loss: {average_loss:.4f}, for 95% confidence interval: ±{confidence_interval:.4f}")
    average_losses_1.append(average_loss)

# Plot the K versus Loss graph
plt.plot(k_values_1, average_losses_1, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Average Loss for Dataset1')
plt.title('KMedoids Elbow Method with dataset 1')
plt.show()

# FOR THE SECOND DATASET 
# Define the range of K values to test, 1 to 10
k_values_2 = range(2, 11)

# Initialize lists to store average loss values
average_losses_2 = []

# Run KMedoids for each K value and calculate average losses
for k in k_values_2:
    losses = []
    for i in range(10):  # Run 10 times for randomness
        kmedoids = KMedoids(dataset2, K=k)
        _, _, min_loss = kmedoids.run()
        for j in range(9):
            kmedoids = KMedoids(dataset2, K=k)
            _, _, loss = kmedoids.run()
            if loss < min_loss:
                min_loss = loss

        losses.append(min_loss)
    average_loss = np.mean(losses)
    std_dev = np.std(losses)
    confidence_interval = 1.96 * (std_dev / np.sqrt(10))  # 1.96 is the z-value for 95% confidence interval

    print(f"Dataset2, k = {k}, average loss: {average_loss:.4f}, for 95% confidence interval: ±{confidence_interval:.4f}")
    average_losses_2.append(average_loss)

# Plot the K versus Loss graph
plt.plot(k_values_2, average_losses_2, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Average Loss for Dataset2')
plt.title('KMedoids Elbow Method with dataset 2')
plt.show()


"""