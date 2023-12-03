import torch
import torch.nn as nn
import numpy as np
import pickle
import matplotlib.pyplot as plt

###############################################################################
###################### DATA LOADING ############################################
x_train, y_train = pickle.load(open("data/mnist_train.data", "rb"))
x_validation, y_validation = pickle.load(open("data/mnist_validation.data", "rb"))
x_test, y_test = pickle.load(open("data/mnist_test.data", "rb"))


###############################################################################
######################3 DATA PREPROCESSING #####################################
x_train = x_train/255.0
x_train = x_train.astype(np.float32)

x_test = x_test / 255.0
x_test = x_test.astype(np.float32)

x_validation = x_validation/255.0
x_validation = x_validation.astype(np.float32)

# and converting them into Pytorch tensors in order to be able to work with Pytorch
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train).to(torch.long)

x_validation = torch.from_numpy(x_validation)
y_validation = torch.from_numpy(y_validation).to(torch.long)

x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test).to(torch.long)


#################################################################################################
############################## MODEL DEFINITION ################################################
class MLPModel(nn.Module):
    def __init__(self, input_size, output_size, num_of_hid_layers, num_of_neurons, activation_function):
        super(MLPModel, self).__init__()
        self.layers = nn.ModuleList()
        m_input_size = input_size

        for i in range(num_of_hid_layers):
            self.layers.append(nn.Linear(m_input_size, num_of_neurons))
            self.layers.append(activation_function)
            m_input_size = num_of_neurons

        self.output_layer = nn.Linear(m_input_size, output_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



############################################################################################
############################### ALL HYPERPARAMETER VARIATIONS ##############################
# number_of_neurons = [16, 32, 64, 128]
# number_of_hidden_layers = [1, 2, 3, 4]
# learning_rates = [0.01, 0.001, 0.0001]
# activation_functions = [nn.ReLU(), nn.Sigmoid(), nn.Tanh()]
# epochs = [30, 60, 180, 360]

#  since I am testing it on my laptop I will use a small variation
number_of_neurons = [16]
number_of_hidden_layers = [1, 2]
learning_rates = [0.01, 0.05, 0.1]
activation_functions = [nn.Sigmoid(), nn.Tanh()]
epochs = [10]

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# already determined variables
input_size = 784 # pixel count
output_size = 10 # digit count 
num_runs = 10
l2_lambda = 0.01  # You can adjust this hyperparameter

# END RESULTS that we want to find out
best_accuracy = 0.0
best_hyperparameters = {}
best_test_accuracy = 0.0
all_the_accuracies= []
number_of_combinations = len(number_of_neurons) * len(number_of_hidden_layers) * len(activation_functions) * len(epochs)

#################################################################################################################
########################################## HELPER FUNCTIONS #####################################################

## CALCULATING THE CONFIDENCE INTERVAL USING THE GIVEN FORMULA 
def confidence_interval(results):
    n = len(results)
    mean = np.mean(results)
    std_err = np.std(results, ddof=1) / np.sqrt(n)
    margin_err = 1.96 * std_err
    return mean - margin_err, mean + margin_err

## plotting graphs for later inspection 
def plot_3d_bar(number_of_hyperparameters, learning_rates, accuracy_scores):
    hyperparameter_values = np.arange(1, number_of_hyperparameters + 1)
    hyperparameter_values = np.concatenate([hyperparameter_values] * len(learning_rates))

    # Ensure learning_rates has the same length as hyperparameter_values
    learning_rates = np.repeat(learning_rates, number_of_hyperparameters)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot 3D bar
    ax.bar(learning_rates,accuracy_scores, hyperparameter_values,  zdir='y', width=0.001, color='b', alpha=0.7)

    # Set labels
    ax.set_xlabel('Learning Rates')
    ax.set_ylabel('Hyperparameter Values')
    ax.set_zlabel('Accuracy (%)')

    # Add color bar using a dummy scatter plot
    sc = ax.scatter([], [], [], c=[], cmap='viridis', marker='o')
    cbar = fig.colorbar(sc, ax=ax, label='Accuracy (%)')

    # Show the plot
    plt.show()
  



#####################################################################################################################
####################################### MAIN TESTING PART ###########################################################


for learning_rate in learning_rates:    
    for hidden_layer_count in number_of_hidden_layers:
        for neuron_count in number_of_neurons:
            for activation_function in activation_functions:
                for epoch_count in epochs:
                    
                    accuracy_values = []
                    # training each spesific combination
                    for run in range(num_runs):
                        
                        # initialization part 
                        model = MLPModel(input_size, output_size, hidden_layer_count, neuron_count, activation_function).to(device)
                        loss_function = nn.CrossEntropyLoss()
                        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                                             
                        # Training part
                        for  e in range(epoch_count):
                            # clear the previous gradient optimizations
                            optimizer.zero_grad()
                            # forward pass
                            predictions = model(x_train)
                            # compute the loss
                            loss_value = loss_function(predictions, y_train)
                            
                            # Apply L2 regularization to prevent overfitting
                            l2_reg = sum(torch.sum(param ** 2) for param in model.parameters())
                            loss_value += l2_lambda * l2_reg
                            
                            # backward pass (calculate gradient)
                            loss_value.backward()
                            # update parameters
                            optimizer.step()
                            
                        ## Validation part
                        with torch.no_grad():
                            # forward pass
                            predictions = model(x_validation)
                            probability_score_values = torch.softmax(predictions, dim = 1)
                            validation_loss = loss_function(predictions, y_validation)
                        
                        ## accuracy calculation and saving part 
                        true_class_indices = y_validation
                        predicted_class_indices = torch.argmax(predictions, dim=1)
                        current_accuracy = torch.sum(predicted_class_indices == true_class_indices).item() / len(true_class_indices) * 100   
                        accuracy_values.append(current_accuracy)
                    
                # Calculate mean accuracy and confidence interval
                mean_accuracy = np.mean(accuracy_values)
                conf_interval = confidence_interval(accuracy_values)
                all_the_accuracies.append(mean_accuracy)
                
                print(" ")
                print(f'Hyperparameters: "learning_rate": {learning_rate}, "hidden_layer_count": {hidden_layer_count}, "neuron_count": {neuron_count}, "activation_function": {activation_function}, "epoch_count": {epoch_count}')
                print(f'Mean Validation Accuracy: {mean_accuracy}, Confidence Interval: {conf_interval}')
                
                
                # Update best hyperparameters if current configuration is better
                if mean_accuracy > best_accuracy:
                    best_accuracy = mean_accuracy
                    best_hyperparameters = {
                        "learning_rate": learning_rate,
                        "hidden_layer_count": hidden_layer_count,
                        "neuron_count": neuron_count,
                        "activation_function": activation_function,
                        "epoch_count": epoch_count
                    }


########################################################################################################################
############################# TRAINING USING COMBINATION OF TRAIN AND VALIDATION. TESTING USING TEST SET  ##############
# Combine training and validation datasets for final training
train_x_combination = torch.cat([x_train, x_validation])
train_y_combination = torch.cat([y_train, y_validation])

# Use the best hyperparameters on the combined dataset and evaluate on the test set
test_accuracies = []

for run in range(num_runs):
    # initialization part 
    model = MLPModel(input_size, output_size, best_hyperparameters["hidden_layer_count"], best_hyperparameters["neuron_count"] , best_hyperparameters["activation_function"]).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                         
    # Training part
    for  e in range(epoch_count):
        # clear the previous gradient optimizations
        optimizer.zero_grad()
        # forward pass
        predictions = model(train_x_combination)
        # compute the loss
        loss_value = loss_function(predictions, train_y_combination)
        
        # Apply L2 regularization to prevent overfitting
        l2_reg = sum(torch.sum(param ** 2) for param in model.parameters())
        loss_value += l2_lambda * l2_reg
        
        # backward pass (calculate gradient)
        loss_value.backward()
        # update parameters
        optimizer.step()

    ## testing on testing data part
    with torch.no_grad():
        # forward pass
        predictions = model(x_test)
        probability_score_values = torch.softmax(predictions, dim = 1)
        validation_loss = loss_function(predictions, y_test)
    
    ## accuracy calculation and saving part
    true_class_indices = y_test
    predicted_class_indices = torch.argmax(predictions, dim=1)
    current_test_accuracy = torch.sum(predicted_class_indices == true_class_indices).item() / len(true_class_indices) * 100   
    test_accuracies.append(current_test_accuracy)

# Calculate mean accuracy and confidence interval for the test set
mean_test_accuracy = np.mean(test_accuracies)
test_conf_interval = confidence_interval(test_accuracies)
print(" ")
print(f'Best Hyperparameters: {best_hyperparameters}')
print(f'Mean Test Accuracy: {mean_test_accuracy}, Confidence Interval: {test_conf_interval}')


### lets plot learning rate vs accuracy graph for all the possible combinations
plot_3d_bar(number_of_combinations, learning_rates, all_the_accuracies)

