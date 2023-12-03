import numpy as np
import pickle, gzip
import hw1_sol as hw1
import sys
from matplotlib import pyplot as plt
from matplotlib import colors

# helper function to plot decision boundary on 2D data
def plot_boundary_on_data(X,Y,pred_func):
    # determine canvas borders
    mins = np.amin(X,0); 
    mins = mins - 0.1*np.abs(mins);
    maxs = np.amax(X,0); 
    maxs = maxs + 0.1*maxs;

    ## generate dense grid
    xs,ys = np.meshgrid(np.linspace(mins[0],maxs[0],300), 
            np.linspace(mins[1], maxs[1], 300));


    # evaluate model on the dense grid 
    Z = pred_func(np.c_[xs.flatten(), ys.flatten()]);
    Z = Z.reshape(xs.shape)

    # Plot the contour and training examples
    plt.contourf(xs, ys, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=50,
            cmap=colors.ListedColormap(['orange', 'blue']))
    #plt.show()


# Read in training data
with gzip.open(sys.argv[1]) as f:
    data, labels = pickle.load(f, encoding='latin1')
    labels = np.array(labels, dtype=np.int8)

# (Random) Initial values for w and b
w0 = .001*np.random.randn(data.shape[1]) 
b0 = 0.001*np.random.randn(1)

# Optimization
w,b,losses = hw1.minimize_l1loss(data, labels, w0,b0, int(sys.argv[3]),
        float(sys.argv[4]))

plt.plot(losses)
plt.xlabel('iterations')
plt.ylabel('loss')
plt.savefig('losses.png')

# Show the result of training
plt.figure()
if w.size==2:
    plot_boundary_on_data(data, labels, lambda x: hw1.f(x,w,b)[0]>0.5)

if w.size==784: # special to MNIST
    plt.imshow(w.reshape(28,28));
    #plt.show()

plt.savefig('weights.png')


# Test on test data
with gzip.open(sys.argv[2]) as f:
    test_data, test_labels = pickle.load(f, encoding='latin1')
    test_labels = np.array(test_labels, dtype=np.int8)

yhat = hw1.f(test_data, w, b)[0]>.5
print(np.mean(yhat==test_labels)*100, "% of test examples classified correctly.")