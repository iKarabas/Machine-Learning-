import numpy as np

def f(x,w,b): # do not change this line!
     # implement the function f() here
     # x is a N-by-D numpy array
     # w is a D dimensional numpy array
     # b is a scalar
     # Should return three things:
     # 1) the output of the f function, as a N dimensional numpy array,
     # 2) gradient of f with respect to w,
     # 3) gradient of f with respect to b

     # In the case of a single-variable function, the gradient is just the partial derivative.
     z = np.dot(x, w) + b
     outputs = np.where((z <= 0) | (z > 1), 0, z)
     gradient_wrt_w = x
     gradient_wrt_b = 1
     return outputs, gradient_wrt_w, gradient_wrt_b



def l1loss(x,y,w,b): # do not change this line! 
     # implement the l1loss here
     # x is a N-by-D numpy array
     # y is a N dimensional numpy array
     # w is a D dimensional numpy array
     # b is a scalar
     # Should return three items:
     # 1) the L1 loss which is a scalar,
     # 2) the gradient of the loss with respect to w,
     # 3) the gradient of the loss with respect to b

    outputs = f(x, w, b)[0]
    differences = y - outputs
    differences = differences.reshape(-1, 1)  # turn the errors' array into an array of 1 sized arrays
    # Calculate L1 loss
    loss = np.mean(np.abs(differences))

    # Calculate signs 
    signs = np.sign(differences)

    # Gradient for w, - for the reverse direction
    gradient_wrt_w = -np.mean(x * signs, axis=0)

    # Gradient for b
    gradient_wrt_b = -np.mean(signs)

    return loss, gradient_wrt_w, gradient_wrt_b



def minimize_l1loss(x,y,w,b, num_iters=1000, eta=0.0001): 
     # do not change this line!
     # implement the gradient descent here
     # x is a N-by-D numpy array
     # y is a N dimensional numpy array
     # w is a D dimensional numpy array
     # b is a scalar
     # num_iters (optional) is number of iterations
     # eta (optional) is the step size for the gradient descent
     # Should return three items:
     # 1) final w
     # 2) final b
     # 3) list of loss values over iterations
     
     
     losses = []
     for i in range(num_iters):
         current_loss, current_gradient__wrt_w, current_gradient_wrt_b = l1loss(x, y, w, b)
         #updating the overall parameters according to algorithm using the determined step_size
         w -= eta * current_gradient__wrt_w
         b -= eta * current_gradient_wrt_b
         # record the losses for displaying them later
         losses.append(current_loss)
     return w, b, losses