import numpy as np
L = .1
    # HYPERPARAMETER Change L to change the learning rate
data = np.array([[1,0,3],[2,0,5],[1,1,4],[1,2,5]]) 
    # last value is the output
numData = data.shape[0]
    # Number of data points
numFeatures = data.shape[1]
    # including the 1 at the end
outputs = data[:, [numFeatures-1]].T
    #outputs
weights = np.zeros(numFeatures)
    # Initialize to zeroes
inputs = np.hstack([np.delete(data, numFeatures - 1, 1), np.ones((numData,1))])
    # Replace output (last entry) of data with a 1 to use as the bias
gradient = np.ones((1, numFeatures))
while gradient.dot(gradient.T) > 10 ** -30:
    # HYPERPARAMETER Change the value above to adjust precision as needed
    estimates = inputs.dot(weights.T).T
    # What the current regression predicts outputs to be given inputs
    loss = (estimates-outputs)
    for x in range (0, numFeatures):
        gradient[0,x] = np.sum(inputs[:, [x]].T * loss)      
    weights = weights - L * gradient
for x in range (0, numFeatures-1):
    print("Weight " + str(x) + ": " + str(weights[0,x]))
print("Bias: " + str(weights[0,numFeatures-1]))