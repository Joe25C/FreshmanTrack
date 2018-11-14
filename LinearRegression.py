import numpy as np
L = .1
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
for y in range (0,2000):
    print("INPUTS")
    print(inputs)
    print(weights)
    estimates = inputs.dot(weights.T).T
    # What the current regression predicts outputs to be given inputs
    print("Estimates: ")
    print(estimates)
    print("Outputs: ")
    print(outputs)
    loss = (estimates-outputs)
    print("Loss: ")
    print(loss)
    lossSum = np.sum(loss)
    print(lossSum)
    gradient = np.zeros((1, numFeatures))
    print(gradient)
    print 
    for x in range (0, numFeatures):
        print("TEST")
        print(inputs[:, [x]].T)
        print(loss)
        print(inputs[:, [x]].T * loss)
        gradient[0,x] = np.sum(inputs[:, [x]].T * loss)      
    print(gradient)
    weights = weights - L * gradient
print ("WEIGHTS")
print (weights)
for x in range (0, numFeatures-1):
    print("Weight " + str(x) + ": " + str(weights[0,x]))
print("Bias: " + str(weights[0,numFeatures-1]))
        
