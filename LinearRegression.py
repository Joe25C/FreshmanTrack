import numpy as np
L = .001
data = np.array([[1,0,2],[2,0,4],[1,1,3],[1,2,4]]) 
    # last value is the output
numData = data.shape[0]
numFeatures = data.shape[1]
    # including the 1 at the end
outputs = data[:, [numFeatures-1]].T
    #outputs
print(outputs)
weights = np.zeros(numFeatures)
    # Initialize to zeroes
inputs = np.hstack([np.delete(data, numFeatures - 1, 1), np.ones((numData,1))])
    # Replace output (last entry) of data with a 1 to use as the bias
print("NumData:")
print(numData)
print("NumFeatures:")
print(numFeatures)
print("Inputs:")
print(inputs)
for y in range (0,600):
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
    # this returns a row and not a scalar, idk why but I'll m ake it work
    print("Loss: ")
    print(loss)
    lossSum = np.sum(loss)
    print(lossSum)
    gradient = np.zeros((1, numFeatures))
    print(gradient)
    for x in range (0, numFeatures):
        gradient[0,x] = np.sum(inputs[:, [x]].T) * lossSum         
    print(gradient)
    weights = weights - L * gradient
    y+=1
print ("WEIGHTS")
print (weights)
