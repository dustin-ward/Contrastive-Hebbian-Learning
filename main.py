import numpy as np
import seaborn as sb
import random
import CHL

# MNIST dataset
layerSizes = (784,16,16,10)
print("Reading dataset")
with np.load('mnist.npz') as data:
  trainingInputs = data['training_images']
  trainingOutputs = data['training_labels']

trainingData = list(zip(trainingInputs, trainingOutputs))
testingData = trainingData[:10000]
trainingData = trainingData[10000:]

# Define and train network
net = CHL.CHLNeuralNetwork(layerSizes)
print("Training network...")
net.train(trainingData, testingData, 5, .1)

# Results
x = random.randint(0, 10000)
dataPoint = testingData[x]
after = net.predict(dataPoint[0])

img = np.reshape(dataPoint[0], (28,28))
sb.heatmap(img)

print("Actual Number:")
print(np.argmax(dataPoint[1]))
print("Network Prediction:")
print(np.argmax(after))
