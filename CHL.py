import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# Neural Network with contrastive hebbian learning
class CHLNeuralNetwork:

  # Constructor Function
  # layerSizes: List of integers describing the size of each sequential layer
  def __init__(self, layerSizes):
    self.layerSizes = layerSizes
    self.L = len(layerSizes)-1

    self.B = [np.zeros((x,1)) for x in layerSizes]
    weightShapes = [(a,b) for a,b in zip(layerSizes[:-1],layerSizes[1:])]
    self.W = [np.zeros((1,1))] + [np.random.standard_normal(s) / s[1] ** .5 for s in weightShapes]

  # Prediction Function
  # a: Input data to feed through layers
  def predict(self, x):
    for k in range(1,self.L+1):
      x = sigmoid(np.matmul(self.W[k].T,x) + self.B[k])
    return x

  # Printing helper function
  def printAccuracy(self, trainingData, testingData):
    correct = 0
    for x,y in trainingData:
      prediction = self.predict(x)
      if np.argmax(prediction) == np.argmax(y[:,0]):
        correct += 1
    print('Training Acc: {0}/{1} ({2}%)'.format(correct, len(trainingData), (correct / len(trainingData)) * 100))

    correct = 0
    for x,y in testingData:
      prediction = self.predict(x)
      if np.argmax(prediction) == np.argmax(y[:,0]):
        correct += 1
    print('Testing Acc: {0}/{1} ({2}%)'.format(correct, len(testingData), (correct / len(testingData)) * 100))

  # Contrastive Hebbian Learning algorithm
  def chl(self, x, y, learningRate, feedback):
    # Free phase
    Xf = [np.zeros((x,1)) for x in self.layerSizes]
    Xf[0] = x
    for _ in range(50):
      for k in range(1,self.L):
        Xf[k] = sigmoid(np.matmul(self.W[k].T,Xf[k-1]) + (feedback * np.matmul(self.W[k+1],Xf[k+1])) + self.B[k])
      Xf[self.L] = sigmoid(np.matmul(self.W[self.L].T,Xf[self.L-1]) + self.B[self.L])

    # Clamped phase
    Xc = [np.zeros((x,1)) for x in self.layerSizes]
    Xc[0] = x
    Xc[self.L] = y
    for _ in range(50):
      for k in range(1, self.L):
        Xc[k] = sigmoid(np.matmul(self.W[k].T,Xc[k-1]) + (feedback * np.matmul(self.W[k+1],Xc[k+1])) + self.B[k])

    # Update Weights and Biases
    for k in range(1,self.L+1):
      self.W[k] += learningRate * (feedback**(k-self.L)) * (np.matmul(Xc[k],Xc[k-1].T) - np.matmul(Xf[k],Xf[k-1].T)).T
      self.B[k] += learningRate * (feedback**(k-self.L)) * (Xc[k] - Xf[k])

  # Train NN via contrastive hebbian learning
  def train(self, trainingData, testingData, epochs, learningRate):
    for j in range(epochs):
      # Shuffle training data to provide different order for each epoch
      # random.shuffle(trainingData)

      for x,y in trainingData:
        self.chl(x, y, learningRate, 0.5)

      # Test current accuracy
      print("Epoch {0} complete".format(j+1))
      self.printAccuracy(trainingData, testingData)

  # Modify for debugging purposes
  def test(self, x,y):
    self.chl(x,y,1,1)
    self.chl(x,y,1,1)
